# coding: utf-8
import math
import re
import os
import json
import linecache
import collections
import tensorflow as tf
from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib
import optimization
import numpy as np

from phone2slot import phone_to_slot
from utils.flag_center import FLAGS
from utils.features import file_based_input_fn_builder_transformer_bridge_mid_decode

## 打印工作路径，在出现系统环境问题的时候，可以帮助诊断文件是否缺失
w_dir = os.getcwd()
print('\n\nWORKDIR is ' + w_dir + '\n\n')
list_dir = os.listdir('./')
print("\n\n")
print(str(list_dir))
print("\n\n")


## fetch intersection vars pre-trained
def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


## 构造模型的计算图
def model_fn_builder():
  def model_fn(features, labels, mode, params):

    phone_tensor,  phone_tensor_mask, phone_masked_groundtruth, phone_masked_positions, valid_pos = features
    decoder_input, decoder_groundtruth, m2l_decoder_input, m2l_decoder_groundtruth, m2r_decoder_input, m2r_decoder_groundtruth, mid_words = labels

    model = phone_to_slot(config_file=FLAGS.config_file,
                          is_training=True,
                          phone_tensor=phone_tensor_mask,
                          phone_masked_positions=phone_masked_positions,
                          phone_masked_groundtruth=phone_masked_groundtruth,
                          decoder_input=decoder_input,
                          decoder_groundtruth=decoder_groundtruth,
                          m2l_decoder_input=m2l_decoder_input,
                          m2l_decoder_groundtruth=m2l_decoder_groundtruth,
                          m2r_decoder_input=m2r_decoder_input,
                          m2r_decoder_groundtruth=m2r_decoder_groundtruth,
                          mid_words = mid_words,
                          )

    tvars = tf.trainable_variables()
    # fixed LM
    #ckpt_dir_or_file = params['fixed_lm'].split(",")
    #forward_ckpt_dir = ckpt_dir_or_file[0]
    #backward_ckpt_dir = ckpt_dir_or_file[1]
    # (assignment_map, forward_initialized_variable_names) \
    #  = get_assignment_map_from_checkpoint(tvars=tvars, init_checkpoint=forward_ckpt_dir)
    # tf.train.init_from_checkpoint(ckpt_dir_or_file=forward_ckpt_dir,
    #                              assignment_map=assignment_map)
    # (assignment_map, backward_initialized_variable_names) \
    #     = get_assignment_map_from_checkpoint(tvars=tvars, init_checkpoint=backward_ckpt_dir)
    # tf.train.init_from_checkpoint(ckpt_dir_or_file=backward_ckpt_dir,
    #                               assignment_map=assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      # if var.name in forward_initialized_variable_names or var.name in backward_initialized_variable_names:
      #   init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s", var.name, var.shape)

    trainable_vars = []
    for var in tvars:
      name = var.name
      # if name in forward_initialized_variable_names or name in backward_initialized_variable_names:
      #   continue
      trainable_vars.append(var)
    tf.logging.info('%d of %d variables are trainable' % (len(trainable_vars), len(tvars)))


    profiler_hook = tf.train.ProfilerHook(
        save_steps=5000,
        output_dir=get_model_output_dir(FLAGS.buckets),
        show_memory=True
    )

    output_spec = None

    if mode == tf.estimator.ModeKeys.TRAIN:
      tf.logging.info(str(params))

      train_op, summeris, lr = optimization.create_optimizer_cy(
          loss=model.loss,
          init_lr=params['learning_rate'],
          num_train_steps=params['num_train_steps'],
          num_warmup_steps=params['num_warmup_steps'],
          #lm_loss=model.lm_loss,
          encoder_loss=model.loss,
          trainable_vars = trainable_vars)

      summary_hook = tf.train.SummarySaverHook(
          save_secs=20,
          output_dir=get_model_output_dir(FLAGS.buckets),
          scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))

      logging_hook = tf.train.LoggingTensorHook(
          {
            "total_loss": model.loss,
            "bridge_loss": model.bridge_loss,
            "back_bridge_loss": model.back_bridge_loss,
            "m2l_bridge_loss": model.m2l_bridge_loss,
            "m2r_bridge_loss": model.m2r_bridge_loss,
            "midwords_loss": model.midwords_loss,
            "zlearning_rate": lr,
            # ,"phone_tensor": phone_tensor
          },
          every_n_iter=20)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=model.loss,
          train_op=train_op,
          training_hooks=[logging_hook])

    return output_spec

  return model_fn


def make_distributed_info_without_evaluator():
  worker_hosts = FLAGS.worker_hosts.split(",")
  ps_hosts = FLAGS.ps_hosts.split(",")
  if len(worker_hosts) > 1:
    cluster = {"chief": [worker_hosts[0]],
               "worker": worker_hosts[1:],
               "ps": ps_hosts}
  else:
    cluster = {"chief": [worker_hosts[0]],
               "ps": ps_hosts}

  if FLAGS.job_name == "worker":
    if FLAGS.task_index == 0:
      task_type = "chief"
      task_index = 0
    else:
      task_type = "worker"
      task_index = FLAGS.task_index - 1
  else:
    task_type = "ps"
    task_index = FLAGS.task_index
  return cluster, task_type, task_index


def dump_into_tf_config(cluster, task_type, task_index):
  os.environ['TF_CONFIG'] = json.dumps(
      {'cluster': cluster,
       'task': {'type': task_type, 'index': task_index}})


def load_tfrecord_file_list(sheet_list):
  '''
  加载训练数据
  '''
  sheet = list()
  items = linecache.getlines(sheet_list)
  for item in items:
    item = item.strip()
    sheet.append(item)
  return sheet

def calculate_steps():
  if FLAGS.num_train_steps != 0:
    return FLAGS.num_train_steps, FLAGS.num_warmup_steps
  total = FLAGS.num_train_sample
  num_train_steps = int(
      math.ceil(total * FLAGS.num_train_epochs / FLAGS.train_batch_size) + 1)
  num_warmup_steps = int(math.ceil(num_train_steps * FLAGS.warmup_ratio))
  num_warmup_steps = min([FLAGS.num_warmup_steps, num_warmup_steps])
  return num_train_steps, num_warmup_steps


def get_model_output_dir(buckets):
  user_name = FLAGS.user_name
  model_name = FLAGS.model_dir
  if user_name == "":
    return buckets
  else:
    return os.path.join(os.path.join(buckets, user_name), model_name)


def main(unused_argv):
  '''
  多卡训练入口函数
  '''

  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.is_standalone:
    if FLAGS.distribution_mode == 'mjmk_a':
      cluster, task_type, task_index = make_distributed_info_without_evaluator()
      dump_into_tf_config(cluster, task_type, task_index)

  num_train_steps, num_warmup_steps = calculate_steps()

  tf.logging.info(FLAGS)
  tf.logging.info("Number of training steps: " + str(num_train_steps))
  tf.logging.info("Number of warm-up steps: " + str(num_warmup_steps))

  # 构建 Estimator
  model_fn = model_fn_builder()

  if not FLAGS.is_standalone:
    distribution = False
    if FLAGS.distribution_mode == 'sjmk_s':

      cross_tower_ops = cross_tower_ops_lib.AllReduceCrossTowerOps('nccl')
      distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=4,
                                                            cross_tower_ops=cross_tower_ops,
                                                            all_dense=True)
    elif FLAGS.distribution_mode == 'mjmk_a':
      distribution = tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=1)

    run_config = tf.estimator.RunConfig(train_distribute=distribution, model_dir=get_model_output_dir(FLAGS.buckets),
                                        save_checkpoints_steps=FLAGS.save_checkpoints_steps)
  else:
    run_config = tf.estimator.RunConfig(model_dir=FLAGS.ckpt_dir,
                                        save_checkpoints_steps=FLAGS.save_checkpoints_steps)

  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     config=run_config,
                                     params={'init_checkpoint': FLAGS.init_checkpoint,
                                             'learning_rate': FLAGS.learning_rate,
                                             'num_warmup_steps': num_warmup_steps,
                                             'num_train_steps': num_train_steps,
                                             'train_batch_size': FLAGS.train_batch_size,
                                             'eval_batch_size': FLAGS.eval_batch_size,
                                             "fixed_lm": FLAGS.fixed_lm
                                             })

  if FLAGS.do_train:
    tf.logging.info("\n\n 开始模型训练\n\n")

    input_file_list = load_tfrecord_file_list(sheet_list=FLAGS.train_sheet_file_list)
    tf.logging.info("\n\ninput file list")
    input_files_tf = []
    for ele in input_file_list:
      input_files_tf.extend(tf.gfile.Glob(ele))
      tf.logging.info(ele)
    tf.logging.info("---------------\n\n")

    train_input_fn = file_based_input_fn_builder_transformer_bridge_mid_decode(input_file_list=input_files_tf,
                                                                    mode=tf.estimator.ModeKeys.TRAIN,
                                                                    drop_remainder=True,
                                                                    num_cpu_threads=16)
    if not FLAGS.is_standalone:
      train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
      if FLAGS.distribution_mode == 'sjmk_s':
        # estimator.train(estimator, train_spec)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
      elif FLAGS.distribution_mode == 'mjmk_a':
        eval_spec = tf.estimator.EvalSpec(input_fn=train_input_fn)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    else:
      estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


if __name__ == '''__main__''':
  tf.app.run()
