# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from utils.flag_center import FLAGS

import tensorflow as tf


def create_optimizer_cy(loss,
                     init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     trainable_vars=None,
                     lm_loss = None,
                     encoder_loss = None,
                     use_lazy_adam = True,
                     pretrain_step = 0):
  '''
  learning rate decay follows:
  1. during the warm-up stage, its linear from zero to "init_lr"
  2. during the trainig stage, learning_rate = learning_rate * (global_step) ** -0.5
  according to the transformer paper, the "init_lr" should be set to (hidden_size * num_warmup_steps) ** -0.5
  num_warmup_steps is 0.1 of num_train_steps
  '''
  if num_warmup_steps is None or num_warmup_steps < 1:
    raise ValueError("num_warmup_steps MUST be set")

  global_step = tf.train.get_or_create_global_step()
  global_steps_int = tf.cast(global_step, tf.int32) - tf.constant(pretrain_step, dtype=tf.int32)
  global_steps_float = tf.cast(global_steps_int, tf.float32)
  # initial learning rate
  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
  warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
  warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
  # Implements linear decay of the learning rate.
  learning_rate *= (warmup_steps_float /(0.00001 + global_steps_float)) ** 0.5

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  warmup_percent_done = global_steps_float / warmup_steps_float
  warmup_learning_rate = init_lr * warmup_percent_done

  is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
  learning_rate = (
      (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
  # use build-in Adam
  if use_lazy_adam == True:
    optimizer =  tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
  else:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

  # any var list provided?
  if trainable_vars is None:
    optimizer = optimizer.minimize(loss, global_step=global_step)
  else:
    optimizer = optimizer.minimize(loss, global_step=global_step, var_list=trainable_vars)

  tf.summary.scalar('lr', learning_rate)
  tf.summary.scalar("loss", loss)
  # tf.summary.scalar("lm_loss", lm_loss)
  tf.summary.scalar("global_step", global_step)
  summaries = tf.summary.merge_all()

  return optimizer, summaries, learning_rate

def create_optimizer_cy_diff(loss,
                     init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     num_init_steps,
                     trainable_vars=None,
                     half_trainable_vars=None,
                     lm_loss = None,
                     encoder_loss = None,
                     use_lazy_adam = True,
                     pretrain_step = 0):
  '''
  learning rate decay follows:
  1. during the warm-up stage, its linear from zero to "init_lr"
  2. during the trainig stage, learning_rate = learning_rate * (global_step) ** -0.5
  according to the transformer paper, the "init_lr" should be set to (hidden_size * num_warmup_steps) ** -0.5
  num_warmup_steps is 0.1 of num_train_steps
  '''
  if num_warmup_steps is None or num_warmup_steps < 1:
    raise ValueError("num_warmup_steps MUST be set")

  global_step = tf.train.get_or_create_global_step()
  global_steps_int = tf.cast(global_step, tf.int32) - tf.constant(pretrain_step, dtype=tf.int32)
  global_steps_float = tf.cast(global_steps_int, tf.float32)
  # initial learning rate
  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
  warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
  warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
  num_init_steps_int = tf.constant(num_init_steps, tf.int32)
  # Implements linear decay of the learning rate.
  learning_rate *= (warmup_steps_float /(0.00001 + global_steps_float)) ** 0.5

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  warmup_percent_done = global_steps_float / warmup_steps_float
  warmup_learning_rate = init_lr * warmup_percent_done

  is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
  learning_rate = (
      (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
  half_learning_rate = tf.cond(global_steps_int < num_init_steps_int, lambda :tf.constant(0.0), lambda : learning_rate / 2.0)
  # use build-in Adam
  if use_lazy_adam == True:
    optimizer =  tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
    optimizer_2 =  tf.contrib.opt.LazyAdamOptimizer(learning_rate=half_learning_rate)
  else:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer_2 = tf.train.AdamOptimizer(learning_rate=half_learning_rate)

  # any var list provided?
  if trainable_vars is None and half_trainable_vars is None:
    all_train_op = optimizer.minimize(loss, global_step=global_step)
  elif len(half_trainable_vars) == 0:
    all_train_op = optimizer.minimize(loss, global_step=global_step, var_list=trainable_vars)
  else:
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=trainable_vars)
    train_op_2 = optimizer_2.minimize(loss, global_step=global_step, var_list=half_trainable_vars)
    all_train_op = tf.group(train_op, train_op_2)

  tf.summary.scalar('lr', learning_rate)
  tf.summary.scalar("loss", loss)
  # tf.summary.scalar("lm_loss", lm_loss)
  tf.summary.scalar("global_step", global_step)
  summaries = tf.summary.merge_all()

  return all_train_op, summaries, learning_rate, half_learning_rate


def create_optimizer_rocket(loss,
                     init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     trainable_vars=None,
                     lm_loss = None,
                     encoder_loss = None,
                     boost_loss = None,
                     light_loss = None,
                     use_lazy_adam = True,
                     light_trainable_vars=None,
                     distance = None):
  '''
  learning rate decay follows:
  1. during the warm-up stage, its linear from zero to "init_lr"
  2. during the trainig stage, learning_rate = learning_rate * (global_step) ** -0.5
  according to the transformer paper, the "init_lr" should be set to (hidden_size * num_warmup_steps) ** -0.5
  num_warmup_steps is 0.1 of num_train_steps
  '''
  if num_warmup_steps is None or num_warmup_steps < 1:
    raise ValueError("num_warmup_steps MUST be set")

  global_step = tf.train.get_or_create_global_step()
  global_steps_int = tf.cast(global_step, tf.int32)
  global_steps_float = tf.cast(global_steps_int, tf.float32)
  # initial learning rate
  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
  warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
  warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
  # Implements linear decay of the learning rate.
  learning_rate *= (warmup_steps_float /(0.00001 + global_steps_float)) ** 0.5

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  warmup_percent_done = global_steps_float / warmup_steps_float
  warmup_learning_rate = init_lr * warmup_percent_done

  is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
  learning_rate = (
      (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
  # use build-in Adam
  if use_lazy_adam == True:
    optimizer =  tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
  else:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

  # any var list provided?
  if trainable_vars is None:
    grads_and_vars = optimizer.compute_gradients(loss)
  else:
    grads_and_vars = optimizer.compute_gradients(loss, var_list=trainable_vars)

  light_grads_and_vars = optimizer.compute_gradients(distance, var_list=light_trainable_vars)

  train_op = optimizer.apply_gradients(grads_and_vars)
  light_train_op = optimizer.apply_gradients(light_grads_and_vars, global_step=global_step)

  all_train_op = tf.group(train_op, light_train_op)

  tf.summary.scalar('lr', learning_rate)
  tf.summary.scalar("loss", loss)
  tf.summary.scalar("boost_loss", boost_loss)
  tf.summary.scalar("light_loss", boost_loss)
  tf.summary.scalar("distance", distance)
  # tf.summary.scalar("lm_loss", lm_loss)
  tf.summary.scalar("encoder_loss", encoder_loss)
  tf.summary.scalar("global_step", global_step)
  summaries = tf.summary.merge_all()

  return all_train_op, summaries, learning_rate

def create_optimizer_asgd(loss,
                     init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     trainable_vars=None,
                     lm_loss = None,
                     encoder_loss = None,
                     use_lazy_adam = True,
                     pretrain_step = 0):
  '''
  learning rate decay follows:
  1. during the warm-up stage, its linear from zero to "init_lr"
  2. during the trainig stage, learning_rate = learning_rate * (global_step) ** -0.5
  according to the transformer paper, the "init_lr" should be set to (hidden_size * num_warmup_steps) ** -0.5
  num_warmup_steps is 0.1 of num_train_steps
  '''
  global_step = tf.train.get_or_create_global_step()
  global_steps_int = tf.cast(global_step, tf.int32)
  global_steps_float = tf.cast(global_steps_int, tf.float32)

  # global_step = tf.Variable(0, trainable=False)

  def triangular_lr(current_step):
    """cyclic learning rate - exponential range."""
    step_size = 700
    base_lr = 0.00001
    max_lr = 0.00003

    cycle = tf.floor(1 + current_step / (2 * step_size))
    x = tf.abs(current_step / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * tf.maximum(0.0, tf.cast((1.0 - x), dtype=tf.float32)) * (0.99999 ** tf.cast(
      current_step,
      dtype=tf.float32))
    return lr

  # cyclic learning rate
  learning_rate = triangular_lr(global_steps_float)

  # Optimizer
  opt = tf.train.AdamOptimizer(learning_rate, beta1=0.7, beta2=0.99)

  # Gradients
  grads, vs = zip(*opt.compute_gradients(loss))
  grads, _ = tf.clip_by_global_norm(grads, 5)
  train_op = opt.apply_gradients(zip(grads, vs), global_step=global_step)

  tf.summary.scalar('lr', learning_rate)
  tf.summary.scalar("loss", loss)
  # tf.summary.scalar("lm_loss", lm_loss)
  tf.summary.scalar("global_step", global_step)
  summaries = tf.summary.merge_all()

  return train_op, summaries, learning_rate


#---------------------------------------------------------------------------------------------------------------
# desperated
#---------------------------------------------------------------------------------------------------------------

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
  '''Noam scheme learning rate decay
  init_lr: initial learning rate. scalar.
  global_step: scalar.
  warmup_steps: scalar. During warmup_steps, learning rate increases
      until it reaches init_lr.
  '''
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def create_optimizer(loss,
                         init_lr,
                         num_train_steps,
                         num_warmup_steps,
                         use_tpu=False,
                         lm_loss = None,
                         encoder_loss = None):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.

  if not FLAGS.is_standalone:
    learning_rate = noam_scheme(learning_rate, global_step, num_warmup_steps)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=global_step)
    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("lm_loss", lm_loss)
    tf.summary.scalar("encoder_loss", encoder_loss)
    tf.summary.scalar("global_step", global_step)
    summaries = tf.summary.merge_all()
    return train_op, summaries
 

def create_optimizer_cy_finetune(loss,
                     init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     trainable_vars=None,
                     trainable_vars_all = None,
                     encoder_loss = None,
                     use_lazy_adam = True):
  '''
  learning rate decay follows:
  1. during the warm-up stage, its linear from zero to "init_lr"
  2. during the trainig stage, learning_rate = learning_rate * (global_step) ** -0.5
  according to the transformer paper, the "init_lr" should be set to (hidden_size * num_warmup_steps) ** -0.5
  num_warmup_steps is 0.1 of num_train_steps
  '''
  if num_warmup_steps is None or num_warmup_steps < 1:
    raise ValueError("num_warmup_steps MUST be set")

  global_step = tf.train.get_or_create_global_step()
  global_steps_int = tf.cast(global_step, tf.int32)
  global_steps_float = tf.cast(global_steps_int, tf.float32)
  # initial learning rate
  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
  warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
  warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
  # Implements linear decay of the learning rate.
  learning_rate *= (warmup_steps_float /(0.00001 + global_steps_float)) ** 0.5

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  warmup_percent_done = global_steps_float / warmup_steps_float
  warmup_learning_rate = init_lr * warmup_percent_done

  is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
  learning_rate = (
      (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
  # use build-in Adam
  if use_lazy_adam == True:
    optimizer =  tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
  else:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

  # any var list provided?
  finetune_steps = FLAGS.num_train_steps + 1000
  finetune_steps_int = tf.constant(finetune_steps, dtype=tf.int32)
  if trainable_vars is None:
    optimizer = optimizer.minimize(loss, global_step=global_step)
  else:
    # if num_train_steps < 5000:
    #   optimizer = optimizer.minimize(loss, global_step=global_step, var_list=trainable_vars)
    # else:
    #   optimizer = optimizer.minimize(loss, global_step=global_step, var_list=trainable_vars_all)
    optimizer = tf.cond(global_steps_int < finetune_steps_int, lambda: optimizer.minimize(loss, global_step=global_step, var_list=trainable_vars),
                        lambda: optimizer.minimize(loss, global_step=global_step, var_list=trainable_vars_all))

  tf.summary.scalar('lr', learning_rate)
  tf.summary.scalar("loss", loss)
  tf.summary.scalar("encoder_loss", encoder_loss)
  tf.summary.scalar("global_step", global_step)
  summaries = tf.summary.merge_all()

  return optimizer, summaries, learning_rate
