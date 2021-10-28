from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import model as modeling
from utils.tritree import build_trie_tree_from_interspeech_file, build_trie_tree_reverse_from_interspeech_file
from utils.tri_decode import get_hypotheses_beam_mid, load_vocab, load_bin_data_test_phone2pinyin
import tensorflow as tf
import numpy as np
import pickle
import time
import collections
import re
import math

flags = tf.flags
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "config_file", "./config.json",
    "The config json file corresponding to the transformer model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "testing_file", "test_data_phoneme/vndc_test_human_phoneme.pkl",
    "testing file")

flags.DEFINE_string(
    "ckpt", "ckpt/speech2slot_0925",
    "checkpoint file path")

flags.DEFINE_string(
    "place_whole_name_dict", "AM/data/VNDC/slot_dict.txt",
    "phones for all songs")

flags.DEFINE_string(
    "target_vocab", "AM/voc/voc_target_pinyin.txt",
    "vocabulary file path")

flags.DEFINE_string(
    "source_vocab", "AM/voc/voc_source_pinyin.txt",
    "vocabulary file path")


flags.DEFINE_string("beam_search_res", "beam_search_res.txt",
                    "beam search result")

flags.DEFINE_string("test_dir", "./",
                    "test result")

flags.DEFINE_integer("test_batch_size", 8, "Total batch size for testing.")

flags.DEFINE_integer("source_max_length", 40,
                     "max sequence length in encoder part")

flags.DEFINE_integer("dest_max_length", 10,
                     "max sequence length in decoder part")

flags.DEFINE_integer("num_beam", 10,
                     "beam search size")

tf.flags.DEFINE_string("bin_test", default="", help="bin test data")

tf.flags.DEFINE_string(name="embedding_file", default=None, help="embedding file")

tf.flags.DEFINE_integer("forward_step", default=8, help="forward-decoder max decoder step")


def load_testing_data():
  if FLAGS.bin_test == "":
    with open(FLAGS.testing_file, 'rb') as inf:
      test_data = pickle.load(inf)
      #for different pickle format
      # try:
      #   if len(test_data) < 4:
      #     test_data = test_data[0]
      # except:
      #   test_data = test_data[0]
    if os.path.exists(FLAGS.testing_file + '.oov'):
      with open(FLAGS.testing_file + '.oov', 'rb') as inf:
        oov = pickle.load(inf)
    else:
      oov = list()
  else:
    test_data, oov = load_bin_data_test_phone2pinyin(FLAGS.bin_test, FLAGS.dest_max_length)
    # test_data, oov = load_bin_data_test_pinpyin2pinyin(FLAGS.bin_test, FLAGS.dest_max_length)
  return test_data, oov


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

class phone_to_slot(object):
  '''
  phone to slot model
  based on transformer
  '''

  def __init__(self,
               config_file,
               test_batch_size,
               source_max_length,
               dest_max_length,
               max_masked_length,
               is_training,
               embedding_file=None):
    '''
    construct model
    '''
    # read config file
    self.config = modeling.transformer_config.from_json_file(config_file)
    with tf.variable_scope("phone_to_slot"):
      # make place holder
      self.phone_tensor = tf.placeholder(
          tf.float32,
          [test_batch_size, source_max_length, self.config.source_vocab_size],
          name="phone_tensor")
      self.phone_mask = tf.sign(tf.reduce_sum(self.phone_tensor[:,:,1:], axis=-1))
      self.decoder_groundtruth = tf.placeholder(
          tf.int32,
          [test_batch_size, dest_max_length],
          name="decoder_groundtruth")
      self.decoder_input = tf.placeholder(
          tf.int32,
          [test_batch_size, dest_max_length],
          name="decoder_input")
      self.reverse_decoder_input = tf.placeholder(
          tf.int32,
          [test_batch_size, dest_max_length],
          name="reverse_decoder_input")
      self.m2l_decoder_input = tf.placeholder(
          tf.int32,
          [test_batch_size, dest_max_length],
          name="m2l_decoder_input")
      self.m2r_decoder_input = tf.placeholder(
          tf.int32,
          [test_batch_size, dest_max_length],
          name="m2r_decoder_input")
      self.valid_pos = tf.placeholder(
          tf.int32,
          [test_batch_size, source_max_length],
          name="valid_pos")
      self.sent_length = tf.reduce_sum(tf.cast(tf.sign(self.decoder_input), tf.int32), -1)
      self.m2l_length = tf.reduce_sum(tf.cast(tf.sign(self.m2l_decoder_input), tf.int32), -1)
      self.m2r_length = tf.reduce_sum(tf.cast(tf.sign(self.m2r_decoder_input), tf.int32), -1)

    # create transformer model
    self.transformer = modeling.transformer_model()

    self.transformer.init_transformer_bridge(
        config=self.config,
        source_input=self.phone_tensor,
        is_source_input_onehot=False,
        source_mask=self.phone_mask,
        dest_input=self.decoder_input,
        back_dest_input=self.reverse_decoder_input,
        m2l_dest_input=self.m2l_decoder_input,
        m2r_dest_input=self.m2r_decoder_input,
        sent_length=self.sent_length,
        m2l_length=self.m2l_length,
        m2r_length=self.m2r_length,
        is_dest_input_onehot=True,
        is_training=is_training
    )
    self.bridge_logits = self.transformer.get_bridge_output_logits()
    self.back_bridge_logits = self.transformer.get_back_bridge_output_logits()
    self.m2l_bridge_logits = self.transformer.get_m2l_bridge_logits()
    self.m2r_bridge_logits = self.transformer.get_m2r_bridge_logits()
    self.midword_logits = self.transformer.get_midwords_logits()
    
    self.attention_probs = self.transformer.get_bridge_attention_probs()
    self.get_memory_gate = self.transformer.get_memory_gate()
    self.transformer.get_bridge_attention_probs()

  def infer(self, sess, phone_input, decoder_input, reverse_decoder_input, m2l_decoder_input, m2r_decoder_input):
    feed_dict = {
        self.phone_tensor: phone_input,
        self.decoder_input: decoder_input,
        self.reverse_decoder_input: reverse_decoder_input,
        self.m2l_decoder_input: m2l_decoder_input,
        self.m2r_decoder_input: m2r_decoder_input,
    }
    logits, back_logits, m2l_logits, m2r_logits = sess.run([self.bridge_logits, self.back_bridge_logits, self.m2l_bridge_logits, self.m2r_bridge_logits], feed_dict)
    return logits, back_logits, m2l_logits, m2r_logits

  def infer_forward(self, sess, phone_input, decoder_input):
    feed_dict = {
        self.phone_tensor: phone_input,
        self.decoder_input: decoder_input
    }
    logits = sess.run(self.bridge_logits, feed_dict)
    return logits
  
  def infer_backward(self, sess, phone_input, decoder_input, reverse_decoder_input):
    feed_dict = {
        self.phone_tensor: phone_input,
        self.decoder_input: decoder_input,
        self.reverse_decoder_input: reverse_decoder_input
    }
    back_logits = sess.run(self.back_bridge_logits, feed_dict)
    return back_logits
  
  def get_midword_logits(self, sess, phone_input):
    feed_dict = {
      self.phone_tensor: phone_input,
    }
    midword_logits = sess.run([self.midword_logits], feed_dict)
    return midword_logits

  def get_att_scores(self, sess, phone_input, decoder_input):
    feed_dict = {
      self.phone_tensor: phone_input,
      self.decoder_input: decoder_input
    }
    attention_probs = sess.run([self.attention_probs], feed_dict)
    memory_gate = sess.run([self.get_memory_gate], feed_dict)
    return attention_probs, memory_gate

  def get_reverse(self, sess, decoder_input, decoder_groundtruth):
    sent_length = tf.reduce_sum(tf.cast(tf.sign(self.decoder_input), tf.int32), -1)
    batch_size = tf.shape(self.decoder_groundtruth)[0]

    reverse_decoder_groundtruth = tf.reverse_sequence(self.decoder_groundtruth, sent_length - 1, seq_dim=1,
                                                            batch_dim=0)
    reverse_decoder_input_tmp = tf.slice(self.decoder_input, [0, 1], [-1, -1])
    reverse_decoder_input_tmp = tf.reverse_sequence(reverse_decoder_input_tmp, sent_length - 1, seq_dim=1, batch_dim=0)
    reverse_decoder_input = tf.concat([tf.ones([batch_size, 1], tf.int32), reverse_decoder_input_tmp], -1)

    feed_dict = {
        self.decoder_input: decoder_input,
        self.decoder_groundtruth: decoder_groundtruth
    }

    reverse_decoder_input_np, reverse_decoder_groundtruth_np = sess.run(
              [reverse_decoder_input, reverse_decoder_groundtruth], feed_dict)
    return reverse_decoder_input_np, reverse_decoder_groundtruth_np



def is_voice_phone(phone):
  # voice index > 3
  for i in range(len(phone)):
    if phone[i] > 0.5 and i > 3:
      return True
  return False

def parse_phone_small(weights):
    return np.argmax(weights)


def data_display_small(phone_input, voc_special):
    phonelist = []
    for item in phone_input:
        phoneindex = parse_phone_small(item)
        # train_data.am_vocab
        phone = voc_special[phoneindex]
        phonelist.append(phone)
    return phonelist
def load_source_vocab(vocab_file):
    
    token2idx_={}
    idx2token_={}
    with open(vocab_file) as f:
        index=0
        for line in f:
            line=line.strip("\n")
            idx2token_[index]=line
            token2idx_[line]=index
            index+=1
    return token2idx_,idx2token_
            

def mk_y_pinyin_dict(file):
    lines = open(file, "r")
    asr20_pinyin_dict = dict()
    for i, line in enumerate(lines):
        asr20_pinyin_dict[line.strip().strip("\n")] = i
    return asr20_pinyin_dict

def cal_per_seq_score(score_list_per_seq):
  score = 1.0
  num_length = 0
  for prob in score_list_per_seq:
    score *= prob
    num_length += 1
  score = math.pow(score, 1.0 / num_length)
  return score

def test():
  '''
  testing start here
  '''
  tf.logging.info("loading vocabulary")
  token2idx, idx2token = load_vocab(FLAGS.target_vocab)
  sourcetoken2idx, idx2sourcetoken = load_source_vocab(FLAGS.source_vocab)
  
  idx2token[10000] = "early_stop"
  token2idx["early_stop"] = 10000
  tf.logging.info("building trie tree ...")
  pinyin_y_dict2id = mk_y_pinyin_dict(FLAGS.target_vocab)
  trietree = build_trie_tree_from_interspeech_file(FLAGS.place_whole_name_dict, pinyin_y_dict2id)
  reversed_trietree = build_trie_tree_reverse_from_interspeech_file(FLAGS.place_whole_name_dict, pinyin_y_dict2id)
  
  tf.logging.info("loading testing data ...")
  testing_data, oov = load_testing_data()
  tf.logging.info('%d instance loaded for testing' % len(testing_data))
  tf.logging.info('%d OOV loaded for testing' % len(oov))
  # output results
  if not os.path.exists(FLAGS.test_dir):
      os.makedirs(FLAGS.test_dir)
  # test result
  results = os.path.join(FLAGS.test_dir, "test.res")
  fout = open(results, 'w')
  # beam search result
  handler = open(os.path.join(FLAGS.test_dir,"beam_search.res"), "w")
  results_att = os.path.join(FLAGS.test_dir, "attscores.pkl")
  fout_results_att = open(results_att, "wb")


  # create model
  tf.logging.info("creating model ...")
  model = phone_to_slot(
      config_file=FLAGS.config_file,
      test_batch_size=None,
      source_max_length=FLAGS.source_max_length,
      max_masked_length=5,
      dest_max_length=None,
      is_training=False
  )
  tf.logging.info("creating model DONE!")
  # do training
  cursor = 0
  start_ticks = time.time()
  saver = tf.train.Saver(max_to_keep=5)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # loading checkpoint
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(FLAGS.ckpt)
    saver.restore(sess, ckpt)
    # run, training, run
    flag = True
    batch_index = 0
    all_correct = 0
    all_recall = 0
    forward_correct = 0
    forward_recall = 0
    backward_correct = 0
    backward_recall = 0
    middle_correct = 0
    middle_recall = 0
    middle_correct_v2 = 0
    middle_recall_v2 = 0
    oov_correct = 0
    oov_count = 0
    res_att_all = list()
    unique_song_count = dict()
    unique_song_correct = dict()
    log_output = {}
    nums = 0

    while(flag):
        phone_input = []
        decoder_input = []
        decoder_groundtruth = []
        m2l_decoder_input = []
        m2l_decoder_groundtruth = []
        m2r_decoder_input = []
        m2r_decoder_groundtruth = []
    
        batch_indeces = []
        for i in range(FLAGS.test_batch_size):
            index = cursor + i
            batch_indeces.append(index)
            phone_input.append(testing_data[index][0])
            decoder_input.append(list(int(x) for x in testing_data[index][1]))
            decoder_groundtruth.append(list(int(x) for x in testing_data[index][2]))
            m2l_decoder_input.append(list(int(x) for x in testing_data[index][4]))
            m2l_decoder_groundtruth.append(list(int(x) for x in testing_data[index][5]))
            m2r_decoder_input.append(list(int(x) for x in testing_data[index][6]))
            m2r_decoder_groundtruth.append(list(int(x) for x in testing_data[index][7]))
        
            if cursor + i >= len(testing_data)-1:
                flag = False
                break

        reverse_decoder_input, reverse_decoder_groundtruth = model.get_reverse(sess, decoder_input, decoder_groundtruth)

        cursor += FLAGS.test_batch_size
        # perform beam search

        hypotheses, cor_cnt, correctness, forward_cor_cnt, backward_cor_cnt, middle_cor_cnt, middle_cor_cnt_v2 = get_hypotheses_beam_mid(
                                                               phone_input,
                                                               decoder_input,
                                                               decoder_groundtruth,
                                                               reverse_decoder_input,
                                                               reverse_decoder_groundtruth,
                                                               m2l_decoder_input,
                                                               m2l_decoder_groundtruth,
                                                               m2r_decoder_input,
                                                               m2r_decoder_groundtruth,
                                                               sess,
                                                               model,
                                                               trietree,
                                                               reversed_trietree,
                                                               FLAGS.num_beam,
                                                               idx2token,
                                                               token2idx,
                                                               handler)
        att_scores_batch, memory_gate = model.get_att_scores(sess, phone_input=phone_input, decoder_input=decoder_input)

        fout.write("\n".join(hypotheses))
        fout.write("\n")
        all_correct += cor_cnt[0]
        all_recall += cor_cnt[-1]
        forward_correct += forward_cor_cnt[0]
        forward_recall += forward_cor_cnt[-1]
        backward_correct += backward_cor_cnt[0]
        backward_recall += backward_cor_cnt[-1]
        middle_correct += middle_cor_cnt[0]
        middle_recall += middle_cor_cnt[-1]
        middle_correct_v2 += middle_cor_cnt_v2[0]
        middle_recall_v2 += middle_cor_cnt_v2[-1]

        # unique song
        for tmp_song, tmp_cor in zip(decoder_groundtruth, correctness):
            tmp_song_str = ""
            for id in tmp_song:
                tmp_song_str += "," + str(id)
            if tmp_song_str not in unique_song_count:
                unique_song_count[tmp_song_str] = 1
                unique_song_correct[tmp_song_str] = 0
            else:
                unique_song_count[tmp_song_str] += 1
            if tmp_cor == 1:
                unique_song_correct[tmp_song_str] += 1

        # count oov
        if len(oov) != 0:
          for i in range(len(correctness)):
              if oov[batch_indeces[i]] == 1:
                  oov_count += 1
                  if correctness[i] == 1:
                      oov_correct += 1
        # show something
        duration = time.time() - start_ticks
        tf.logging.info('batch: %d, dur: %f, all: %d, cor: %d, forward_cor: %d, backward_cor: %d, middle_l2f_cor: %d, middle_r2b_cor: %d' % (
            batch_index, round(duration, 3), len(phone_input), cor_cnt[0], forward_cor_cnt[0], backward_cor_cnt[0], middle_cor_cnt[0], middle_cor_cnt_v2[0]))
        batch_index += 1
        start_ticks = time.time()
        if len(res_att_all) == 0:
          res_att_all = att_scores_batch
        else:
          res_att_all.extend(att_scores_batch)
    pickle.dump(res_att_all, fout_results_att)
    fout.close()

    avg_unique_song_corr = 0.0
    unique_song_num = 0.0
    for tmp_key in unique_song_count:
        unique_song_num += 1.0
        avg_unique_song_corr += float(unique_song_correct[tmp_key]) / float(unique_song_count[tmp_key])
    tf.logging.info("Unique Song Num: %d" % unique_song_num)
    tf.logging.info("Avg Unique Correct Rate: %f" % (avg_unique_song_corr / unique_song_num))
    tf.logging.info('All test: %d, correct: %d, %.2f%%, forward: %d, %.2f%%, backward: %d, %.2f%%, middle_l2f: %d, %.2f%%, middle_r2b: %d, %.2f%%' %
                    (len(testing_data), all_correct, all_correct/len(testing_data)*100, forward_correct, forward_correct/len(testing_data)*100, backward_correct, backward_correct/len(testing_data)*100, middle_correct, middle_correct/len(testing_data)*100, middle_correct_v2, middle_correct_v2/len(testing_data)*100))
    print("Test End")


if __name__ == "__main__":
  flags.mark_flag_as_required("testing_file")
  flags.mark_flag_as_required("config_file")
  flags.mark_flag_as_required("ckpt")
  print("Start Test")
  test()

