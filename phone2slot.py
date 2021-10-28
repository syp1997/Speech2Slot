# -*- coding: utf-8 -*-
# ASR Phone to slot model, with multi-task
# mix LM training with the task training
# written by Yohn CAO
# created on July 9, 2019
# Alibaba AI Labs

import tensorflow as tf
import six
import json
import sys
import copy
import utils.model_util as mu
import model as modeling
import numpy as np
import math
import optimization
from utils.flag_center import FLAGS


np.set_printoptions(threshold=sys.maxsize)


def num_to_str(num):
  if num is None:
    return "?"
  return str(num)


class phone_to_slot(object):
  '''
  phone to slot model
  based on transformer
  '''

  def __init__(self,
               config_file,
               is_training,
               phone_tensor,
               phone_masked_positions,
               phone_masked_groundtruth,
               decoder_input,
               decoder_groundtruth,
               m2l_decoder_input,
               m2l_decoder_groundtruth,
               m2r_decoder_input,
               m2r_decoder_groundtruth,
               mid_words,
               ):

    '''
    construct model
    '''
    # read config file
    self.config = modeling.transformer_config.from_json_file(config_file)
    with tf.variable_scope("phone_to_slot"):

      # make place holder

      self.phone_tensor = phone_tensor
      # self.phone_tensor = tf.sparse.to_dense(phone_tensor)

      # calculate mask here, assuming that the padding positions are filled with zeros
      # assuming that the masked encoder positions are filled with <PAD> so that the first element is 1.0
      # to mask <PAD> from the encoder training task, here we calculate mask bypassing the first element

      self.phone_mask = tf.sign(tf.reduce_sum(self.phone_tensor[:,:,1:], axis=-1))
      # self.phone_mask = tf.sign(self.phone_tensor)

      self.decoder_groundtruth = decoder_groundtruth
      self.decoder_input = decoder_input
      self.m2l_decoder_groundtruth = m2l_decoder_groundtruth
      self.m2l_decoder_input = m2l_decoder_input
      self.m2r_decoder_groundtruth = m2r_decoder_groundtruth
      self.m2r_decoder_input = m2r_decoder_input
      self.mid_words = mid_words
      self.sent_length = tf.reduce_sum(tf.cast(tf.sign(self.decoder_input), tf.int32), -1)
      self.m2l_length = tf.reduce_sum(tf.cast(tf.sign(self.m2l_decoder_input), tf.int32), -1)
      self.m2r_length = tf.reduce_sum(tf.cast(tf.sign(self.m2r_decoder_input), tf.int32), -1)

      batch_size = mu.get_shape_list(self.decoder_groundtruth)[0]
      self.reverse_decoder_groundtruth = tf.reverse_sequence(self.decoder_groundtruth, self.sent_length-1, seq_dim=1, batch_dim=0)
      decoder_input = tf.slice(self.decoder_input, [0, 1], [-1, -1])
      reverse_decoder_input = tf.reverse_sequence(decoder_input, self.sent_length-1, seq_dim=1, batch_dim=0)
      self.reverse_decoder_input = tf.concat([tf.ones([batch_size,1], tf.int32), reverse_decoder_input], -1)

      self.phone_masked_positions = phone_masked_positions
      self.phone_masked_groundtruth = phone_masked_groundtruth
    # create transformer model
    self.transformer = modeling.transformer_model()
    self.transformer.init_transformer_bridge(
        config=self.config,
        source_input=self.phone_tensor,
        is_source_input_onehot=False,
        # is_source_input_onehot=True,
        source_mask=self.phone_mask,
        dest_input=self.decoder_input,
        back_dest_input=self.reverse_decoder_input,
        m2l_dest_input=self.m2l_decoder_input,
        m2r_dest_input=self.m2r_decoder_input,
        is_dest_input_onehot=True,
        sent_length=self.sent_length,
        m2l_length=self.m2l_length,
        m2r_length=self.m2r_length,
        is_training=is_training
    )

    # forward loss
    decoder_groundtruth = mu.label_smoothing(tf.one_hot(
        self.decoder_groundtruth, depth=self.config.dest_vocab_size))
    decoder_groundtruth = mu.reshape_to_matrix(decoder_groundtruth)
    loss_mask = tf.cast(tf.reshape(
        tf.sign(self.decoder_groundtruth), [-1]), tf.float32)
#     loss_mask = tf.Print(loss_mask,["*"*999+"loss_mask:", tf.shape(loss_mask), loss_mask], message='debug message:', summarize=100)
    self.bridge_logits = self.transformer.get_bridge_output_logits()
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=self.bridge_logits, labels=decoder_groundtruth)
    # mask loss
    loss *= loss_mask
    self.bridge_loss = tf.reduce_sum(loss) / (0.000000001 + tf.reduce_sum(loss_mask))

    # backward loss
    reverse_decoder_groundtruth = mu.label_smoothing(tf.one_hot(
        self.reverse_decoder_groundtruth, depth=self.config.dest_vocab_size))
    reverse_decoder_groundtruth = mu.reshape_to_matrix(reverse_decoder_groundtruth)
    self.back_bridge_logits = self.transformer.get_back_bridge_output_logits()
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=self.back_bridge_logits, labels=reverse_decoder_groundtruth)
    # mask loss
    loss *= loss_mask
    self.back_bridge_loss = tf.reduce_sum(loss) / (0.000000001 + tf.reduce_sum(loss_mask))

    # m2l loss
    m2l_decoder_groundtruth = mu.label_smoothing(tf.one_hot(
        self.m2l_decoder_groundtruth, depth=self.config.dest_vocab_size))
    m2l_decoder_groundtruth = mu.reshape_to_matrix(m2l_decoder_groundtruth)
    loss_mask = tf.cast(tf.reshape(
        tf.sign(self.m2l_decoder_groundtruth), [-1]), tf.float32)
    self.m2l_bridge_logits = self.transformer.get_m2l_bridge_logits()
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=self.m2l_bridge_logits, labels=m2l_decoder_groundtruth)
    # mask loss
    loss *= loss_mask
    self.m2l_bridge_loss = tf.reduce_sum(loss) / (0.000000001 + tf.reduce_sum(loss_mask))
    
    # m2r loss
    m2r_decoder_groundtruth = mu.label_smoothing(tf.one_hot(
        self.m2r_decoder_groundtruth, depth=self.config.dest_vocab_size))
    m2r_decoder_groundtruth = mu.reshape_to_matrix(m2r_decoder_groundtruth)
    loss_mask = tf.cast(tf.reshape(
        tf.sign(self.m2r_decoder_groundtruth), [-1]), tf.float32)
    self.m2r_bridge_logits = self.transformer.get_m2r_bridge_logits()
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=self.m2r_bridge_logits, labels=m2r_decoder_groundtruth)
    # mask loss
    loss *= loss_mask
    self.m2r_bridge_loss = tf.reduce_sum(loss) / (0.000000001 + tf.reduce_sum(loss_mask))
    
    # midwords loss
#     mid_words_groundtruth = mu.label_smoothing(tf.one_hot(
#         self.mid_words, depth=self.config.dest_vocab_size))
#     mid_words_groundtruth = mu.reshape_to_matrix(mid_words_groundtruth)
#     self.midwords_logits = self.transformer.get_midwords_logits()
#     loss = tf.nn.softmax_cross_entropy_with_logits_v2(
#         logits=self.midwords_logits, labels=mid_words_groundtruth)
#     # mask loss
#     self.midwords_loss = tf.reduce_sum(loss) / batch_size

    mid_words_groundtruth = tf.cast(self.mid_words, dtype=tf.float32)
#     mid_words_groundtruth = mu.label_smoothing(mid_words_groundtruth)
#     mid_words_groundtruth = mu.reshape_to_matrix(mid_words_groundtruth)
    self.midwords_logits = self.transformer.get_midwords_logits()
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.midwords_logits, labels=mid_words_groundtruth)
    # mask loss
    self.midwords_loss = tf.reduce_sum(loss) / tf.reduce_sum(mid_words_groundtruth)

    # # memory training
    # train_batch_size = mu.get_shape_list(self.transformer.get_memory())[0]
    # # multi-task loss
    # memory_output = self.gather_indexes(
    #     self.transformer.get_memory(),
    #     train_batch_size,
    #     FLAGS.maxlen1,
    #     self.phone_masked_positions
    # )
    # # use an GELU layer
    # output_tensor = tf.layers.dense(
    #     memory_output,
    #     units=self.config.source_hidden_size,
    #     activation=mu.get_activation_func('gelu'))
    # output_tensor = mu.layer_norm(output_tensor)
    # # regression
    # logits = tf.layers.dense(
    #     output_tensor,
    #     units=self.config.source_vocab_size)
    # # loss
    # groundtruth = tf.reshape(
    #     self.phone_masked_groundtruth, [-1, self.config.source_vocab_size])
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(
    #     logits=logits, labels=groundtruth)
    # mask = tf.cast(tf.reshape(
    #     tf.sign(self.phone_masked_positions), [-1]), tf.float32)
    # loss = loss * mask
    # self.encoder_loss = tf.reduce_sum(
    #     loss) / (0.000000001 + tf.reduce_sum(mask))

    # merge loss
    # self.loss = (self.encoder_loss + self.bridge_loss + self.back_bridge_loss) / 2.0
    
    self.loss = (self.bridge_loss + self.back_bridge_loss + (self.m2l_bridge_loss+self.m2r_bridge_loss)/2.0 + self.midwords_loss) / 4.0

  def gather_indexes(self, sequence_tensor, batch_size, seq_length, positions):
    seq_shape = mu.get_shape_list(sequence_tensor)
    width = seq_shape[-1]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor
