# -*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf

import copy
from utils.flag_center import FLAGS

PHONE_TOKEN_NUM = 124

def file_based_input_fn_builder(input_file_list, mode, drop_remainder):
  name_to_features = {
    'x': tf.VarLenFeature(dtype=tf.float32),
    'x_len': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    #'x_id': tf.VarLenFeature(dtype=tf.int64),
    'decode_inputs': tf.VarLenFeature(dtype=tf.int64),
    'y': tf.VarLenFeature(dtype=tf.int64),
    'y_len': tf.FixedLenFeature(shape=[1], dtype=tf.int64)
  }
  
  def _decode_record(record, name_to_features):
    tf.logging.info('decode a record')
    example = tf.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      if name in {'x', 'decode_inputs', 'y'}:
        t = tf.sparse.to_dense(t)
      if name in {'x'}:
        t = tf.reshape(tensor=t, shape=[-1, FLAGS.vocab_size])
      example[name] = t
    return  [
      (example['x'],example['x_len']),
      (example['decode_inputs'], example['y_len'], example['y'])
    ]
  
  def input_fn(params):
    
    d = tf.data.TFRecordDataset(input_file_list)
    if mode == tf.estimator.ModeKeys.TRAIN:
      d = d.repeat()
      d = d.shuffle(5)
    batch_size_key = mode + '_batch_size'
    batch_size = params[batch_size_key]
    tf.logging.info('{} is {}'.format(batch_size_key, batch_size))
    d = d.map(lambda record: _decode_record(record, name_to_features))
    shapes = (
      ([None,124], [None]),
      ([None],[None],[None])
    )
    paddings = (
      (0.0, 0),
      (0, 0, 0)
    )
    d = d.padded_batch(batch_size=batch_size, padded_shapes=shapes, padding_values=paddings)
    
    '''
    d = d.apply(
      tf.contrib.data.map_and_batch(
        lambda record: _decode_record(record, name_to_features),
        batch_size=batch_size,
        drop_remainder=drop_remainder
      )
    )
    '''
    return d
  
  return input_fn

def get_phone_input_mask(phone_inputs, masked_position, indexes):
  masked_position_pos = tf.gather_nd(masked_position, indexes)
  paddings_pos = tf.one_hot(tf.zeros_like(masked_position_pos), PHONE_TOKEN_NUM)

  rn_t = tf.range(0, phone_inputs.shape[0])
  def index1d(t, val):
    return tf.reduce_min(tf.where(tf.equal([t], val)))
  def index1dd(t,val):
    return tf.argmax(tf.cast(tf.equal(t,val), tf.int64), axis=0)
  r = tf.map_fn(lambda x: tf.where(tf.equal(index1d(masked_position_pos, x), 0),
                                   paddings_pos[index1dd(masked_position_pos, x)] , phone_inputs[x]), rn_t, dtype=tf.float32)
  return r

def get_copy(phone_inputs):
  return phone_inputs


def file_based_input_fn_builder_transformer_bridge(input_file_list,
                                                   mode,
                                                   drop_remainder=False,
                                                   num_cpu_threads=4,
                                                   max_len1=104,
                                                   vocab_num=124):

  name_to_features = {
    'x': tf.VarLenFeature(dtype=tf.float32),
    'valid_pos': tf.VarLenFeature( dtype=tf.int64),
    'decode_inputs': tf.VarLenFeature(dtype=tf.int64),
    'y': tf.VarLenFeature(dtype=tf.int64)
  }

  def get_padding_phone(mask_size, token_num):

    padding_ = np.zeros([mask_size, token_num], dtype=np.float32)
    for i in range(len(padding_)):
      padding_[i][0] = 1.0
    padding = tf.constant(padding_)
    return padding

  def make_mask_lm_data_pre(valid_pos, phone_inputs):

    masked_position = tf.random.shuffle(valid_pos)[0:10]
    paddings = get_padding_phone(10, PHONE_TOKEN_NUM)
    phone_masked_groundtruth = tf.gather(phone_inputs, masked_position)
    phone_masked_groundtruth = tf.where(masked_position > 0, phone_masked_groundtruth, paddings)
    indexes = tf.where(masked_position > 0)

    phone_inputs_copy = tf.cond(tf.reduce_sum(masked_position) > 0,
                                lambda: get_phone_input_mask(phone_inputs, masked_position, indexes),
                                lambda: get_copy(phone_inputs))

    # return phone_masked_positions, phone_masked_groundtruth
    return masked_position, phone_masked_groundtruth, phone_inputs_copy

  def get_padding_matrix(max_x, phone_size):
    pad = list(np.zeros(phone_size))
    pad[0] = 1
    padding = np.tile(np.expand_dims(pad, 0), (max_x, 1))
    # print(padding)
    return padding

  def get_padded_matrix(phone_input, padded_input, phone_size):
    mask_ = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(phone_input), -1)), axis=1)
    mask = tf.tile(mask_, [1, phone_size])
    mask_op = tf.ones_like(mask) - mask
    masked_phone_input = phone_input * mask
    masked_padding = padded_input * mask_op
    padded_output = masked_phone_input + masked_padding
    return padded_output

  def process_for_onehot(valid_pos_10):
    valid_pos_10_zeros = tf.zeros_like(valid_pos_10)
    valid_pos_10_ones = tf.ones_like(valid_pos_10)
    valid_pos_10_sign = tf.sign(valid_pos_10)
    valid_pos_10_unsign = valid_pos_10_ones - valid_pos_10_sign
    valid_pos_10_zero_matrix = valid_pos_10_zeros - valid_pos_10_unsign
    valid_pos_final = valid_pos_10_zero_matrix + valid_pos_10
    return valid_pos_final

  def make_mask_lm_data(valid_pos, phone_inputs):
    # valid 104
    # phone_input: 104 * 124
    # mask_len : 10

    INPUT_MAX_X = FLAGS.maxlen1
    PHONE_SIZE = FLAGS.vocab_size
    MASK_LEN = FLAGS.encoder_masked_size
    index = np.asarray(range(INPUT_MAX_X))
    index_copy = copy.deepcopy(index)
    np.random.shuffle(index_copy)
    index_copy_10 = index_copy[0:MASK_LEN]

    get_padding_matrix_10 = get_padding_matrix(MASK_LEN, PHONE_SIZE)

    valid_pos_10 = tf.gather(valid_pos, index_copy_10)
    valid_pos_final = process_for_onehot(valid_pos_10)
    valid_pos_final_matrix = tf.one_hot(valid_pos_final, INPUT_MAX_X, dtype=tf.float32)
    phone_masked_groundtruth = tf.matmul(valid_pos_final_matrix, phone_inputs)
    masked_position = valid_pos_10

    valid_pos_final_matrix_broad_cast = tf.cast(tf.reduce_sum(valid_pos_final_matrix, 0), tf.float32)
    valid_pos_final_matrix_max = 1.0 - valid_pos_final_matrix_broad_cast
    phone_inputs_masked = phone_inputs * tf.expand_dims(valid_pos_final_matrix_max, -1)
    padding = get_padding_phone(INPUT_MAX_X, PHONE_SIZE)
    phone_inputs_mask = padding * tf.expand_dims(valid_pos_final_matrix_broad_cast, -1)
    phone_masked_groundtruth = get_padded_matrix(phone_masked_groundtruth, get_padding_matrix_10, PHONE_SIZE)
    phone_inputs_copy = phone_inputs_masked + phone_inputs_mask

    return masked_position, phone_masked_groundtruth, phone_inputs_copy

  def _decode_record(record, name_to_features):
    tf.logging.info('decode a record')
    example = tf.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      if name in {'x', 'decode_inputs', 'valid_pos', 'y'}:
        t = tf.sparse.to_dense(t)
      if name in {'x'}:
        t = tf.reshape(tensor=t, shape=[FLAGS.maxlen1, FLAGS.vocab_size])
      example[name] = t

    valid_pos = example['valid_pos']
    phone_inputs = example['x']
    phone_masked_positions, phone_masked_groundtruth, phone_inputs_copy = \
      make_mask_lm_data(valid_pos=valid_pos, phone_inputs=phone_inputs)

    # if masked_phone == True:
    return [
      (phone_inputs, phone_inputs_copy, phone_masked_groundtruth, phone_masked_positions, valid_pos),
      (example['decode_inputs'], example['y'] , example['y'])
    ]

  def input_fn(params):

    batch_size_key = mode + '_batch_size'
    batch_size = params[batch_size_key]

    if mode == tf.estimator.ModeKeys.TRAIN:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_file_list))


      # `cycle_length` is the number of parallel files that get read.
      cycle_length = num_cpu_threads

      tf.logging.info("\n\nNum of threads is %d\n\n".format(num_cpu_threads))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_file_list))
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=True,
              cycle_length=cycle_length))

      d = d.shuffle(buffer_size=batch_size * 1000)

    else:
      d = tf.data.TFRecordDataset(input_file_list)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_calls=num_cpu_threads,#！！！！
            drop_remainder=drop_remainder))
    '''
    d = d.map(lambda record: _decode_record(record, name_to_features), num_parallel_calls=num_cpu_threads)
    d = d.batch(batch_size=batch_size)
    '''
    d = d.prefetch(buffer_size=batch_size)
    return d

  return input_fn

def file_based_input_fn_builder_transformer_bridge_mid_decode(input_file_list,
                                                   mode,
                                                   drop_remainder=False,
                                                   num_cpu_threads=4,
                                                   max_len1=104,
                                                   vocab_num=124):

  name_to_features = {
    'x': tf.VarLenFeature(dtype=tf.float32),
    'valid_pos': tf.VarLenFeature( dtype=tf.int64),
    'decode_inputs': tf.VarLenFeature(dtype=tf.int64),
    'y': tf.VarLenFeature(dtype=tf.int64),
    'm2l_decode_inputs': tf.VarLenFeature(dtype=tf.int64),
    'm2l_y': tf.VarLenFeature(dtype=tf.int64),
    'm2r_decode_inputs': tf.VarLenFeature(dtype=tf.int64),
    'm2r_y': tf.VarLenFeature(dtype=tf.int64),
    'mid_words': tf.VarLenFeature(dtype=tf.int64),
  }

  def get_padding_phone(mask_size, token_num):

    padding_ = np.zeros([mask_size, token_num], dtype=np.float32)
    for i in range(len(padding_)):
      padding_[i][0] = 1.0
    padding = tf.constant(padding_)
    return padding

  def make_mask_lm_data_pre(valid_pos, phone_inputs):

    masked_position = tf.random.shuffle(valid_pos)[0:10]
    paddings = get_padding_phone(10, PHONE_TOKEN_NUM)
    phone_masked_groundtruth = tf.gather(phone_inputs, masked_position)
    phone_masked_groundtruth = tf.where(masked_position > 0, phone_masked_groundtruth, paddings)
    indexes = tf.where(masked_position > 0)

    phone_inputs_copy = tf.cond(tf.reduce_sum(masked_position) > 0,
                                lambda: get_phone_input_mask(phone_inputs, masked_position, indexes),
                                lambda: get_copy(phone_inputs))

    # return phone_masked_positions, phone_masked_groundtruth
    return masked_position, phone_masked_groundtruth, phone_inputs_copy

  def get_padding_matrix(max_x, phone_size):
    pad = list(np.zeros(phone_size))
    pad[0] = 1
    padding = np.tile(np.expand_dims(pad, 0), (max_x, 1))
    # print(padding)
    return padding

  def get_padded_matrix(phone_input, padded_input, phone_size):
    mask_ = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(phone_input), -1)), axis=1)
    mask = tf.tile(mask_, [1, phone_size])
    mask_op = tf.ones_like(mask) - mask
    masked_phone_input = phone_input * mask
    masked_padding = padded_input * mask_op
    padded_output = masked_phone_input + masked_padding
    return padded_output

  def process_for_onehot(valid_pos_10):
    valid_pos_10_zeros = tf.zeros_like(valid_pos_10)
    valid_pos_10_ones = tf.ones_like(valid_pos_10)
    valid_pos_10_sign = tf.sign(valid_pos_10)
    valid_pos_10_unsign = valid_pos_10_ones - valid_pos_10_sign
    valid_pos_10_zero_matrix = valid_pos_10_zeros - valid_pos_10_unsign
    valid_pos_final = valid_pos_10_zero_matrix + valid_pos_10
    return valid_pos_final

  def make_mask_lm_data(valid_pos, phone_inputs):
    # valid 104
    # phone_input: 104 * 124
    # mask_len : 10

    INPUT_MAX_X = FLAGS.maxlen1
    PHONE_SIZE = FLAGS.vocab_size
    MASK_LEN = FLAGS.encoder_masked_size
    index = np.asarray(range(INPUT_MAX_X))
    index_copy = copy.deepcopy(index)
    np.random.shuffle(index_copy)
    index_copy_10 = index_copy[0:MASK_LEN]

    get_padding_matrix_10 = get_padding_matrix(MASK_LEN, PHONE_SIZE)

    valid_pos_10 = tf.gather(valid_pos, index_copy_10)
    valid_pos_final = process_for_onehot(valid_pos_10)
    valid_pos_final_matrix = tf.one_hot(valid_pos_final, INPUT_MAX_X, dtype=tf.float32)
    phone_masked_groundtruth = tf.matmul(valid_pos_final_matrix, phone_inputs)
    masked_position = valid_pos_10

    valid_pos_final_matrix_broad_cast = tf.cast(tf.reduce_sum(valid_pos_final_matrix, 0), tf.float32)
    valid_pos_final_matrix_max = 1.0 - valid_pos_final_matrix_broad_cast
    phone_inputs_masked = phone_inputs * tf.expand_dims(valid_pos_final_matrix_max, -1)
    padding = get_padding_phone(INPUT_MAX_X, PHONE_SIZE)
    phone_inputs_mask = padding * tf.expand_dims(valid_pos_final_matrix_broad_cast, -1)
    phone_masked_groundtruth = get_padded_matrix(phone_masked_groundtruth, get_padding_matrix_10, PHONE_SIZE)
    phone_inputs_copy = phone_inputs_masked + phone_inputs_mask

    return masked_position, phone_masked_groundtruth, phone_inputs_copy

  def _decode_record(record, name_to_features):
    tf.logging.info('decode a record')
    example = tf.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      if name in {'x', 'valid_pos', 'decode_inputs', 'y', 'm2l_decode_inputs', 'm2l_y', 'm2r_decode_inputs','m2r_y', 'mid_words'}:
        t = tf.sparse.to_dense(t)
      if name in {'x'}:
        t = tf.reshape(tensor=t, shape=[FLAGS.maxlen1, FLAGS.vocab_size])
      example[name] = t

    valid_pos = example['valid_pos']
    phone_inputs = example['x']
    phone_masked_positions, phone_masked_groundtruth, phone_inputs_copy = \
      make_mask_lm_data(valid_pos=valid_pos, phone_inputs=phone_inputs)

    # if masked_phone == True:
#     print("*"*999)
#     print(example['decode_inputs'])
    return [
      (phone_inputs, phone_inputs_copy, phone_masked_groundtruth, phone_masked_positions, valid_pos),
      (example['decode_inputs'], example['y'], example['m2l_decode_inputs'], example['m2l_y'], example['m2r_decode_inputs'], example['m2r_y'], example['mid_words'])
    ]

  def input_fn(params):

    batch_size_key = mode + '_batch_size'
    batch_size = params[batch_size_key]

    if mode == tf.estimator.ModeKeys.TRAIN:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_file_list))


      # `cycle_length` is the number of parallel files that get read.
      cycle_length = num_cpu_threads

      tf.logging.info("\n\nNum of threads is %d\n\n".format(num_cpu_threads))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_file_list))
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=True,
              cycle_length=cycle_length))

      d = d.shuffle(buffer_size=batch_size * 1000)

    else:
      d = tf.data.TFRecordDataset(input_file_list)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_calls=num_cpu_threads,#！！！！
            drop_remainder=drop_remainder))
    '''
    d = d.map(lambda record: _decode_record(record, name_to_features), num_parallel_calls=num_cpu_threads)
    d = d.batch(batch_size=batch_size)
    '''
    d = d.prefetch(buffer_size=batch_size)
    return d

  return input_fn
