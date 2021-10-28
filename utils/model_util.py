# -*- coding: utf-8 -*-
# Transformer utility funcitons (independent of the model)
#
# !!! IMPORTANT !!! Do NOT put any model related code here since the module is shared across
# different model implementation. One change here will affect all related models
#
# written by Yohn CAO
# created on July 8, 2019
# Alibaba AI Labs

import os
import sys
ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(ROOT_PATH)

import tensorflow as tf
import six
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import ops



def dropout(input_tensor, dropout_probability):
  '''
  verify dropout ratio and apply dropout
  '''
  if dropout_probability is None or dropout_probability == 0.0:
    return input_tensor
  # apply dropout
  return tf.nn.dropout(
      x=input_tensor,
      keep_prob=1.0 - dropout_probability)

def dropout_layer(input_tensor, dropout_probability, is_training):
  '''
  verify dropout ratio and apply dropout
  '''
  if dropout_probability is None or dropout_probability == 0.0:
    return input_tensor
  # apply dropout
  return tf.layers.dropout(
      inputs=input_tensor,
      rate=dropout_probability,
      training=is_training)

def layer_norm(input_tensor, name=None):
  '''
  layer normalization
  https://arxiv.org/abs/1607.06450
  '''
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor,
      begin_norm_axis=-1,
      begin_params_axis=-1,
      scope=name)

# this is a simpler version of Tensorflow's 'official' version. See:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 1e-10)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 1e-10)

def layer_norm_and_dropout(input_tensor, dropout_probability, name=None):
  '''
  dropout and then layer normalization
  '''
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_probability)
  return output_tensor


def gelu(input_tensor):
  '''
  Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    input_tensor: float Tensor to perform activation.

  Returns:
    `input_tensor` with the GELU activation applied.
  '''
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf


def get_activation_func(func_name):
  '''
  get activation funciton by name
  '''
  # verify input
  if not func_name:
    return None

  act = func_name.lower()
  # check activation function name
  if act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def get_shape_list(tensor):
  '''
  Returns a list of the shape of tensor, preferring static dimensions.
  -- All static dimensions will be returned as python integers, and dynamic dimensions will be returned
  as tf.Tensor scalars.
  '''
  shape = tensor.shape.as_list()
  # check non static index (tensor)
  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)
  # anything bad happen?
  if not non_static_indexes:
    return shape
  # tensor shape
  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    # replace dynamic ones with tensors
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  '''
  Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)
  '''
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  return tf.reshape(input_tensor, [-1, width])


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    vocab = get_shape_list(inputs)[-1]
#     print(inputs)
    return ((1 - epsilon) * inputs) + (epsilon / vocab)


def get_shuffle_matrix(decode_loss_mask):
  '''
  shuffle matrix generation for row exchange
  :param decode_loss_mask:  shape: bs * decoder_length
  :return: shuffle matrix: bs * bs
  '''
  bs = tf.shape(decode_loss_mask)[0]
  # [0, 1, 0, 1, 0]
  mask = tf.cast(tf.sign(tf.reduce_sum(decode_loss_mask, axis=-1)), tf.int32)
  # [0, 1, 2, 3, 4]
  raw_index = tf.range(bs)
  # [0, 1, 0, 3, 0]
  index_keep = raw_index * mask
  # [0, 2, 4]
  index_need_shuffle = tf.boolean_mask(raw_index, tf.cast(1 - mask, tf.bool))
  # shuffle: [4,0,2]
  index_shuffled = tf.random_shuffle(index_need_shuffle)
  # [4, 0, 0, 0, 2]
  index_shuffle = tf.scatter_nd(tf.expand_dims(index_need_shuffle, -1), index_shuffled, tf.expand_dims(bs, -1) )

  index_final = index_keep + index_shuffle

  shuffle_matrix = tf.one_hot(
      index_final, depth=bs, dtype=tf.int32)

  return shuffle_matrix

def get_max_pooling_with_mask(rep_2d, rep_cross_att, batch_size, seq_length, mask):
  # batch_size * length * (2 * hidden_size)
  rep_concat_3d = tf.reshape(tf.concat([rep_2d, rep_cross_att], axis=0), [batch_size, seq_length, -1])
  # masked representation
  rep_concat_mask_3d = tf.multiply(rep_concat_3d ,tf.expand_dims(mask, -1)) + tf.multiply((rep_concat_3d + 1), (1.0 - tf.expand_dims(mask, -1) ) * -100000.0 )
  # batch_size * length * (2 * hidden_size) * 1
  rep_concat_mask_4d = tf.expand_dims(rep_concat_mask_3d, axis=-1)
  # batch_size * 1 * (2 * hidden_size) * 1
  representation_pool = tf.nn.max_pool(rep_concat_mask_4d, [1, seq_length, 1, 1],
                                              strides=[1, 1, 1, 1], padding="VALID")
  # batch_size * (2 * hidden_size)
  representation_pool = tf.squeeze(representation_pool)

  return representation_pool

def get_max_pooling_with_mask_2d(rep_2d, batch_size, seq_length, mask, hidden_size):
  # batch_size * length * (2 * hidden_size)
  rep_concat_3d = tf.reshape(rep_2d, [-1, seq_length, hidden_size ])
  # masking
  rep_concat_3d = tf.multiply(rep_concat_3d ,tf.expand_dims(mask, -1))
  # masked representation
  rep_concat_mask_3d = rep_concat_3d +  (1.0 - tf.expand_dims(mask, -1) ) * -100000.0
  # batch_size * length * (hidden_size) * 1
  rep_concat_mask_4d = tf.expand_dims(rep_concat_mask_3d, axis=-1)
  # batch_size * 1 * (hidden_size) * 1
  representation_pool = tf.nn.max_pool(rep_concat_mask_4d, [1, seq_length, 1, 1],
                                              strides=[1, 1, 1, 1], padding="VALID")
  # batch_size * (hidden_size)
  representation_pool = tf.squeeze(representation_pool, axis=[1,3])

  return representation_pool

def dropout_attention(x, keep_prob, noise_shape=None, seed=None, name=None):  # pylint: disable=invalid-name
  """Computes dropout.

  With probability `keep_prob`, outputs the input element scaled up by
  `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
  sum is unchanged.

  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.

  Args:
    x: A floating point tensor.
    keep_prob: A scalar `Tensor` with the same type as x. The probability
      that each element is kept.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.set_random_seed`
      for behavior.
    name: A name for this operation (optional).

  Returns:
  Returns:
  Returns:
    A Tensor of the same shape of `x`.

  Raises:
    ValueError: If `keep_prob` is not in `(0, 1]` or if `x` is not a floating
      point tensor.
  """
  with ops.name_scope(name, "dropout", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if not x.dtype.is_floating:
      raise ValueError("x has to be a floating point tensor since it's going to"
                       " be scaled. Got a %s tensor instead." % x.dtype)
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)

    # Early return if nothing needs to be dropped.
    if isinstance(keep_prob, float) and keep_prob == 1:
      return x
    if context.executing_eagerly():
      if isinstance(keep_prob, ops.EagerTensor):
        if keep_prob.numpy() == 1:
          return x
    else:
      keep_prob = ops.convert_to_tensor(
          keep_prob, dtype=x.dtype, name="keep_prob")
      keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

      # Do nothing if we know keep_prob == 1
      if tensor_util.constant_value(keep_prob) == 1:
        return x

    noise_shape = _get_noise_shape(x, noise_shape)

    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob
    random_tensor += random_ops.random_uniform(
        noise_shape, seed=seed, dtype=x.dtype)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = math_ops.floor(random_tensor)
    ret = math_ops.div(x, keep_prob) * binary_tensor
    if not context.executing_eagerly():
      ret.set_shape(x.get_shape())
    return ret
