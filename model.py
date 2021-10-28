# -*- coding: utf-8 -*-
import tensorflow as tf
import six
import json
import copy
import utils.model_util as mu
import numpy as np
import math


class transformer_config(object):
  '''
  all hyper parameters required in Transformer
  '''

  def __init__(self,
               source_vocab_size,
               dest_vocab_size,
               source_hidden_size=512,
               source_hidden_layers=8,
               source_attention_heads=8,
               source_intermediate_size=2048,
               source_intermediate_dropout=0.1,
               source_attention_dropout=0.1,
               dest_hidden_size=512,
               dest_hidden_layers=8,
               dest_attention_heads=8,
               dest_intermediate_size=2048,
               dest_intermediate_dropout=0.0,
               dest_attention_dropout=0.0,
               bridge_hidden_layers=2,
               hidden_act="gelu",
               initializer_range=0.02,
               max_position_embeddings=512,
               is_position_embedding_trainable=0):
    self.source_vocab_size = source_vocab_size
    self.dest_vocab_size = dest_vocab_size
    self.source_hidden_size = source_hidden_size
    self.source_hidden_layers = source_hidden_layers
    self.source_attention_heads = source_attention_heads
    self.source_intermediate_size = source_intermediate_size
    self.source_intermediate_dropout = source_intermediate_dropout
    self.source_attention_dropout = source_attention_dropout
    self.dest_hidden_size = dest_hidden_size
    self.dest_hidden_layers = dest_hidden_layers
    self.dest_attention_heads = dest_attention_heads
    self.dest_intermediate_size = dest_intermediate_size
    self.dest_intermediate_dropout = dest_intermediate_dropout
    self.dest_attention_dropout = dest_attention_dropout
    self.hidden_act = hidden_act
    self.max_position_embeddings = max_position_embeddings
    self.is_position_embedding_trainable = is_position_embedding_trainable
    self.initializer_range = initializer_range
    self.bridge_hidden_layers = bridge_hidden_layers

  @classmethod
  def from_dict(cls, json_object):
    '''
    construct from a Python dictionary of parameters
    '''
    config = transformer_config(source_vocab_size=None, dest_vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    '''
    construct from a JSON file
    '''
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))


class transformer_model(object):
  '''
  transformer model
  https://arxiv.org/abs/1706.03762
  '''

  def __init__(self):
    '''
    do nothing here
    '''
    return

  def init_transformer_bridge(self,
                              config,
                              source_input,
                              is_source_input_onehot,
                              source_mask,
                              dest_input,
                              back_dest_input,
                              m2l_dest_input,
                              m2r_dest_input,
                              sent_length,
                              m2l_length,
                              m2r_length,
                              is_dest_input_onehot,
                              is_training):
    '''
    construct transformer model
    '''
    # make a deep copy so that we can change the values freely
    config = copy.deepcopy(config)
    # stop drop-out during prediction
    if not is_training:
      config.source_intermediate_dropout = 0.0
      config.source_attention_dropout = 0.0
      config.dest_intermediate_dropout = 0.0
      config.dest_attention_dropout = 0.0

    with tf.variable_scope("transformer"):
      # the result memory shape is [batch_size, seq_length, hidden_size]
      #gate [B*L,1]
      self.memory, self.memory_gate, self.midwords_logits = self.build_encoder(
          config=config,
          source_input=source_input,
          is_source_input_onehot=is_source_input_onehot,
          source_mask=source_mask
      )

      if not is_training:
#         # acceleration for beam search
#         # during beam search, decoder input batch size is N times of the encoder batch size, where N is the beam size
#         # to save time, the encoder memory are calculated once and tiled for the decoder
#         source_input_shape = mu.get_shape_list(source_input)
#         dest_input_shape = mu.get_shape_list(dest_input)
#         # assume that dest_batch_size can be divided by source_batch_size, or tensor error will be thrown
#         source_batch_size = source_input_shape[0]
#         source_max_length = source_input_shape[1]
#         dest_batch_size = dest_input_shape[0]
#         beam_size = tf.cast(dest_batch_size / source_batch_size, dtype=tf.int32)
#         # tile memory (will throw error when input/output cannot match)
#         self.memory_expend = tf.reshape(
#             tf.tile(tf.expand_dims(self.memory, axis=1), [1, beam_size, 1, 1]),
#             [dest_batch_size, source_max_length, config.source_hidden_size])
#         # tile attention mask
#         source_mask = tf.reshape(
#             tf.tile(tf.expand_dims(source_mask, axis=1), [1, beam_size, 1]),
#             [dest_batch_size, source_max_length])
        self.memory_expend = self.memory
      else:
        # make sure self.memory_expend is assigned during training stage
        self.memory_expend = self.memory

      # go on with decoder construction
      self.decoder_output = self.build_decoder(
            config=config,
            dest_input=dest_input,
            is_dest_input_onehot=is_dest_input_onehot
      )
      # make a bridge
      self.bridge_output, self.att_scores = self.build_bridge(
            config=config,
            memory=self.memory_expend,
            memory_gate=self.memory_gate,
            decoder_output=self.decoder_output,
            source_mask=source_mask
      )



      with tf.variable_scope("bridge_classification"):
        # decoder logit calculation
        if config.bridge_hidden_layers > 0:
          output_tensor = mu.reshape_to_matrix(self.bridge_output)
          # linear projection
          self.bridge_logits = tf.layers.dense(
                output_tensor,
                config.dest_vocab_size,
                use_bias=False)
        else:
          # decoder logit calculation
          output_tensor = mu.reshape_to_matrix(self.bridge_output)
          output_hidden_size = mu.get_shape_list(output_tensor)[1]
          output_tensor = tf.layers.dense(
                    output_tensor,
                    units=output_hidden_size,
                    activation=mu.get_activation_func('gelu'))
          output_tensor = mu.layer_norm(output_tensor)
          # linear projection
          self.bridge_logits = tf.layers.dense(
                    output_tensor,
                    config.dest_vocab_size,
                    use_bias=False)

      with tf.variable_scope("backward"):
        self.back_decoder_output = self.build_decoder(
            config=config,
            dest_input=back_dest_input,
            is_dest_input_onehot=is_dest_input_onehot
        )
        # make a bridge
        self.back_bridge_output, self.back_att_scores = self.build_bridge(
            config=config,
            memory=self.memory_expend,
            memory_gate=self.memory_gate,
            decoder_output=self.back_decoder_output,
            source_mask=source_mask
        )

        with tf.variable_scope("bridge_classification"):
            # decoder logit calculation
            if config.bridge_hidden_layers > 0:
                output_tensor = mu.reshape_to_matrix(self.back_bridge_output)
                # linear projection
                self.back_bridge_logits = tf.layers.dense(
                    output_tensor,
                    config.dest_vocab_size,
                    use_bias=False)
            else:
                # decoder logit calculation
                output_tensor = mu.reshape_to_matrix(self.back_bridge_output)
                output_hidden_size = mu.get_shape_list(output_tensor)[1]
                output_tensor = tf.layers.dense(
                    output_tensor,
                    units=output_hidden_size,
                    activation=mu.get_activation_func('gelu'))
                output_tensor = mu.layer_norm(output_tensor)
                # linear projection
                self.back_bridge_logits = tf.layers.dense(
                    output_tensor,
                    config.dest_vocab_size,
                    use_bias=False)
                
      with tf.variable_scope("middle2left"):
        self.m2l_decoder_output = self.build_decoder(
            config=config,
            dest_input=m2l_dest_input,
            is_dest_input_onehot=is_dest_input_onehot,
        )
        # make a bridge
        self.m2l_bridge_output, self.m2l_att_scores = self.build_bridge(
            config=config,
            memory=self.memory_expend,
            memory_gate=self.memory_gate,
            decoder_output=self.m2l_decoder_output,
            source_mask=source_mask
        )

        with tf.variable_scope("bridge_classification"):
          # decoder logit calculation
          if config.bridge_hidden_layers > 0:
            output_tensor = mu.reshape_to_matrix(self.m2l_bridge_output)
            # linear projection
            self.m2l_bridge_logits = tf.layers.dense(
                output_tensor,
                config.dest_vocab_size,
                use_bias=False)
          else:
            # decoder logit calculation
            output_tensor = mu.reshape_to_matrix(self.m2l_bridge_output)
            output_hidden_size = mu.get_shape_list(output_tensor)[1]
            output_tensor = tf.layers.dense(
                output_tensor,
                units=output_hidden_size,
                activation=mu.get_activation_func('gelu'))
            output_tensor = mu.layer_norm(output_tensor)
            # linear projection
            self.m2l_bridge_logits = tf.layers.dense(
                output_tensor,
                config.dest_vocab_size,
                use_bias=False)
            
      with tf.variable_scope("middle2right"):
        self.m2r_decoder_output = self.build_decoder(
            config=config,
            dest_input=m2r_dest_input,
            is_dest_input_onehot=is_dest_input_onehot,
        )
        # make a bridge
        self.m2r_bridge_output, self.m2r_att_scores = self.build_bridge(
            config=config,
            memory=self.memory_expend,
            memory_gate=self.memory_gate,
            decoder_output=self.m2r_decoder_output,
            source_mask=source_mask
        )

        with tf.variable_scope("bridge_classification"):
          # decoder logit calculation
          if config.bridge_hidden_layers > 0:
            output_tensor = mu.reshape_to_matrix(self.m2r_bridge_output)
            # linear projection
            self.m2r_bridge_logits = tf.layers.dense(
                output_tensor,
                config.dest_vocab_size,
                use_bias=False)
          else:
            # decoder logit calculation
            output_tensor = mu.reshape_to_matrix(self.m2r_bridge_output)
            output_hidden_size = mu.get_shape_list(output_tensor)[1]
            output_tensor = tf.layers.dense(
                output_tensor,
                units=output_hidden_size,
                activation=mu.get_activation_func('gelu'))
            output_tensor = mu.layer_norm(output_tensor)
            # linear projection
            self.m2r_bridge_logits = tf.layers.dense(
                output_tensor,
                config.dest_vocab_size,
                use_bias=False)

  def get_bridge_output_logits(self):
    return self.bridge_logits
  
  def get_back_bridge_output_logits(self):
    return self.back_bridge_logits
  
  def get_m2l_bridge_logits(self):
    return self.m2l_bridge_logits
  
  def get_m2r_bridge_logits(self):
    return self.m2r_bridge_logits

  def get_memory_gate(self):
    return self.memory_gate

  def get_bridge_attention_probs(self):
    return self.att_scores

  def get_memory(self):
    return self.memory_expend
  
  def get_midwords_logits(self):
    return self.midwords_logits

  def get_single_memory(self):
    return self.memory

  def is_embedding_reusable(self, config):
    # does source and dest share the same vocabulary?
    return config.dest_hidden_size == config.source_hidden_size and config.dest_vocab_size == config.source_vocab_size

  def build_bridge(self,
                   config,
                   memory,
                   memory_gate,
                   decoder_output,
                   source_mask
                   ):
    '''
    bridge network is a special network that connect the decoder and the encoder
    the input is the memory that attended to the decoder LM
    decoder LM is isolated in this way
    '''
    with tf.variable_scope("bridge"):
      # make embedding mask
      source_attention_mask = None
      if source_mask is not None:
        source_attention_mask = self.create_attention_mask_from_input_mask(
            decoder_output, source_mask)
      dest_attention_mask = self.get_language_model_mask(decoder_output)
      # keep shape
      output_shape = mu.get_shape_list(decoder_output)
      batch_size = output_shape[0]
      decoder_seq_length = output_shape[1]
      encoder_seq_length = mu.get_shape_list(memory)[1]
      # attention on decoder
      #[B*L,H]
      memory_2d = mu.reshape_to_matrix(memory)
      decoder_output_2d = mu.reshape_to_matrix(decoder_output)
      with tf.variable_scope("initial_memory_attention"):
        bridge_input, att_scores = self.attention_pooling_gate(
            from_tensor=decoder_output_2d,
            to_tensor=memory_2d,
            gate_tensor=memory_gate,
            batch_size=batch_size,
            from_seq_length=decoder_seq_length,
            to_seq_length=encoder_seq_length,
            from_hidden_size=config.dest_hidden_size,
            to_hidden_size=config.source_hidden_size,
            attention_mask=source_attention_mask,
            attention_probs_dropout_prob=config.dest_attention_dropout
        )

      # apply positional embedding (optional: does NOT show much help)
      # bridge_input = tf.reshape(bridge_input, output_shape)
      # bridge_input = self.apply_absolute_positional_embedding(
      #   bridge_input,
      #   config.is_position_embedding_trainable,
      #   config.max_position_embeddings
      # )
      # bridge_input = mu.reshape_to_matrix(bridge_input)

      # now we need additional transformer network to conbine the memories
      prev_output = bridge_input
      # set config.bridge_hidden_layers to ZERO to remove bridge layer
      if config.bridge_hidden_layers > 0:
        # create relative position encoding
        relative_position_embedding_all, rel_index = self.create_relative_positional_encoding(
          layer_num=config.bridge_hidden_layers,
          max_position_embeddings=config.max_position_embeddings,
          num_attention_heads=config.dest_attention_heads,
          seq_length=decoder_seq_length,
          width=config.dest_hidden_size
        )
        # TODO: check if we need to do layer_norm here
        for layer_index in range(config.bridge_hidden_layers):
          with tf.variable_scope("layer_%d" % layer_index):
            with tf.variable_scope("multihead_attention_self"):
              # fetch relative position embedding
              relative_position_embedding = tf.squeeze(
                relative_position_embedding_all[layer_index:(layer_index+1),:,:,:], 
                axis=0)
              # multi-head attention on self
              attention_output = self.multihead_attention_resnet_with_relative_positional_encoding(
                  prev_output=prev_output,
                  batch_size=batch_size,
                  from_seq_length=decoder_seq_length,
                  to_seq_length=decoder_seq_length,
                  from_hidden_size=config.dest_hidden_size,
                  num_attention_heads=config.dest_attention_heads,
                  attention_mask=dest_attention_mask,
                  hidden_dropout_prob=config.dest_intermediate_dropout,
                  attention_probs_dropout_prob=config.dest_attention_dropout,
                  relative_position_embedding=relative_position_embedding,
                  rel_index=rel_index
              )

            with tf.variable_scope("fullconnected_resnet"):
              layer_output = self.fullconnected_resnet(
                  attention_output=attention_output,
                  hidden_size=config.dest_hidden_size,
                  intermediate_size=config.dest_intermediate_size,
                  hidden_act=config.hidden_act,
                  hidden_dropout_prob=config.dest_intermediate_dropout
              )
              # set to prev
              prev_output = layer_output
      # reshape to the original shape
      return tf.reshape(prev_output, output_shape), att_scores

  def build_decoder(self,
                    config,
                    dest_input,
                    is_dest_input_onehot):
    '''
    build Decoder network
    '''
    with tf.variable_scope("decoder"):
      # embedding lookup
      embedding_output, self.decoder_embedding_table = self.embedding_lookup(
          vocab_size = config.dest_vocab_size,
          embedding_size = config.dest_hidden_size,
          input_tensor = dest_input,
          is_input_onehot = is_dest_input_onehot,
          embedding_table = None,
          scale= False
      )

      embedding_output = mu.dropout(
          embedding_output,
          config.dest_attention_dropout)

      dest_attention_mask = self.get_language_model_mask(embedding_output)
      output_shape = mu.get_shape_list(embedding_output)
      batch_size = output_shape[0]
      decoder_seq_length = output_shape[1]

      # batch_size * decoder_seq_length * config.dest_hidden_size
      output_shape = mu.get_shape_list(embedding_output)
      batch_size = output_shape[0]
      decoder_seq_length = output_shape[1]

      # create relative position encoding
      relative_position_embedding_all, rel_index = self.create_relative_positional_encoding(
        layer_num=config.dest_hidden_layers,
        max_position_embeddings=config.max_position_embeddings,
        num_attention_heads=config.dest_attention_heads,
        seq_length=decoder_seq_length,
        width=config.dest_hidden_size
      )

      prev_output = mu.reshape_to_matrix(embedding_output)
      for layer_index in range(config.dest_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_index):
          with tf.variable_scope("multihead_attention_self"):
            # normalize Q & K at the first layer
            # the embeddings are scaled so Q & K are not aligned with the relative position encoding
            # multi-head attention on self
            # fetch relative position embedding
            relative_position_embedding = tf.squeeze(
              relative_position_embedding_all[layer_index:(layer_index+1),:,:,:],
              axis=0)
            attention_output = self.multihead_attention_resnet_with_relative_positional_encoding(
                prev_output=prev_output,
                batch_size=batch_size,
                from_seq_length=decoder_seq_length,
                to_seq_length=decoder_seq_length,
                from_hidden_size=config.dest_hidden_size,
                num_attention_heads=config.dest_attention_heads,
                attention_mask=dest_attention_mask,
                hidden_dropout_prob=config.dest_intermediate_dropout,
                attention_probs_dropout_prob=config.dest_attention_dropout,
                relative_position_embedding=relative_position_embedding,
                rel_index=rel_index
            )

          with tf.variable_scope("fullconnected_resnet"):
            layer_output = self.fullconnected_resnet(
                attention_output=attention_output,
                hidden_size=config.dest_hidden_size,
                intermediate_size=config.dest_intermediate_size,
                hidden_act=config.hidden_act,
                hidden_dropout_prob=config.dest_intermediate_dropout
            )
            # set to prev
            prev_output = layer_output
      return tf.reshape(prev_output, output_shape)

  def get_language_model_mask(self, input_tensor):
    '''
    get a lower diag like
    1 0 0 0 0
    1 1 0 0 0
    1 1 1 0 0
    1 1 1 1 0
    1 1 1 1 1
    '''
    input_shape = mu.get_shape_list(input_tensor)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    # all "1"
    diag_vals = tf.ones([seq_length, seq_length], tf.float32)
    # make a lower diag: [seq_length, seq_length]
    mask = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
    # extend [1, seq_length, seq_length]
    mask = tf.expand_dims(mask, axis=0)
    # tile [batch_size, seq_length, seq_length]
    mask = tf.tile(mask, [batch_size, 1, 1])

    return mask

  def build_encoder(self,
                    config,
                    source_input,
                    is_source_input_onehot,
                    source_mask):
    '''
    build encoder network
    '''
    with tf.variable_scope("encoder"):
      # embedding lookup
      embedding_output, self.encoder_embedding_table = self.embedding_lookup(
          config.source_vocab_size,
          config.source_hidden_size,
          source_input,
          is_source_input_onehot
      )
      # use position embedding (use relative positional encoding)
      # embedding_output = self.apply_absolute_positional_embedding(
      #     embedding_output,
      #     config.is_position_embedding_trainable,
      #     config.max_position_embeddings
      # )
      # drop-out
      # here we use "source_attention_dropout" to sync with attention drop out
      embedding_output = mu.dropout(
          embedding_output,
          config.source_attention_dropout)

      # !!! important
      # different from the BERT implementation, here we do NOT apply layer normalization for the embedding

      # make embedding mask
      attention_mask = None
      if source_mask is not None:
        attention_mask = self.create_attention_mask_from_input_mask(
            embedding_output, source_mask)

      # next: transformer layers
      # from now on, all tensors are flatten (2d) for speed-up
      output_shape = mu.get_shape_list(embedding_output)
      batch_size = output_shape[0]
      encoder_seq_length = output_shape[1]
      # change to 2d
      prev_output = mu.reshape_to_matrix(embedding_output)
      # NOT sure if layer_norm is needed here or NOT
      prev_output = mu.layer_norm(prev_output)

      # create relative position encoding
      relative_position_embedding_all, rel_index = self.create_relative_positional_encoding(
        layer_num=config.source_hidden_layers,
        max_position_embeddings=config.max_position_embeddings,
        num_attention_heads=config.source_attention_heads,
        seq_length=encoder_seq_length,
        width=config.source_hidden_size
      )
      for layer_index in range(config.source_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_index):
          with tf.variable_scope("multihead_attention"):
            # fetch relative position embedding
            relative_position_embedding = tf.squeeze(
              relative_position_embedding_all[layer_index:(layer_index+1),:,:,:], 
              axis=0)
            # multi-head attention on self
            attention_output = self.multihead_attention_resnet_with_relative_positional_encoding(
                prev_output=prev_output,
                batch_size=batch_size,
                from_seq_length=encoder_seq_length,
                to_seq_length=encoder_seq_length,
                from_hidden_size=config.source_hidden_size,
                num_attention_heads=config.source_attention_heads,
                attention_mask=attention_mask,
                hidden_dropout_prob=config.source_intermediate_dropout,
                attention_probs_dropout_prob=config.source_attention_dropout,
                relative_position_embedding=relative_position_embedding,
                rel_index=rel_index
            )

          with tf.variable_scope("fullconnected_resnet"):
            layer_output = self.fullconnected_resnet(
                attention_output=attention_output,
                hidden_size=config.source_hidden_size,
                intermediate_size=config.source_intermediate_size,
                hidden_act=config.hidden_act,
                hidden_dropout_prob=config.source_intermediate_dropout
            )
            # set to prev
            prev_output = layer_output
      # change back shape

      #[B*L,1]
      prev_output_res = tf.layers.dense(
        prev_output,
        1,
        bias_initializer=tf.zeros_initializer(),
        activation=None)

      #[B*L,1]
      output_gate=tf.sigmoid(prev_output_res)
      
      
      # [B*L,1] -> [B,1,L]
      output_gate_rs = tf.reshape(output_gate, (output_shape[0],1,output_shape[1]))
      # [B*L,H] -> [B,L,H]
      prev_output_rs = tf.reshape(prev_output, output_shape)
      # [B,1,L] * [B,L,H] -> [B,1,H] -> [B,H]
      output_pooing = tf.squeeze(tf.matmul(output_gate_rs, prev_output_rs), axis=1)

      #[B,H] -> [B,vocab_size]
      midwords_logits = tf.layers.dense(
        output_pooing,
        config.source_vocab_size,
        bias_initializer=tf.zeros_initializer(),
        use_bias=False)

      prev_output=tf.reshape(prev_output, output_shape)
      return prev_output, output_gate, midwords_logits

  def fullconnected_resnet(self,
                           attention_output,
                           hidden_size,
                           intermediate_size,
                           hidden_act,
                           hidden_dropout_prob
                           ):
    # intermediate layer
    intermediate_output = tf.layers.dense(
        attention_output,
        intermediate_size,
        activation=mu.get_activation_func(hidden_act))
    # down projection
    layer_output = tf.layers.dense(
        intermediate_output,
        hidden_size)
    # drop out
    layer_output = mu.dropout(
        layer_output, hidden_dropout_prob)
    # resnet
    return mu.layer_norm(attention_output + layer_output)

  def create_relative_positional_encoding(
      self,
      layer_num,
      max_position_embeddings,
      num_attention_heads,
      seq_length,
      width):

    def create_position_index(seq_length, half_win, embedding_count):
      '''
      seq_length is a tensor
      '''
      idxs = tf.range(half_win + 1 - seq_length, half_win + seq_length)
      # prevent index from out_of_range
      idxs_shape = mu.get_shape_list(idxs)
      idxs = tf.maximum(idxs, tf.zeros(idxs_shape, dtype=tf.int32))
      idxs = tf.minimum(idxs, tf.zeros(
          idxs_shape, dtype=tf.int32) + embedding_count - 1)
      return idxs

    def generate_positional_embedding(
        layer_num,
        seq_length,
        width
    ):
      # define relative embedding
      #   "max_position_embeddings" defines the maximum relative position boundary, inside which different position means something
      #   at the embedding level, we need the embedding dimension to be 2 * seq_length - 1
      #   if max_position_embeddings < 2 * seq_length - 1, a replicated embedding will be assigned at both sides
      # return: 2*seq-1,position_embedding
      half_win = int(max_position_embeddings / 2)
      actual_max_position_embeddings = half_win * 2 + 1
      # embedding vectors
      relative_position_embedding = tf.get_variable(
          shape=[layer_num, actual_max_position_embeddings, width],
          name='relative_position_embedding',
          initializer=tf.contrib.layers.xavier_initializer())
      # map to the real index
      embedding_indices_tensor = create_position_index(
          seq_length, half_win, actual_max_position_embeddings)
      # fetch embeddings, shape = [layer_num, 2 * seq_length - 1, width]
      relative_position_embedding = tf.gather(
          relative_position_embedding,
          embedding_indices_tensor,
          axis=1)
      # tf.nn.embedding_lookup()
      # important! do layer_norm to align with the to_tensor
      #   key = concat(to, relative_position) * w + b
      return mu.layer_norm(relative_position_embedding)

    def get_relative_embedding_index_tensor(seq_length, relative_embedding_count):
      '''
      get relative embedding index tensor
      '''
      # now, seq_length is a tensor
      tensor_i = tf.range(0, seq_length) * (relative_embedding_count - 1)
      tensor_j = tf.range(0, seq_length) + (seq_length - 1)
      # do tile
      tensor_i = tf.reshape(tf.tile(tf.expand_dims(
          tensor_i, axis=-1), [1, seq_length]), [-1])
      tensor_j = tf.tile(tensor_j, [seq_length])
      # done!
      return tensor_i + tensor_j


    #
    # merge all relative position encoding generation op here
    #
    # calculate head size
    size_per_head = int(math.floor(width / num_attention_heads))
    if size_per_head * num_attention_heads != width:
      raise ValueError(
        "attention head number MUST be a divisor of the width. num_attention_heads = %d, from_hidden_size = %d" % (
          num_attention_heads, width))
    # generate relative position encoding for all layers
    relative_position_embedding = generate_positional_embedding(
      layer_num,
      seq_length,
      width
    )
    relative_embedding_count = mu.get_shape_list(relative_position_embedding)[1]
    # projection, shape = [layer_num, 2 * seq_length - 1, num_attention_heads * size_per_head]
    relative_position_embedding = tf.layers.dense(
        inputs=relative_position_embedding,
        units=num_attention_heads * size_per_head,
        use_bias=False,
        name="relative_position_embedding_linear_projection")
    relative_position_embedding = tf.reshape(
        relative_position_embedding,
        [layer_num, relative_embedding_count, num_attention_heads, size_per_head])
    # shape = [layer_num, num_attention_heads, relative_embedding_count, size_per_head]
    relative_position_embedding_forquery = tf.transpose(
        relative_position_embedding, [0, 2, 1, 3])
    # create relative index
    rel_index = get_relative_embedding_index_tensor(seq_length, relative_embedding_count)

    return relative_position_embedding_forquery, rel_index


  def multihead_attention_resnet(self,
                                 prev_output,
                                 from_tensor,
                                 to_tensor,
                                 batch_size,
                                 from_seq_length,
                                 to_seq_length,
                                 from_hidden_size,
                                 num_attention_heads,
                                 attention_mask,
                                 attention_probs_dropout_prob,
                                 hidden_dropout_prob,
                                 do_return_2d_tensor=True):
    # multihead attention, result tensor is 2d
    attention_output = self.multihead_attention(
        from_tensor=from_tensor,
        to_tensor=to_tensor,
        batch_size=batch_size,
        from_seq_length=from_seq_length,
        to_seq_length=to_seq_length,
        from_hidden_size=from_hidden_size,
        num_attention_heads=num_attention_heads,
        attention_mask=attention_mask,
        attention_probs_dropout_prob=attention_probs_dropout_prob
    )
    # linear projection (without bias)
    attention_output = tf.layers.dense(
        inputs=attention_output,
        units=from_hidden_size,
        use_bias=False)
    # dropout
    attention_output = mu.dropout(
        attention_output, hidden_dropout_prob)
    # res-net
    return mu.layer_norm(prev_output + attention_output)

  def multihead_attention_resnet_with_relative_positional_encoding(self,
                                                                   prev_output,
                                                                   batch_size,
                                                                   from_seq_length,
                                                                   to_seq_length,
                                                                   from_hidden_size,
                                                                   num_attention_heads,
                                                                   attention_mask,
                                                                   attention_probs_dropout_prob,
                                                                   hidden_dropout_prob,
                                                                   relative_position_embedding,
                                                                   rel_index,
                                                                   do_return_2d_tensor=True):
    # multihead attention, result tensor is 2d
    attention_output = self.multihead_attention_with_relative_positional_encoding(
        input_tensor=prev_output,
        batch_size=batch_size,
        from_seq_length=from_seq_length,
        to_seq_length=to_seq_length,
        from_hidden_size=from_hidden_size,
        num_attention_heads=num_attention_heads,
        attention_mask=attention_mask,
        relative_position_embedding=relative_position_embedding,
        rel_index=rel_index,
        attention_probs_dropout_prob=attention_probs_dropout_prob
    )
    # linear projection (without bias)
    attention_output = tf.layers.dense(
        inputs=attention_output,
        units=from_hidden_size,
        use_bias=False)
    # dropout
    attention_output = mu.dropout(
        attention_output, hidden_dropout_prob)
    # res-net
    return mu.layer_norm(prev_output + attention_output)

  def attention_pooling_gate(self,
                        from_tensor,
                        to_tensor,
                        gate_tensor,
                        batch_size,
                        from_seq_length,
                        to_seq_length,
                        from_hidden_size,
                        to_hidden_size,
                        attention_mask,
                        attention_probs_dropout_prob,
                        do_return_2d_tensor=True
                        ):
    '''
    attention pooling implementation
    from_tensor has shape [batch_size x from_seq_length, from_hidden_size]
    to_tensor has shape [batch_size x to_seq_length, to_hidden_size]
    '''
    # `query_layer` = [batch_size, from_seq_length, from_hidden_size]
    query_layer = tf.layers.dense(
        inputs=from_tensor,
        units=from_hidden_size,
        use_bias=False,
        name="query")
    query_layer = tf.reshape(
        query_layer, [batch_size, from_seq_length, from_hidden_size])

    # `key_layer` = [batch_size, to_seq_length, from_hidden_size]
    key_layer = tf.layers.dense(
        inputs=to_tensor,
        units=from_hidden_size,
        use_bias=False,
        name="key")
    key_layer = tf.reshape(
        key_layer, [batch_size, to_seq_length, to_hidden_size])

    # `value_layer` = [batch_size, to_seq_length, from_hidden_size]
    value_layer = to_tensor
    value_layer = tf.reshape(
        value_layer, [batch_size, to_seq_length, to_hidden_size])

    # `attention_scores` = [batch_size, from_seq_length,  to_seq_length]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(from_hidden_size)))

    if attention_mask is not None:
      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -100000.0

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [batch_size, from_seq_length, to_seq_length]
    attention_probs_b = tf.nn.softmax(attention_scores)

    gate_tensor=tf.reshape(gate_tensor,[-1,1,to_seq_length])

    attention_probs_b=tf.math.multiply(gate_tensor,attention_probs_b)


    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = mu.dropout(attention_probs_b, attention_probs_dropout_prob)

    # `context_layer` = [batch_size, from_seq_length, from_hidden_size]
    context_layer = tf.matmul(attention_probs, value_layer)

    if do_return_2d_tensor:
      # `context_layer` = [B*F, N*V]
      context_layer = tf.reshape(
          context_layer,
          [batch_size * from_seq_length, to_hidden_size])
    else:
      # `context_layer` = [B, F, N*V]
      context_layer = tf.reshape(
          context_layer,
          [batch_size, from_seq_length, to_hidden_size])
    return context_layer, attention_probs_b

  def attention_pooling(self,
                        from_tensor,
                        to_tensor,
                        batch_size,
                        from_seq_length,
                        to_seq_length,
                        from_hidden_size,
                        to_hidden_size,
                        attention_mask,
                        attention_probs_dropout_prob,
                        do_return_2d_tensor=True
                        ):
    '''
    attention pooling implementation
    from_tensor has shape [batch_size x from_seq_length, from_hidden_size]
    to_tensor has shape [batch_size x to_seq_length, to_hidden_size]
    '''
    # `query_layer` = [batch_size, from_seq_length, from_hidden_size]
    query_layer = tf.layers.dense(
        inputs=from_tensor,
        units=from_hidden_size,
        use_bias=False,
        name="query")
    query_layer = tf.reshape(
        query_layer, [batch_size, from_seq_length, from_hidden_size])

    # `key_layer` = [batch_size, to_seq_length, from_hidden_size]
    key_layer = tf.layers.dense(
        inputs=to_tensor,
        units=from_hidden_size,
        use_bias=False,
        name="key")
    key_layer = tf.reshape(
        key_layer, [batch_size, to_seq_length, to_hidden_size])

    # `value_layer` = [batch_size, to_seq_length, from_hidden_size]
    value_layer = to_tensor
    value_layer = tf.reshape(
        value_layer, [batch_size, to_seq_length, to_hidden_size])

    # `attention_scores` = [batch_size, from_seq_length,  to_seq_length]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(from_hidden_size)))

    if attention_mask is not None:
      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -100000.0

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [batch_size, from_seq_length, to_seq_length]
    attention_probs_b = tf.nn.softmax(attention_scores)



    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = mu.dropout(attention_probs_b, attention_probs_dropout_prob)

    # `context_layer` = [batch_size, from_seq_length, from_hidden_size]
    context_layer = tf.matmul(attention_probs, value_layer)

    if do_return_2d_tensor:
      # `context_layer` = [B*F, N*V]
      context_layer = tf.reshape(
          context_layer,
          [batch_size * from_seq_length, to_hidden_size])
    else:
      # `context_layer` = [B, F, N*V]
      context_layer = tf.reshape(
          context_layer,
          [batch_size, from_seq_length, to_hidden_size])
    return context_layer, attention_probs_b

  def multihead_attention_with_relative_positional_encoding(self,
                                                            input_tensor,
                                                            batch_size,
                                                            from_seq_length,
                                                            to_seq_length,
                                                            from_hidden_size,
                                                            num_attention_heads,
                                                            attention_mask,
                                                            attention_probs_dropout_prob,
                                                            relative_position_embedding,
                                                            rel_index,
                                                            do_return_2d_tensor=True
                                                            ):
    '''
    relative positional encoding can ONLY be used for self-attention!!!
    '''
    # for self-attention only
    if from_seq_length != to_seq_length:
      raise ValueError(
        "relative positional encoding can ONLY be used for self-attention. from_seq_length = %d, to_seq_length = %d" %
        (from_seq_length, to_seq_length))
    # calculate head size
    size_per_head = int(math.floor(from_hidden_size / num_attention_heads))
    if size_per_head * num_attention_heads != from_hidden_size:
      raise ValueError(
        "attention head number MUST be a divisor of the from_hidden_size. num_attention_heads = %d, from_hidden_size = %d" % (
          num_attention_heads, from_hidden_size))
    
    # linear projection of Q,K,V is done in one dense layer
    # output shape = [batch_size, to_seq_length, num_attention_heads * size_per_head * 3]
    projected_tensor = tf.layers.dense(
        inputs=input_tensor,
        units=num_attention_heads * size_per_head * 3,
        use_bias=False,
        name="projected_tensor")
    # reshape to [batch_size, to_seq_length, 3, num_attention_heads, size_per_head]
    projected_tensor = tf.reshape(projected_tensor,
      [batch_size, to_seq_length, 3, num_attention_heads, size_per_head])
    # transpose to [3, num_attention_heads, batch_size, to_seq_length, size_per_head]
    projected_tensor = tf.transpose(projected_tensor,
      [2,3,0,1,4])
    
    # shape = [num_attention_heads, batch_size, to_seq_length,  size_per_head]
    query_layer = tf.squeeze(projected_tensor[0:1,:,:,:,:], axis=0)
    key_layer = tf.squeeze(projected_tensor[1:2,:,:,:,:], axis=0)
    value_layer = tf.squeeze(projected_tensor[2:3,:,:,:,:], axis=0)

    # shape = [num_attention_heads, batch_size, to_seq_length, to_seq_length]
    attention_context = tf.matmul(query_layer, key_layer, transpose_b=True)

    # *********** relative position attention ***********
    # calculate a set of redundant attention scores for all possible positions
    # shape = [num_attention_heads, batch_size * to_seq_length, size_per_head]
    rel_query = tf.reshape(query_layer, [num_attention_heads, -1, size_per_head])
    # shape = [num_attention_heads, batch_size * to_seq_length, relative_embedding_count]
    attention_embedding = tf.matmul(
        rel_query, relative_position_embedding, transpose_b=True)
    # try to gather the attention value at right positions
    pos_indices_tensor = rel_index
    # gather attention scores, shape = [num_attention_heads, batch_size, to_seq_length * to_seq_length]
    attention_embedding = tf.gather(
        tf.reshape(attention_embedding, [num_attention_heads, batch_size, -1]),
        pos_indices_tensor,
        axis=2
    )
    attention_embedding = tf.reshape(attention_embedding, [
      num_attention_heads, batch_size, to_seq_length, to_seq_length])

    # sum attention scores
    attention_scores = attention_context + attention_embedding
    # normalize
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

      # shape = [1, batch_size, to_seq_length, to_seq_length]
      adder = tf.expand_dims(adder, axis=0)

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [N, B, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = mu.dropout(attention_probs, attention_probs_dropout_prob)

    # *********** now we need to calculate the attention result ***********
    # shape = [num_attention_heads, batch_size, to_seq_length, size_per_head]
    context_layer = tf.matmul(attention_probs, value_layer)
    # reshape for concat
    context_layer = tf.transpose(context_layer, [1, 2, 0, 3])

    if do_return_2d_tensor:
      # `context_layer` = [B*F, N*V]
      context_layer = tf.reshape(
          context_layer,
          [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
      # `context_layer` = [B, F, N*V]
      context_layer = tf.reshape(
          context_layer,
          [batch_size, from_seq_length, num_attention_heads * size_per_head])
    return context_layer

  def multihead_attention(self,
                          from_tensor,
                          to_tensor,
                          batch_size,
                          from_seq_length,
                          to_seq_length,
                          from_hidden_size,
                          num_attention_heads,
                          attention_mask,
                          attention_probs_dropout_prob,
                          do_return_2d_tensor=True
                          ):
    '''
    multihead attention implementation
    from_tensor has shape [batch_size x from_seq_length, from_hidden_size]
    to_tensor has shape [batch_size x to_seq_length, to_hidden_size]
    '''
    # calculate head size
    size_per_head = int(math.floor(from_hidden_size / num_attention_heads))
    if size_per_head * num_attention_heads != from_hidden_size:
      raise ValueError(
        "attention head number MUST be a divisor of the from_hidden_size. num_attention_heads = %d, from_hidden_size = %d" % (
          num_attention_heads, from_hidden_size))
    # `query_layer` = [batch_size, from_seq_length, num_attention_heads * size_per_head]
    query_layer = tf.layers.dense(
        inputs=from_tensor,
        units=num_attention_heads * size_per_head,
        use_bias=False,
        name="query")

    # `key_layer` = [batch_size, to_seq_length, num_attention_heads * size_per_head]
    key_layer = tf.layers.dense(
        inputs=to_tensor,
        units=num_attention_heads * size_per_head,
        use_bias=False,
        name="key")

    # `value_layer` = [batch_size, to_seq_length, num_attention_heads * size_per_head]
    value_layer = tf.layers.dense(
        inputs=to_tensor,
        units=num_attention_heads * size_per_head,
        use_bias=False,
        name="value")

    # `query_layer` = [batch_size, num_attention_heads, from_seq_length,  size_per_head]
    query_layer = tf.reshape(
        query_layer, [batch_size, from_seq_length, num_attention_heads, size_per_head])
    query_layer = tf.transpose(query_layer, [0, 2, 1, 3])

    # `key_layer` = [batch_size, num_attention_heads, to_seq_length,  size_per_head]
    key_layer = tf.reshape(
        key_layer, [batch_size, to_seq_length, num_attention_heads, size_per_head])
    key_layer = tf.transpose(key_layer, [0, 2, 1, 3])

    # `attention_scores` = [batch_size, num_attention_heads, from_seq_length,  to_seq_length]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
      # `attention_mask` = [batch_size, 1, from_seq_length,  to_seq_length]
      attention_mask = tf.expand_dims(attention_mask, axis=[1])

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -100000.0

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [batch_size, num_attention_heads, from_seq_length,  to_seq_length]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = mu.dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [batch_size, num_attention_heads, to_seq_length,  size_per_head]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [batch_size, num_attention_heads, from_seq_length,  size_per_head]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [batch_size, from_seq_length,  num_attention_heads, size_per_head]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
      # `context_layer` = [B*F, N*V]
      context_layer = tf.reshape(
          context_layer,
          [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
      # `context_layer` = [B, F, N*V]
      context_layer = tf.reshape(
          context_layer,
          [batch_size, from_seq_length, num_attention_heads * size_per_head])
    return context_layer

  def create_attention_mask_from_input_mask(self, from_tensor, to_mask):
    '''
    This converts a 2D mask of shape [batch_size, seq_length] to a 3D
    mask of shape [batch_size, seq_length, seq_length] which is used
    for the attention scores.
    # 1 1 1 0 0
    # 1 1 1 0 0
    # 1 1 1 0 0
    # 1 1 1 0 0

    for self attention: from_tensor and to_mask are from one same source
    for memory attention: from_tensor is decoder tensor and to_mask is the encoder mask
    '''
    from_shape = mu.get_shape_list(from_tensor)
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = mu.get_shape_list(to_mask)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]),
        tf.float32)
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    return tf.matmul(broadcast_ones, to_mask)

  def apply_absolute_positional_embedding(self,
                                          embedding_input_tensor,
                                          is_position_embedding_trainable,
                                          max_position_embeddings
                                          ):
    '''
    when is_position_embedding_trainable = 1, use trainable position embeddings, otherwise fix them
    '''
    input_shape = mu.get_shape_list(embedding_input_tensor)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    hidden_size = input_shape[2]
    # max_position_embeddings must be larger than seq_length
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      # use fixed positional embeddings or trainable ones
      # in theory, the two choices are almost the same
      if is_position_embedding_trainable == 1:
        # use trainable positional embedding
        full_position_embeddings = tf.get_variable(
            name='full_position_embeddings',
            shape=[max_position_embeddings, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer())
      else:
        # use fixed positional embedding
        full_position_embeddings = self.fixed_positional_encoding(
            max_position_embeddings, hidden_size)
      # apply lookup
      pos_index = tf.range(start=0, limit=seq_length)
      # shape = [seq_length, hidden_size]
      position_embedding = tf.nn.embedding_lookup(
          full_position_embeddings, pos_index)
      # tile it through the batch dim
      position_embedding = tf.tile(
          tf.expand_dims(position_embedding, axis=0),
          [batch_size, 1, 1]
      )
      # add and return
      return embedding_input_tensor + position_embedding

  def fixed_positional_encoding(self, max_position_embeddings, hidden_size):
    '''
    max_position_embeddings: number of positions
    hidden_size: embedding size
    '''
    position_encoding = np.array([
      [pos / np.power(10000, (i - i % 2) / hidden_size)
       for i in range(hidden_size)]
      for pos in range(max_position_embeddings)])
    # apply the cosine to even columns and sin to odds.
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
    position_embedding_tensor = tf.convert_to_tensor(
        position_encoding, tf.float32)
    # shape = [max_position_embeddings, hidden_size]
    return position_embedding_tensor

  def embedding_lookup(self,
                       vocab_size,
                       embedding_size,
                       input_tensor,
                       is_input_onehot,
                       embedding_table=None,
                       scale=True
                       ):
    '''
    embedding lookup: one hot or dense
    '''
    # build embedding table
    if embedding_table is None:
      embedding_table = tf.get_variable(
          name='embedding_table',
          shape=[vocab_size, embedding_size],
          initializer=tf.contrib.layers.xavier_initializer())

    if is_input_onehot:
      # one hot input, input shape is [batch_size, seq_length]
      # output shape is [batch_size, seq_length, embedding_size]
      output = tf.nn.embedding_lookup(embedding_table, input_tensor)
    else:
      # input is not one hot, input shape is [batch_size, seq_length, vocab_size]
      # output shape is [batch_size, seq_length, embedding_size]
      output = tf.einsum('ntd,dk->ntk', input_tensor, embedding_table)

    # scale embedding with d_model (=embedding_size) ** 0.5
    if scale:
      output *= embedding_size ** 0.5

    return output, embedding_table
