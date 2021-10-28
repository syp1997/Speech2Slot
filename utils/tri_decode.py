# newwww
# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Utility functions
'''

import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow
import numpy as np
import json
import os, re
import logging
import math
from utils.compress_decode import load_bytes_to_ram, get_one_instance_x_65536, get_one_instance_x_256, get_one_instance_y_asr20
import pickle

def load_vocab(vocab_fpath):
  '''Loads vocabulary file and returns idx<->token maps
  vocab_fpath: string. vocabulary file path.
  Note that these are reserved
  0: <pad>, 1: <unk>, 2: <s>, 3: </s>

  Returns
  two dictionaries.
  '''
  vocab = [line.split()[0] for line in open(vocab_fpath, 'r').read().splitlines()]
  token2idx = {token: idx for idx, token in enumerate(vocab)}
  idx2token = {idx: token for idx, token in enumerate(vocab)}
  return token2idx, idx2token


def postprocess(hypotheses, idx2token):
  '''Processes translation outputs.
  hypotheses: list of encoded predictions
  idx2token: dictionary

  Returns
  processed hypotheses
  '''
  _hypotheses = []
  for h in hypotheses:
    sent = " ".join(idx2token[idx] for idx in h)
    sent = sent.split("</s>")[0].strip()
    sent = sent.replace("<s>", "")
    # sent = sent.replace("▁", " ") # remove bpe symbols
    _hypotheses.append(sent.strip())
  return _hypotheses

def postprocess_with_score(hypotheses, idx2token, res_score):
  '''Processes translation outputs.
  hypotheses: list of encoded predictions
  idx2token: dictionary

  Returns
  processed hypotheses
  '''
  _hypotheses = []
  for h, s in zip(hypotheses, res_score):
    sent = " ".join(idx2token[idx] for idx in h)
    sent = sent.split("</s>")[0].strip()
    sent = sent.replace("<s>", "")
    sent += "\t" + str(s)
    # sent = sent.replace("▁", " ") # remove bpe symbols
    _hypotheses.append(sent.strip())
  return _hypotheses


def get_real_label_score_mid(phone_input, 
                             decoder_input, 
                             decoder_groundtruth, 
                             reverse_decoder_input, 
                             reverse_decoder_groundtruth, 
                             m2l_decoder_input, 
                             m2l_decoder_groundtruth, 
                             m2r_decoder_input, 
                             m2r_decoder_groundtruth, 
                             model, 
                             sess, 
                             idx2token):
  history_score_ = [[] for i in range(np.shape(phone_input)[0])]
  back_history_score_ = [[] for i in range(np.shape(phone_input)[0])]
  m2l_history_score_ = [[] for i in range(np.shape(phone_input)[0])]
  m2r_history_score_ = [[] for i in range(np.shape(phone_input)[0])]
  logits_, back_logits_, m2l_logits_, m2r_logits_ = model.infer(sess, 
                                      phone_input=phone_input, 
                                      decoder_input=decoder_input, 
                                      reverse_decoder_input=reverse_decoder_input,
                                      m2l_decoder_input=m2l_decoder_input, 
                                      m2r_decoder_input=m2r_decoder_input)
  logits_ = np.asarray(logits_)
  back_logits_ = np.asarray(logits_)
  m2l_logits_ = np.asarray(m2l_logits_)
  m2r_logits_ = np.asarray(m2r_logits_)

  dim1 = np.shape(phone_input)[0]
  dim2 = int(logits_.shape[0] / dim1)
  dim3 = int(m2l_logits_.shape[0] / dim1)
  #[B,S,C]
  logits_ = np.reshape(logits_, (dim1, dim2, len(idx2token)-1))
  for i in range(len(decoder_groundtruth[0])):
    real = decoder_groundtruth[:, i]
    for jj in range(len(decoder_groundtruth)):
      if real[jj] != 0:
        history_score_[jj].append(softmax(logits_[jj, i])[int(real[jj])])
    if np.sum(real) == 0:
      break

  back_logits_ = np.reshape(back_logits_, (dim1, dim2, len(idx2token)-1))
  for i in range(len(reverse_decoder_groundtruth[0])):
    real = reverse_decoder_groundtruth[:, i]
    for jj in range(len(reverse_decoder_groundtruth)):
      if real[jj] != 0:
        back_history_score_[jj].append(softmax(back_logits_[jj, i])[int(real[jj])])
    if np.sum(real) == 0:
      break
      
  m2l_logits_ = np.reshape(m2l_logits_, (dim1, dim3, len(idx2token)-1))
  for i in range(len(m2l_decoder_groundtruth[0])):
    real = m2l_decoder_groundtruth[:, i]
    for jj in range(len(m2l_decoder_groundtruth)):
      if real[jj] != 0:
        m2l_history_score_[jj].append(softmax(m2l_logits_[jj, i])[int(real[jj])])
    if np.sum(real) == 0:
      break
      
  m2r_logits_ = np.reshape(m2r_logits_, (dim1, dim3, len(idx2token)-1))
  for i in range(len(m2r_decoder_groundtruth[0])):
    real = m2r_decoder_groundtruth[:, i]
    for jj in range(len(m2r_decoder_groundtruth)):
      if real[jj] != 0:
        m2r_history_score_[jj].append(softmax(m2r_logits_[jj, i])[int(real[jj])])
    if np.sum(real) == 0:
      break

  return history_score_, back_history_score_, m2l_history_score_, m2r_history_score_


def get_hypotheses_beam_mid(phone_input,
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
                        reverse_trietree,
                        num_beam,
                        idx2token,
                        token2idx,
                        handler):
  '''Gets hypotheses.
  num_batches: scalar.
  num_samples: scalar.
  sess: tensorflow sess object
  tensor: target tensor to fetch
  dict: idx2token dictionary

  Returns
  hypotheses: list of sents
  '''
  hypotheses = []
  phone_input = np.asarray(phone_input)
  decoder_input = np.asarray(decoder_input)
  decoder_groundtruth = np.asarray(decoder_groundtruth)
  m2l_decoder_input = np.asarray(m2l_decoder_input)
  m2l_decoder_groundtruth = np.asarray(m2l_decoder_groundtruth)
  m2r_decoder_input = np.asarray(m2r_decoder_input)
  m2r_decoder_groundtruth = np.asarray(m2r_decoder_groundtruth)
  
  # input is ground truth decoder input
  real_label_score, back_real_label_score, m2l_real_label_score, m2r_real_label_score = get_real_label_score_mid(phone_input, decoder_input, decoder_groundtruth, reverse_decoder_input, reverse_decoder_groundtruth, m2l_decoder_input, m2l_decoder_groundtruth, m2r_decoder_input, m2r_decoder_groundtruth, model, sess, idx2token)

  # beam search
  phone_input_extend = []
  for item in phone_input:
    for i in range(num_beam):
      phone_input_extend.append(item)

  pos = np.zeros(num_beam)
  forward_pos = np.zeros(num_beam)
  backward_pos = np.zeros(num_beam)
  middle_pos = np.zeros(num_beam)
  middle_pos_v2 = np.zeros(num_beam)
  
  stop = False
  forward_stop = False
  backward_stop = False
  m2l_stop = False
  m2r_stop = False
  
  # [B*num_beam,1]
  decoder_input_pred = np.ones((np.shape(phone_input_extend)[0], 1), np.int32) * token2idx["<s>"]
  back_decoder_input_pred = np.ones((np.shape(phone_input_extend)[0], 1), np.int32) * token2idx["<s>"]
#   m2l_decoder_input_pred = np.ones((np.shape(phone_input_extend)[0], 1), np.int32) * token2idx["<s>"]
#   m2r_decoder_input_pred = np.ones((np.shape(phone_input_extend)[0], 1), np.int32) * token2idx["<s>"]
  m2l_decoder_input_pred = np.zeros((np.shape(phone_input_extend)[0], 1), np.int32)
  m2r_decoder_input_pred = np.zeros((np.shape(phone_input_extend)[0], 1), np.int32)
  
  history_score = [[] for i in range(np.shape(phone_input_extend)[0])]
  back_history_score = [[] for i in range(np.shape(phone_input_extend)[0])]
  m2l_history_score = [[] for i in range(np.shape(phone_input_extend)[0])]
  m2r_history_score = [[] for i in range(np.shape(phone_input_extend)[0])]
  
  # get middle word
  # softmax
  use_softmax = True
  midword_logits = model.get_midword_logits(sess, phone_input)
  midword_logits = np.asarray(midword_logits[0])
  if use_softmax:
    midword_score = sess.run(tf.nn.softmax(midword_logits, axis=-1))
  else:
    midword_score = sess.run(tf.math.sigmoid(midword_logits))
  
  mid_corr1 = 0
  mid_corr2 = 0
  mid_corr3 = 0
  for i in range(len(midword_score)):
    word_index = midword_score[i].argsort()[::-1][:num_beam]
    if word_index[0] in decoder_groundtruth[i]:
      mid_corr1 += 1
      mid_corr2 += 1
      mid_corr3 += 1
    elif word_index[1] in decoder_groundtruth[i]:
      mid_corr2 += 1
      mid_corr3 += 1
    elif word_index[2] in decoder_groundtruth[i]:
      mid_corr3 += 1
  
  first_beam_search = True
  if first_beam_search:
    for i in range(0, m2l_decoder_input_pred.shape[0], num_beam):
      word_index = midword_score[i//num_beam].argsort()[::-1][:num_beam]
      word_scores = midword_score[i//num_beam][word_index]
      for j in range(num_beam):
        m2l_decoder_input_pred[i+j] = word_index[j]
        m2r_decoder_input_pred[i+j] = word_index[j]
        m2l_history_score[i+j].append(word_scores[j])
        m2r_history_score[i+j].append(word_scores[j])     
  else:
    for i in range(0, m2l_decoder_input_pred.shape[0], num_beam):
      word_idx = midword_score[i//num_beam].argsort()[::-1][0]
      word_score = midword_score[i//num_beam][word_index]
      for j in range(num_beam):
        m2l_decoder_input_pred[i+j] = word_idx
        m2r_decoder_input_pred[i+j] = word_idx
        m2l_history_score[i+j].append(word_score)
        m2r_history_score[i+j].append(word_score)
        
  
  while not (forward_stop and backward_stop and m2l_stop and m2r_stop):
    # history=convert_to_history(decoder_input_pred, idx2token)
    decoder_inputs_all = decoder_input_pred
    back_decoder_inputs_all = back_decoder_input_pred
    m2l_decoder_inputs_all = m2l_decoder_input_pred
    m2r_decoder_inputs_all = m2r_decoder_input_pred
    
    if np.shape(decoder_input_pred)[1] < np.shape(decoder_input)[1]:
      padding = np.zeros([np.shape(phone_input_extend)[0], np.shape(decoder_input)[1] - np.shape(decoder_input_pred)[1]])
      decoder_inputs_all = np.concatenate((decoder_input_pred, padding), axis=-1)
    if np.shape(back_decoder_input_pred)[1] < np.shape(decoder_input)[1]:
      padding = np.zeros([np.shape(phone_input_extend)[0], np.shape(decoder_input)[1] - np.shape(back_decoder_input_pred)[1]])
      back_decoder_inputs_all = np.concatenate((back_decoder_input_pred, padding), axis=-1)
    if np.shape(m2l_decoder_input_pred)[1] < np.shape(m2l_decoder_input)[1]:
      padding = np.zeros([np.shape(phone_input_extend)[0], np.shape(m2l_decoder_input)[1] - np.shape(m2l_decoder_input_pred)[1]])
      m2l_decoder_inputs_all = np.concatenate((m2l_decoder_input_pred, padding), axis=-1)
    if np.shape(m2r_decoder_input_pred)[1] < np.shape(m2r_decoder_input)[1]:
      padding = np.zeros([np.shape(phone_input_extend)[0], np.shape(m2r_decoder_input)[1] - np.shape(m2r_decoder_input_pred)[1]])
      m2r_decoder_inputs_all = np.concatenate((m2r_decoder_input_pred, padding), axis=-1)

    logits, back_logits, m2l_logits, m2r_logits = model.infer(sess, 
                                                              phone_input=phone_input_extend, 
                                                              decoder_input=decoder_inputs_all, 
                                                              reverse_decoder_input=back_decoder_inputs_all,
                                                              m2l_decoder_input=m2l_decoder_inputs_all, 
                                                              m2r_decoder_input=m2r_decoder_inputs_all)
    logits = np.asarray(logits)
    back_logits = np.asarray(back_logits)
    m2l_logits = np.asarray(m2l_logits)
    m2r_logits = np.asarray(m2r_logits)

    dim1 = np.shape(phone_input_extend)[0]
    dim2 = int(logits.shape[0] / dim1)
    logits = np.reshape(logits, (dim1, dim2, len(idx2token)-1))
    logits = logits[:, :np.shape(decoder_input_pred)[1], :]
    
    dim1 = np.shape(phone_input_extend)[0]
    dim2 = int(back_logits.shape[0] / dim1)
    back_logits = np.reshape(back_logits, (dim1, dim2, len(idx2token)-1))
    back_logits = back_logits[:, :np.shape(back_decoder_input_pred)[1], :]
    
    dim1 = np.shape(phone_input_extend)[0]
    dim2 = int(m2l_logits.shape[0] / dim1)
    m2l_logits = np.reshape(m2l_logits, (dim1, dim2, len(idx2token)-1))
    m2l_logits = m2l_logits[:, :np.shape(m2l_decoder_input_pred)[1], :]
    
    dim1 = np.shape(phone_input_extend)[0]
    dim2 = int(m2r_logits.shape[0] / dim1)
    m2r_logits = np.reshape(m2r_logits, (dim1, dim2, len(idx2token)-1))
    m2r_logits = m2r_logits[:, :np.shape(m2r_decoder_input_pred)[1], :]

    if not forward_stop:
      decoder_input_pred, history_score, forward_stop = get_res_beam(trietree, logits, token2idx, decoder_input_pred,
                                                           history_score, num_beam)
    if not backward_stop:
      back_decoder_input_pred, back_history_score, backward_stop = get_res_beam(reverse_trietree, back_logits, token2idx, back_decoder_input_pred, back_history_score, num_beam)
    
    if not m2l_stop:
      m2l_decoder_input_pred, m2l_history_score, m2l_stop = get_res_beam(reverse_trietree, m2l_logits, token2idx, m2l_decoder_input_pred, m2l_history_score, num_beam)
      
    if not m2r_stop:
      m2r_decoder_input_pred, m2r_history_score, m2r_stop = get_res_beam(trietree, m2r_logits, token2idx, m2r_decoder_input_pred, m2r_history_score, num_beam)

  forward_res, forward_correctness = test_beam_search_result(decoder_groundtruth, num_beam, decoder_input_pred, forward_pos)
  backward_res, backward_correctness = test_beam_search_result(decoder_groundtruth, num_beam, back_decoder_input_pred, backward_pos, is_reverse=True)
  middle_res, middle_correctness, middle_str2score = test_beam_search_result_mid(decoder_groundtruth, num_beam, m2l_decoder_input_pred, m2r_decoder_input_pred, m2l_history_score, m2r_history_score, middle_pos)
  
  seq_length = np.shape(decoder_input)[1]
#   use_v2 = True
#   if use_v2:
#     decode_input_m_all = []
#     decode_input_m_all_back = []
#     for i in range(len(middle_str2score)):
#       for pred_str, score in middle_str2score[i]:
#         pred = [int(item) for item in pred_str.split(" ")]
#         decode_input_m = [token2idx['<s>']] + pred
#         decode_input_m_back = [token2idx['<s>']] + pred[::-1]
#         if len(decode_input_m) < seq_length:
#           decode_input_m = decode_input_m + [0] * (seq_length-len(decode_input_m))
#         if len(decode_input_m_back) < seq_length:
#           decode_input_m_back = decode_input_m_back + [0] * (seq_length-len(decode_input_m_back))
#         decode_input_m_all.append(decode_input_m[:seq_length])
#         decode_input_m_all_back.append(decode_input_m_back[:seq_length])
#     decode_input_m_all = np.asarray(decode_input_m_all)
#     decode_input_m_all_back = np.asarray(decode_input_m_all_back)

#     logits_m_forward = model.infer_forward(sess, phone_input=phone_input_extend, decoder_input=decode_input_m_all)
#     logits_m_forward = np.asarray(logits_m_forward)
#     logits_m_forward = logits_m_forward.reshape(-1,seq_length,logits_m_forward.shape[-1])
#     logits_m_forward_score = sess.run(tf.nn.softmax(logits_m_forward, axis=-1))

#     logits_m_backward = model.infer_backward(sess, phone_input=phone_input_extend, decoder_input=decode_input_m_all, reverse_decoder_input=decode_input_m_all_back)
#     logits_m_backward = np.asarray(logits_m_backward)
#     logits_m_backward = logits_m_backward.reshape(-1,seq_length,logits_m_backward.shape[-1])
#     logits_m_backward_score = sess.run(tf.nn.softmax(logits_m_backward, axis=-1))

#     middle_res_v2, middle_correctness_v2 = test_beam_search_result_mid_v2(decoder_groundtruth, num_beam, decode_input_m_all, logits_m_forward_score, decode_input_m_all_back, logits_m_backward_score, token2idx, middle_pos_v2)
  
  # mid v3
  l2f_pos = np.zeros(num_beam)
  r2b_pos = np.zeros(num_beam)
  
  l2f_decoder_input_pred = []
  r2b_decoder_input_pred = []
  for i in range(len(decoder_groundtruth)):
    m2l_pred = [int(l) for l in m2l_decoder_input_pred[i * num_beam] if int(l) >= 3]
    m2l_pred.reverse()
    l2f_pred = [token2idx['<s>']] + m2l_pred[:]
    l2f_decoder_input_pred_beam = [l2f_pred for i in range(num_beam)]
    l2f_decoder_input_pred.append(l2f_decoder_input_pred_beam)
  for i in range(len(decoder_groundtruth)):
    r2b_decoder_input_pred_beam = []
    m2r_pred = [int(l) for l in m2r_decoder_input_pred[i * num_beam] if int(l) >= 3]
    m2r_pred.reverse()
    r2b_pred = [token2idx['<s>']] + m2r_pred[:]
    r2b_decoder_input_pred_beam = [r2b_pred for i in range(num_beam)]
    r2b_decoder_input_pred.append(r2b_decoder_input_pred_beam)
  
  l2f_decoder_input_pred_final = []
  r2b_decoder_input_pred_final = []
  l2f_history_score_final = []
  r2b_history_score_final = []
  for i, (l2f_decoder_input_pred_beam, r2b_decoder_input_pred_beam) in enumerate(zip(l2f_decoder_input_pred, r2b_decoder_input_pred)):
    
    l2f_stop = False
    r2b_stop = False
    
    l2f_history_score = [m2l_history_score[i*num_beam] for _ in range(num_beam)]
    r2b_history_score = [m2r_history_score[i*num_beam] for _ in range(num_beam)]
    
    l2f_decoder_input_pred_beam = np.asarray(l2f_decoder_input_pred_beam)
    r2b_decoder_input_pred_beam = np.asarray(r2b_decoder_input_pred_beam)
    
#     l2f_history_score = [item[:l2f_decoder_input_pred_beam.shape[1]] for item in l2f_history_score] 
#     r2b_history_score = [item[:r2b_decoder_input_pred_beam.shape[1]] for item in l2f_history_score]
#     print("l2f_history_score: ", len(l2f_history_score), len(l2f_history_score[0]), l2f_history_score[0])
#     print("r2b_history_score: ", len(r2b_history_score), len(r2b_history_score[0]), r2b_history_score[0])
#     print("l2f_decoder_input_pred_beam.shape1: ", l2f_decoder_input_pred_beam.shape[1])
#     print("r2b_decoder_input_pred_beam.shape1: ", r2b_decoder_input_pred_beam.shape[1])
    
    while not (l2f_stop and r2b_stop):
      l2f_decoder_input_all = l2f_decoder_input_pred_beam
      r2b_decoder_input_all = r2b_decoder_input_pred_beam

#       print("l2f_decoder_input_pred_beam: ", l2f_decoder_input_pred_beam.shape)
#       print("r2b_decoder_input_pred_beam: ", r2b_decoder_input_pred_beam.shape)

      if np.shape(l2f_decoder_input_pred_beam)[1] < seq_length:
        padding = np.zeros([num_beam, seq_length - np.shape(l2f_decoder_input_pred_beam)[1]])
        l2f_decoder_input_all = np.concatenate((l2f_decoder_input_pred_beam, padding), axis=-1)
      if np.shape(r2b_decoder_input_pred_beam)[1] < seq_length:
        padding = np.zeros([num_beam, seq_length - np.shape(r2b_decoder_input_pred_beam)[1]])
        r2b_decoder_input_all = np.concatenate((r2b_decoder_input_pred_beam, padding), axis=-1)
      
      l2f_logits = model.infer_forward(sess, phone_input=phone_input_extend[i*num_beam : i*num_beam+num_beam], decoder_input=l2f_decoder_input_all)
      r2b_logits = model.infer_backward(sess, phone_input=phone_input_extend[i*num_beam : i*num_beam+num_beam], decoder_input=l2f_decoder_input_all, reverse_decoder_input=r2b_decoder_input_all)

      l2f_logits = np.asarray(l2f_logits)
      r2b_logits = np.asarray(r2b_logits)

      dim1 = num_beam
      dim2 = int(l2f_logits.shape[0] / dim1)
      l2f_logits = np.reshape(l2f_logits, (dim1, dim2, len(idx2token)-1))
      l2f_logits = l2f_logits[:, :np.shape(l2f_decoder_input_pred_beam)[1], :]

      dim1 = num_beam
      dim2 = int(r2b_logits.shape[0] / dim1)
      r2b_logits = np.reshape(r2b_logits, (dim1, dim2, len(idx2token)-1))
      r2b_logits = r2b_logits[:, :np.shape(r2b_decoder_input_pred_beam)[1], :]

      if not l2f_stop:
        l2f_decoder_input_pred_beam, l2f_history_score, l2f_stop = get_res_beam(trietree, l2f_logits, token2idx, l2f_decoder_input_pred_beam,
                                                             l2f_history_score, num_beam)
      if not r2b_stop:
        r2b_decoder_input_pred_beam, r2b_history_score, r2b_stop = get_res_beam(reverse_trietree, r2b_logits, token2idx, r2b_decoder_input_pred_beam, r2b_history_score, num_beam)

    l2f_decoder_input_all = l2f_decoder_input_pred_beam
    r2b_decoder_input_all = r2b_decoder_input_pred_beam
    if np.shape(l2f_decoder_input_pred_beam)[1] < (seq_length+2):
      padding = np.zeros([num_beam, seq_length+2 - np.shape(l2f_decoder_input_pred_beam)[1]])
      l2f_decoder_input_all = np.concatenate((l2f_decoder_input_pred_beam, padding), axis=-1)
    if np.shape(r2b_decoder_input_pred_beam)[1] < (seq_length+2):
      padding = np.zeros([num_beam, seq_length+2 - np.shape(r2b_decoder_input_pred_beam)[1]])
      r2b_decoder_input_all = np.concatenate((r2b_decoder_input_pred_beam, padding), axis=-1)
      
    l2f_decoder_input_pred_final.append(l2f_decoder_input_all)
    r2b_decoder_input_pred_final.append(r2b_decoder_input_all)
    l2f_history_score_final = l2f_history_score_final + l2f_history_score
    r2b_history_score_final = r2b_history_score_final + r2b_history_score
  
  l2f_decoder_input_pred_final = np.concatenate(l2f_decoder_input_pred_final,axis=0)
  r2b_decoder_input_pred_final = np.concatenate(r2b_decoder_input_pred_final,axis=0)
  l2f_res, l2f_correctness = test_beam_search_result(decoder_groundtruth, num_beam, l2f_decoder_input_pred_final, l2f_pos)
  r2b_res, r2b_correctness = test_beam_search_result(decoder_groundtruth, num_beam, r2b_decoder_input_pred_final, r2b_pos, is_reverse=True)

  res, res_score, correctness, combine_pred, combine_score = test_beam_search_result_combine_v3(decoder_groundtruth, 
                                                                                             num_beam, 
                                                                                             decoder_input_pred, 
                                                                                             back_decoder_input_pred, 
                                                                                             history_score, 
                                                                                             back_history_score, 
                                                                                             l2f_decoder_input_pred_final,
                                                                                             r2b_decoder_input_pred_final,
                                                                                             l2f_history_score_final,
                                                                                             r2b_history_score_final,
                                                                                             token2idx,
                                                                                             pos)

#   print("combine_pred:",len(combine_pred))
  for i in range(len(decoder_groundtruth)):
    handler.write("Real " + " :" + " ".join([str(item) for item in real_label_score[i]]) + "\n")
    for j in range(i * num_beam, i * num_beam + num_beam):
      handler.write("Pred " + str(j) + " :" + combine_pred[j] + "\n")
      handler.write("score " + str(j) + " :" + str(combine_score[j]) + "\n")
    handler.write("\n")
  handler.flush()
  hypotheses.extend(res)

  hypotheses = postprocess_with_score(hypotheses, idx2token, res_score)
  
  return hypotheses, pos, correctness, forward_pos, backward_pos, l2f_pos, r2b_pos


def test_beam_search_result(decoder_groundtruth, num_beam, decoder_input_pred, pos, is_reverse=False):
  res = list()
  correctness = [0] * len(decoder_groundtruth)
  for i in range(len(decoder_groundtruth)):
    rel = [int(l) for l in decoder_groundtruth[i] if l >= 3]
    if is_reverse:
      rel.reverse()
    flag = False
    for j in range(i * num_beam, i * num_beam + num_beam):
      pred_a = [int(item) for item in decoder_input_pred[j]]
      pred = [m for m in pred_a if m >= 3]
      if j % num_beam == 0:
        res.append(pred)
      if rel == pred:
        flag = True
        rank_idx = j % num_beam
        if rank_idx == 0:
          correctness[i] = 1
        for index in range(len(pos)):
          if index >= rank_idx:
            pos[index] = pos[index] + 1
        break
  return res, correctness


def test_beam_search_result_mid(decoder_groundtruth, num_beam, m2l_decoder_input_pred, m2r_decoder_input_pred, m2l_history_scores, m2r_history_scores, pos):
  
  middle_history_scores = [[None] * len(m2l_history_scores) for _ in range(len(m2l_history_scores))]
  for i in range(len(m2l_history_scores)):
    for j in range(len(m2l_history_scores)):
      middle_history_scores[i][j] = m2l_history_scores[i] + m2r_history_scores[j][1:]
      
  res = list()
  correctness = [0] * len(decoder_groundtruth)
  middle_str2score = []
  for i in range(len(decoder_groundtruth)):
    rel = [int(l) for l in decoder_groundtruth[i] if l >= 3]
    flag = False
    str2score = {}
#     first_print = True
    for j in range(i * num_beam, i * num_beam + num_beam):
      for k in range(i * num_beam, i * num_beam + num_beam):
        m2l_pred = [int(item) for item in m2l_decoder_input_pred[j]]
        m2r_pred = [int(item) for item in m2r_decoder_input_pred[k]]
#         if first_print:
#           print(m2l_pred,m2r_pred)
#           first_print = False
        if m2l_pred[0] == m2r_pred[0]:
          m2l_pred.reverse()
          pred_a = m2l_pred+m2r_pred[1:] # remember middle word appears twice, remove one
          pred = [str(m) for m in pred_a if m >= 3]
          pred_str = " ".join(pred)
          score = cal_per_seq_score(middle_history_scores[j][k])
          if pred_str not in str2score:
            str2score[pred_str] = score
          else:
            str2score[pred_str] = max(str2score[pred_str], score)
    str2score = sorted(str2score.items(), key = lambda kv: kv[1], reverse=True)[:num_beam]
    while len(str2score) < num_beam:
      str2score.append((("1 0 0 0 0 0 0 0 0 2"),0))
#       print("less than num_beam: ",str2score)
    middle_str2score.append(str2score)
    
    for idx, (pred_str, score) in enumerate(str2score):
      pred = [int(item) for item in pred_str.split(" ")]
      if idx == 0:
        res.append(pred)
      if rel == pred:
        flag = True
        rank_idx = idx % num_beam
        if rank_idx == 0:
          correctness[i] = 1
        for index in range(len(pos)):
          if index >= rank_idx:
            pos[index] = pos[index] + 1
        break
  return res, correctness, middle_str2score

# def test_beam_search_result_mid_v2(decoder_groundtruth, num_beam, decode_input_m_all, logits_m_forward_score, decode_input_m_all_back, logits_m_backward_score, token2idx, pos):
      
#   res = list()
#   correctness = [0] * len(decoder_groundtruth)
#   for i in range(len(decoder_groundtruth)):
#     recall_list = dict()
#     rel = [int(l) for l in decoder_groundtruth[i] if l >= 3]
#     for j in range(i * num_beam, i * num_beam + num_beam):
#       pred_a = [int(item) for item in decode_input_m_all[j]]
#       pred_a_move = pred_a[1:]+[token2idx['</s>']]
#       pred_str = " ".join([str(m) for m in pred_a if m >= 3])
#       scores_m = []
#       for k in range(len(pred_a_move)):
#         scores_m.append(logits_m_forward_score[j][k][pred_a_move[k]])
#       score = cal_per_seq_score(scores_m)
# #       print("middle ff: ",score)
#       if len(pred_str) == 0:
#         continue
#       if pred_str in recall_list:
#         recall_list[pred_str] += score
#       else:
#         recall_list[pred_str] = score
        
#     for j in range(i * num_beam, i * num_beam + num_beam):
#       pred_a = [int(item) for item in decode_input_m_all_back[j]]
#       pred_a_move = pred_a[1:]+[token2idx['</s>']]
#       pred_str = " ".join([str(m) for m in pred_a[::-1] if m >= 3])
#       scores_m = []
#       for k in range(len(pred_a_move)):
#         scores_m.append(logits_m_forward_score[j][k][pred_a_move[k]])
#       score = cal_per_seq_score(scores_m)
# #       print("middle bb: ",score)
#       if len(pred_str) == 0:
#         continue
#       if pred_str in recall_list:
#         recall_list[pred_str] += score
#       else:
#         recall_list[pred_str] = score 
    
#     ranking = sorted(recall_list.items(), key=lambda item: item[1], reverse=True)[:num_beam]
#     for j, rank in enumerate(ranking):
#       pred = [int(item) for item in rank[0].split(" ")]
#       if j % num_beam == 0:
#         res.append(pred)
#       if rel == pred:
#         flag = True
#         rank_idx = j % num_beam
#         if rank_idx == 0:
#           correctness[i] = 1
#         for index in range(len(pos)):
#           if index >= rank_idx:
#             pos[index] = pos[index] + 1
#         break
#   return res, correctness


def test_beam_search_result_combine_v3(decoder_groundtruth, 
                                    num_beam, 
                                    decoder_input_pred, 
                                    back_decoder_input_pred, 
                                    history_scores, 
                                    back_history_scores, 
                                    l2f_decoder_input_pred_final,
                                    r2b_decoder_input_pred_final,
                                    l2f_history_score,
                                    r2b_history_score,
                                    token2idx,
                                    pos):
  res = list()
  res_score = list()
  beam_pred = list()
  beam_score = list()
  correctness = [0] * len(decoder_groundtruth)
  weight_f = 1.0
  weight_b = 1.0
  weight_m = 0.5

  for i in range(len(decoder_groundtruth)):
    rel = [int(l) for l in decoder_groundtruth[i] if l >= 3]
    flag = False
    recall_list = dict()
    backward_pred = []
    forward_pred = []
    l2f_pred = []
    r2b_pred = []
    b_recall_list = {}
    f_recall_list = {}
    l2f_recall_list = {}
    r2b_recall_list = {}
    for j in range(i * num_beam, i * num_beam + num_beam):
      pred_a = [int(item) for item in back_decoder_input_pred[j]]
      pred = [str(m) for m in pred_a if m >= 3]
      pred.reverse()
      pred_str = " ".join(pred)
      backward_pred.append(pred_str)
      score = weight_b*cal_per_seq_score(back_history_scores[j])
      if pred_str not in b_recall_list:
        b_recall_list[pred_str] = score
      else:
        b_recall_list[pred_str] = max(score, b_recall_list[pred_str])
        

    for j in range(i * num_beam, i * num_beam + num_beam):
      pred_a = [int(item) for item in decoder_input_pred[j]]
      pred = [str(m) for m in pred_a if m >= 3]
      pred_str = " ".join(pred)
      forward_pred.append(pred_str)
      score = weight_f*cal_per_seq_score(history_scores[j])
      if pred_str not in f_recall_list:
        f_recall_list[pred_str] = score
      else:
        f_recall_list[pred_str] = max(score, f_recall_list[pred_str])
        
    for j in range(i * num_beam, i * num_beam + num_beam):
      pred_a = [int(item) for item in l2f_decoder_input_pred_final[j]]
      pred = [str(m) for m in pred_a if m >= 3]
      pred_str = " ".join(pred)
      l2f_pred.append(pred_str)
      score = weight_m*cal_per_seq_score(l2f_history_score[j])
      if pred_str not in l2f_recall_list:
        l2f_recall_list[pred_str] = score
      else:
        l2f_recall_list[pred_str] = max(score, l2f_recall_list[pred_str])
  
    for j in range(i * num_beam, i * num_beam + num_beam):
      pred_a = [int(item) for item in r2b_decoder_input_pred_final[j]]
      pred = [str(m) for m in pred_a if m >= 3]
      pred.reverse()
      pred_str = " ".join(pred)
      r2b_pred.append(pred_str)
      score = weight_m*cal_per_seq_score(r2b_history_score[j])
      if pred_str not in r2b_recall_list:
        r2b_recall_list[pred_str] = score
      else:
        r2b_recall_list[pred_str] = max(score, r2b_recall_list[pred_str])
      
    for pred_str, score in b_recall_list.items():
      if pred_str in recall_list:
        recall_list[pred_str] += score
      else:
        recall_list[pred_str] = score 
        
    for pred_str, score in f_recall_list.items():
      if pred_str in recall_list:
        recall_list[pred_str] += score
      else:
        recall_list[pred_str] = score 
        
    for pred_str, score in l2f_recall_list.items():
      if pred_str in recall_list:
        recall_list[pred_str] += score
      else:
        recall_list[pred_str] = score 
        
    for pred_str, score in r2b_recall_list.items():
      if pred_str in recall_list:
        recall_list[pred_str] += score
      else:
        recall_list[pred_str] = score 

    ranking = sorted(recall_list.items(), key=lambda item: item[1], reverse=True)[:num_beam]
    for j, rank in enumerate(ranking):
      pred = [int(item) for item in rank[0].split(" ")]
      if j % num_beam == 0:
        res.append(pred)
        res_score.append(rank[1])
      beam_pred.append(rank[0])
      beam_score.append(rank[1])
      if rel == pred:
        flag = True
        rank_idx = j % num_beam
        if rank_idx == 0:
          correctness[i] = 1
        for index in range(len(pos)):
          if index >= rank_idx:
            pos[index] = pos[index] + 1
#         break
  return res, res_score, correctness, beam_pred, beam_score


def convert_to_history(decoder_inputs, idx2token):
  history = list()
  for i in range(len(decoder_inputs)):
    history.append([index for index in decoder_inputs[i]])
  return history


def softmax(eles):
  eles_e = [math.exp(ele) for ele in eles]
  sum = np.sum(eles_e)
  return [ele / sum for ele in eles_e]


def cal_per_seq_score(score_list_per_seq):
  score = 1.0
  num_length = 0
  for prob in score_list_per_seq:
    score *= prob
    num_length += 1
  score = math.pow(score, 1.0 / num_length)
  return score


def get_cur_sample_topn_res_per_beam(cur_case_index, beam_num, logits, trietree, decoder_inputs, token2idx,
                                     history_score, stop):
  seq_per_sample = []
  seq_score_per_sample = []
  for i in range(cur_case_index, cur_case_index + beam_num):
    # shape [voc]
    score = logits[i, -1, ::]
    score = np.array(softmax(score))
    # print(" ".join([str(item) for item in score]))
    # input("next")
    candidates = trietree.get_next_candis(decoder_inputs[i])
#     print("decoder_inputs: ", decoder_inputs[i])
#     print("candidates length: ", len(candidates))
    if len(candidates) == 0:
      seq_temp = list(decoder_inputs[i])
      seq_temp.append(0)
      seq_per_sample.append(list(seq_temp))
      seq_score_temp = list(history_score[i])
      seq_score_per_sample.append(list(seq_score_temp))
    else:
      indexes = [index_cand for index_cand in candidates]
      # if token2idx["</s>"] not in indexes:
      #   indexes.append(token2idx["</s>"])
      indexes = np.asarray(indexes)
      ##rank top beam case
      # topn_indexes=score[indexes].argsort()[-beam_num:][::-1]
      # rank whole index
      topn_indexes = score[indexes].argsort()[::-1]
      for index_temp in topn_indexes:
        seq_temp = list(decoder_inputs[i])
        seq_temp.append(indexes[index_temp])
        seq_per_sample.append(list(seq_temp))

        seq_score_temp = list(history_score[i])
        seq_score_temp.append(score[indexes[index_temp]])
        seq_score_per_sample.append(list(seq_score_temp))
      stop = False
  return seq_per_sample, seq_score_per_sample, stop


def select_best_beam_per_sample(seq_per_sample, seq_score_per_sample, beam_num):
  wordstr2score = {}
  wordstr2index = {}
  scoredebuglist = []
  try:
    for i in range(len(seq_per_sample)):
      wordstr = " ".join([str(item_temp) for item_temp in seq_per_sample[i]])
      wordstr2index[wordstr] = i
      score = cal_per_seq_score(seq_score_per_sample[i])
      if wordstr in wordstr2score:
#         print("wordstr in wordstr2score")
        wordstr2score[wordstr] = max(wordstr2score[wordstr], score)
      else:
        wordstr2score[wordstr] = score
      scoredebuglist.append(score)
  except ValueError:
    print("!!!!!!!!!!!error!!!!!!!")
    input("debug")

  ranking = sorted(wordstr2score.items(), key=lambda item: item[1], reverse=True)
  result_index_list = []
  for result_rank in ranking:
    wordstr = result_rank[0]
    index_temp = wordstr2index[wordstr]
    result_index_list.append(index_temp)
    if (len(result_index_list) == beam_num):
      break

  while (len(result_index_list) < beam_num):
#     print("less than beam num!!!!!")
    result_index_list.append(result_index_list[-1])
  return result_index_list


def get_res_beam(trietree, logits, token2idx, decoder_inputs, history_score, beam_num):
  # res = list()
  stop = True

  # logits [B*num_beam,max_length,voc]
  # history [B*num_bean,len]
  # history_score [B*num_bean,len]

  decoder_inputs_results = []
  history_score_result = []
  # print("============")
  for cur_case_index in range(0, len(logits), beam_num):

    # input: beam num case for 1 test case
    # output: beam*beam search result
    seq_per_sample, seq_score_per_sample, stop = get_cur_sample_topn_res_per_beam(cur_case_index, beam_num, logits,
                                                                                  trietree, decoder_inputs, token2idx,
                                                                                  history_score, stop)
    # ranking history_temp
    assert len(seq_per_sample) == len(seq_score_per_sample)

    # select the top beam seq from the beam*beam seq
    result_index_list = select_best_beam_per_sample(seq_per_sample, seq_score_per_sample, beam_num)

    for index_temp in range(beam_num):
      decoder_inputs_results.append(list(seq_per_sample[result_index_list[index_temp]]))
      history_score_result.append(list(seq_score_per_sample[result_index_list[index_temp]]))
  # decoder_inputs = np.concatenate((decoder_inputs, np.asarray(np.expand_dims(res, -1))), -1)
  decoder_inputs_results = np.array(decoder_inputs_results)
  return decoder_inputs_results, history_score_result, stop



def load_bin_data_test_phone2pinyin(bin_dir, maxlen2):
  phone_input_path = os.path.join(bin_dir, "test_90_sil_256_res.x.bin")
  decoder_input_path = os.path.join(bin_dir, "test_90_sil_256_res.y1.txt")
  decoder_output_path = os.path.join(bin_dir, "test_90_sil_256_res.y2.txt")

  ret_val_x = load_bytes_to_ram(phone_input_path, file_type="x")
  y1_bytes = open(decoder_input_path, "r")
  y2_bytes = open(decoder_output_path, "r")
  ret_list_y1 = get_one_instance_y_asr20(y1_bytes, maxlen2)
  ret_list_y2 = get_one_instance_y_asr20(y2_bytes, maxlen2)

  test_data = list()
  index = list()
  for i in range(len(ret_val_x)):
    phone_online_index1 = ret_list_y1[i]
    phone_online_index2 = ret_list_y2[i]
    phone_seq_index, val = get_one_instance_x_256(ret_val_x[i])
    test_data.append((phone_seq_index, phone_online_index1, phone_online_index2))
    index.append(0)

  return test_data, index


def get_lines(file):
  '''
  get all lines
  :param file:
  :return:
  '''
  file_handler = open(file)
  lines = [line.strip() for line in file_handler.readlines()]
  return lines



def padded(inputs, leng):
  res = list(np.zeros(leng, dtype=int))
  for i in range(len(inputs)):
    res[i] = inputs[i]
  return res


def get_reversed_input(x, y):
  input1 = [i for i in x if i > 2]
  input1.reverse()
  new_input = padded([1] + input1, len(x))
  new_output = padded(input1 + [2], len(x))
  return new_input, new_output
