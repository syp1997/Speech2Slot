# coding=utf-8
import os
from pypinyin import pinyin, lazy_pinyin, Style
import pypinyin
import threading
import queue
import random

import difflib
import tensorflow as tf
import numpy as np
from utils import decode_ctc, GetEditDistance
from tqdm import tqdm
import collections
import pickle

from utils import get_data, data_hparams
from cnn_ctc import Am, am_hparams

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string(name="data_dir", default="../test_data_phoneme/", help="save test data to this dir")
tf.flags.DEFINE_string(name="output_file", default="test_phoneme.pkl", help="output file name")
tf.flags.DEFINE_string("vndc_test_file", "VNDC/vndc_test.txt", help="train file")
tf.flags.DEFINE_string("am_model_name", "AM_model_0915.h5", help="model file")

PAD_ID = 0
BEG_ID = 1
END_ID = 2

t_val_map = {90: 0.9, 95: 0.95, 99: 0.99}

if not os.path.exists(FLAGS.data_dir):
    command = "mkdir " + FLAGS.data_dir
    os.system(command)
        
def parse_phone_small(weights):
    return np.argmax(weights)

def data_display_small(phone_input, voc_special):
    phonelist = []
    for item in phone_input:
        phoneindex = parse_phone_small(item)
        phone = voc_special[phoneindex]
        phonelist.append(phone)
    return phonelist


def make_special_token_softmax_embedding(token, softmax_embedding_size):
    token_emb = np.zeros(softmax_embedding_size)
    if token == '<s>':
        token_emb[BEG_ID] = 1.0
    elif token == '</s>':
        token_emb[END_ID] = 1.0
    elif token == '<pad>':
        token_emb[PAD_ID] = 0.0
    return token_emb.reshape((1, -1))


def encode_am_feature_256_mid(am_feature, am_vocab_new, phone_vocab_size=1293, sil_threshold=90, max_len1=40):
    valid_pos = np.zeros(max_len1, dtype=int)
    
    t_val = t_val_map[sil_threshold] # 0.9
    am_concat1 = np.concatenate((np.zeros((am_feature.shape[0], 3)), am_feature), axis=1)
    end_token = make_special_token_softmax_embedding('</s>', phone_vocab_size + 3)
    
    # 输入结尾加了end token
    am_concat2 = np.concatenate((am_concat1, end_token), axis=0)
    
    idx = []
    tokens = [am_vocab_new[int(index)] for index in np.argmax(am_concat2, axis=1)]
    
    tokens_filter = []
    for k in range(len(tokens)):
        token = tokens[k]
        max_val = np.max(am_concat2[k])
        if (token != "_" or max_val < t_val) and np.sum(am_concat2[k]) > 0.5:
            idx.append(k)
            tokens_filter.append(token)
    valid_pos[0] = 0
    for index in range(len(tokens_filter)):
        token = tokens_filter[index]
        if (token != '_') and (token != '<s>') and (token != "</s>") and (token != "<pad>"):
            valid_pos[index+1] = index+1
    
    am_concat2_ = am_concat2[idx]
    
    start = np.zeros((1, am_concat2_.shape[1]))
    start[0][1] = 1
    
    # 输入开头加了start token
    am_concat3 = np.concatenate((start, am_concat2_), axis=0)
    
    if len(am_concat3) > max_len1:
        return None
    # [l,D]
    padding_matrix = np.zeros((max_len1 - len(am_concat3), phone_vocab_size + 3), dtype=float)
    am_matrix_final = np.concatenate((am_concat3, padding_matrix), axis=0)
    
    return am_matrix_final, valid_pos
  
  
def encode_y_pinyin(slot_value, pinyin_dict, maxlen2):
    pinyin_res = pinyin(slot_value, style=pypinyin.TONE3, heteronym=True)
    pinyin_st_pre = []
    for item in pinyin_res:
        pinyin_st_pre.append(item[0])
    # print(" ".join(pinyin_st_pre))
    y = [pinyin_dict[ele] for ele in pinyin_st_pre]
    y = [pinyin_dict["<s>"]] + y + [pinyin_dict["</s>"]]
    
    if len(y) + 1 > maxlen2:
        y = y[:maxlen2 + 1]
        # return None, None
    
    y1 = y[:-1]
    y2 = y[1:]
    
    ret1 = np.zeros(maxlen2, dtype=int)
    ret2 = np.zeros(maxlen2, dtype=int)
    for i, x in enumerate(y1):
        ret1[i] = x
    for i, x in enumerate(y2):
        ret2[i] = x
    
    return ret1, ret2," ".join([str(item) for item in y1])," ".join([str(item) for item in y2])

  
def encode_y_pinyin_mids(slot_value, pinyin_dict, phone_tensor, am_vocab_new, maxlen2):
    pinyin_res = pinyin(slot_value, style=pypinyin.TONE3, heteronym=True)
    pinyin_st_pre = []
    for item in pinyin_res:
        pinyin_st_pre.append(item[0])
#     print(pinyin_st_pre)
    tokens2prob = {}
    for tensor in phone_tensor:
      max_idx = int(np.argmax(tensor))
      token = am_vocab_new[max_idx]
      if token not in tokens2prob:
        tokens2prob[token] = tensor[max_idx]
      else:
        tokens2prob[token] = max(tensor[max_idx],tokens2prob[token])
        
    mid_word = None
    num = 1
    mid_idx = len(pinyin_st_pre)//2
    for i in range(len(pinyin_st_pre)):
      if i % 2 == 0:
        cur_idx = mid_idx + num//2
      elif i % 2 == 1:
        cur_idx = mid_idx - num//2
      if pinyin_st_pre[cur_idx] in tokens2prob:
        mid_word = pinyin_st_pre[cur_idx]
        break
      num += 1
    if mid_word != None:
#       print("found: {}: {}".format(mid_word, tokens2prob[mid_word]))
      pass
    elif mid_word == None:
      cur_idx = len(pinyin_st_pre)//2
      mid_word = pinyin_st_pre[cur_idx]
#       print("not found: {}: {}".format(mid_word, "0000"))
      
    mid_words = []
    for i in range(len(pinyin_st_pre)):
      if pinyin_st_pre[i] in tokens2prob:
        mid_words.append(pinyin_st_pre[i])
    if mid_words == []:
      mid_words.append(mid_word)
#     print(mid_words)
      
    m2l = pinyin_st_pre[:cur_idx+1][::-1]
    m2r = pinyin_st_pre[cur_idx:]
#     print(m2l,m2r)
    m2l_y = [pinyin_dict[ele] for ele in m2l] + [pinyin_dict["</s>"]]
    m2r_y = [pinyin_dict[ele] for ele in m2r] + [pinyin_dict["</s>"]]
    
#     y = [pinyin_dict[ele] for ele in pinyin_st_pre]
#     y = [pinyin_dict["<s>"]] + y + [pinyin_dict["</s>"]]
    
    m2l_y = m2l_y[:maxlen2 + 1]
    m2r_y = m2r_y[:maxlen2 + 1]
    
    m2l_y1 = m2l_y[:-1]
    m2l_y2 = m2l_y[1:]
    
    m2r_y1 = m2r_y[:-1]
    m2r_y2 = m2r_y[1:]
    
    m2l_ret1 = np.zeros(maxlen2, dtype=int)
    m2l_ret2 = np.zeros(maxlen2, dtype=int)
    for i, x in enumerate(m2l_y1):
        m2l_ret1[i] = x
    for i, x in enumerate(m2l_y2):
        m2l_ret2[i] = x
        
    m2r_ret1 = np.zeros(maxlen2, dtype=int)
    m2r_ret2 = np.zeros(maxlen2, dtype=int)
    for i, x in enumerate(m2r_y1):
        m2r_ret1[i] = x
    for i, x in enumerate(m2r_y2):
        m2r_ret2[i] = x
        
    vocab_size = len(am_vocab_new)
    mid_words_ret = np.zeros(vocab_size, dtype=int)
    mid_words_idx = np.array([pinyin_dict[i] for i in mid_words], dtype=int)
    mid_words_ret[mid_words_idx] = 1
#     print(mid_words_ret.shape)
    
    return mid_words_ret, m2l_ret1, m2l_ret2, m2r_ret1, m2r_ret2
  
  
  
def main():    
  
    data_args = data_hparams()
    data_args.data_type = 'test'
    data_args.data_path = 'data/'
    data_args.thchs30 = False
    data_args.aishell = False
    data_args.prime = False
    data_args.stcmd = False
    data_args.vndc = True
    data_args.batch_size = 32
    data_args.data_length = None
    data_args.shuffle = True
    data_args.vndc_test_file=FLAGS.vndc_test_file
    test_data = get_data(data_args)
    
    print("=======am vocab length:" + str(len(test_data.am_vocab)))
    
    am_vocab_new = ['<pad>', '<s>', '</s>']
    for item in test_data.am_vocab:
        am_vocab_new.append(item)
    
    am_args = am_hparams()
    am_args.vocab_size = len(test_data.am_vocab)
    am = Am(am_args)
    print('loading acoustic model: ' + FLAGS.am_model_name)
    am.ctc_model.load_weights('models/' + FLAGS.am_model_name)
    am_batch = test_data.get_am_batch_p2s()
    test_res_data = list()
    
    index=0
    for i in tqdm(range(len(test_data.pny_lst) // data_args.batch_size)):

        inputs, _ = next(am_batch)
        x = inputs['the_inputs']
        x_res_lens = inputs['input_length']
        y_place_list = inputs['place_list']
        results = am.model.predict(x, steps=1)
    
        for result, x_res_len, y_place in zip(results, x_res_lens, y_place_list):
            am_value = result[:x_res_len, :]
            #[max_l1,voc_source]
            phone_tensor, phone_valid_position = encode_am_feature_256_mid(am_value, am_vocab_new, max_len1=40)
            mid_words, m2l_decode_inputs, m2l_y, m2r_decode_inputs, m2r_y = encode_y_pinyin_mids(y_place, test_data.pinyin_y_dict2id, phone_tensor, am_vocab_new, maxlen2=8)
            decoder_input, decoder_ground_truth, _, _ = encode_y_pinyin(y_place, test_data.pinyin_y_dict2id, maxlen2=10)
            test_res_data.append((phone_tensor,
                                  decoder_input,
                                  decoder_ground_truth,
                                  phone_valid_position,
                                  m2l_decode_inputs,
                                  m2l_y,
                                  m2r_decode_inputs,
                                  m2r_y,
                                  mid_words))
            index+=1
    print("num:"+str(len(test_res_data)))
    fw=open(os.path.join(FLAGS.data_dir,FLAGS.output_file),"wb")
    pickle.dump(test_res_data,fw)
    fw.close()

if __name__ == "__main__":
    main()




