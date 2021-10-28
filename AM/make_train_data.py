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
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
from utils import get_data, data_hparams
from cnn_ctc import Am, am_hparams

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string(name="data_dir", default="../train_data_phoneme/", help="save train data to this dir")
tf.flags.DEFINE_string("vndc_train_file", "VNDC/vndc_train.txt", help="train file")
tf.flags.DEFINE_string("am_model_name", "AM_model_0915.h5", help="model file")
tf.flags.DEFINE_integer("data_length", 10000, help="the number of training samples, start from a samll number")

random.seed(1)
data_queue = queue.Queue(300)
tf.logging.set_verbosity(tf.logging.INFO)
    
class ExampleMid:

  def __init__(self, x, mid_words, decode_inputs, y, m2l_decode_inputs, m2l_y, m2r_decode_inputs, m2r_y, valid_pos):
    self.x = list()
    for ele in x:
      self.x += ele.tolist()

    self.mid_words = mid_words
    self.decode_inputs = decode_inputs.tolist()
    self.y = y.tolist()
    self.m2l_decode_inputs = m2l_decode_inputs.tolist()
    self.m2l_y = m2l_y.tolist()
    self.m2r_decode_inputs = m2r_decode_inputs.tolist()
    self.m2r_y = m2r_y.tolist()
    self.valid_pos = valid_pos.tolist()

PAD_ID = 0
BEG_ID = 1
END_ID = 2

t_val_map = {90: 0.9, 95: 0.95, 99: 0.99}

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
  


def create_tfrecord(batch, index, data_dir):
  '''
  :param batch: batched
  :param index: the tf-record index
  :param data_dir: the dir for storing tf-record
  :return: tf-record file
  '''
  tf_record_file = os.path.join(data_dir, str(int(index)) + '.record')
  tf.logging.info('file %s generation stared ' % (tf_record_file))
  writer = tf.python_io.TFRecordWriter(tf_record_file)
  for (ex_index, example) in enumerate(batch):
    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=values))
      return f

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
      return f

    features = collections.OrderedDict()
    features["x"] = create_float_feature(example.x)
    features['valid_pos'] = create_int_feature(example.valid_pos)
    features['decode_inputs'] = create_int_feature(example.decode_inputs)
    features['y'] = create_int_feature(example.y)
    features["m2l_decode_inputs"] = create_int_feature(list(example.m2l_decode_inputs))
    features['m2l_y'] = create_int_feature(list(example.m2l_y))
    features["m2r_decode_inputs"] = create_int_feature(list(example.m2r_decode_inputs))
    features['m2r_y'] = create_int_feature(list(example.m2r_y))
    features['mid_words'] = create_int_feature(list(example.mid_words))

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())

  writer.close()
  tf.logging.info('file %s generation ended' % (tf_record_file))
  return tf_record_file

def data_pipeline_thread(thread_name,data_dir,train_data,data_args,am_vocab_new):
    tf.logging.info('thread %s started' % (thread_name))

    # delete previous_dc tf-record
    if os.path.exists(data_dir):
        command = "rm -rf " + os.path.join(data_dir, "*.record")
        os.system(command)
    else:
        command = "mkdir " + data_dir
        os.system(command)
    
    am_args = am_hparams()
    am_args.vocab_size = len(train_data.am_vocab)
    am = Am(am_args)
    print('loading acoustic model...')
    am.ctc_model.load_weights('models/'+FLAGS.am_model_name)
    am_batch = train_data.get_am_batch_p2s()
    tfrecord_index = 0
    batch = list()
    for i in tqdm(range(len(train_data.pny_lst) // data_args.batch_size)):
        # for i in range(len(train_data.pny_lst) // data_args.batch_size):
        begin = i * data_args.batch_size
        end = begin + data_args.batch_size
        # y_list = []
        inputs, _ = next(am_batch)
        x = inputs['the_inputs']
        x_res_lens = inputs['input_length']
        y_place_list = inputs['place_list']
        results = am.model.predict(x, steps=1)
    
        for result, x_res_len, y_place in zip(results, x_res_lens, y_place_list):
            am_value = result[:x_res_len, :]
            phone_tensor, phone_valid_position = encode_am_feature_256_mid(am_value, am_vocab_new, max_len1=40)
            mid_words, m2l_decode_inputs, m2l_y, m2r_decode_inputs, m2r_y = encode_y_pinyin_mids(y_place, train_data.pinyin_y_dict2id, phone_tensor, am_vocab_new, maxlen2=8)
            decoder_input, decoder_ground_truth, _, _ = encode_y_pinyin(y_place, train_data.pinyin_y_dict2id, maxlen2=10)
            batch.append(ExampleMid(x=phone_tensor, mid_words=mid_words, 
                                    decode_inputs=decoder_input, y=decoder_ground_truth,
                                    m2l_decode_inputs=m2l_decode_inputs, m2l_y=m2l_y,
                                    m2r_decode_inputs=m2r_decode_inputs, m2r_y=m2r_y,
                                    valid_pos=phone_valid_position))

            batch_size = len(batch)
            if batch_size > 0 and batch_size % 10000 == 0:
                data_file = create_tfrecord(batch=batch, index=tfrecord_index, data_dir=data_dir)
                data_queue.put(data_file)
                batch.clear()
                tfrecord_index += 1
        # 写入末段
    if len(batch) % 10000:
        data_file = create_tfrecord(batch=batch, index=tfrecord_index, data_dir=data_dir)
        data_queue.put(data_file)
    data_queue.put(None)
    
    
def main():

    data_args = data_hparams()
    data_args.data_type = 'train'
    data_args.data_path = 'data/'
    data_args.thchs30 = False
    data_args.aishell = False
    data_args.prime = False
    data_args.stcmd = False
    data_args.vndc = True
    data_args.batch_size = 32
    data_args.data_length = FLAGS.data_length
    data_args.shuffle = True
    data_args.vndc_train_file=FLAGS.vndc_train_file
    train_data = get_data(data_args)
    if data_args.data_length >= 800000:
        data_args.data_length = None

    print("=======am vocab length:" + str(len(train_data.am_vocab)))

    am_vocab_new=['<pad>','<s>','</s>']
    for item in train_data.am_vocab:
        am_vocab_new.append(item)
    
    #start the data pipeline thread
    data_pipe_th = threading.Thread(
        target=data_pipeline_thread,
        args=('data_pipeline_thread',FLAGS.data_dir, train_data,data_args,am_vocab_new))
    data_pipe_th.start()
    
    prefix = FLAGS.data_dir.replace('..','.')
    handler = open("../tf_list.txt", "w")
    for i in range(math.ceil(data_args.data_length/10000)):
        handler.write(os.path.join(prefix, str(i) + ".record") + "\n")
    
    

if __name__=="__main__":
    main()
        
        

    
