#coding=utf-8
import os
import difflib
import tensorflow as tf
import numpy as np
from utils import decode_ctc, GetEditDistance, get_data, data_hparams
from tqdm import tqdm
from cnn_ctc import Am, am_hparams
from utils import get_data, data_hparams


flags = tf.flags
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_string("vndc_testing_file", "VNDC/vndc_test_human.txt", help="test file")
flags.DEFINE_string("am_model_name", "model_AM_0915.h5", help='model file')

data_args = data_hparams()
data_args.data_type = 'test'
data_args.data_path = 'data/'
data_args.thchs30 = False
data_args.aishell = False
data_args.prime = False
data_args.stcmd = False
data_args.vndc = True
data_args.shuffle = False
data_args.batch_size = 1
data_args.data_length=None
data_args.vndc_test_file=FLAGS.vndc_testing_file
test_data = get_data(data_args)

am_args = am_hparams()
am_args.vocab_size = len(test_data.am_vocab)
am = Am(am_args)
print('loading acoustic model...')
am.ctc_model.load_weights('models/'+FLAGS.am_model_name)

am_vocab_new = ['<pad>', '<s>', '</s>']
for item in test_data.am_vocab:
    am_vocab_new.append(item)

am_vocab_new = ['<pad>', '<s>', '</s>']
for item in test_data.am_vocab:
    am_vocab_new.append(item)

am_batch = test_data.get_am_batch()
pny_num=0
pny_error_num=0

for i in tqdm(range(len(test_data.pny_lst)//data_args.batch_size)):
    begin = i * data_args.batch_size
    end = begin + data_args.batch_size
    y_list=[]
    label_list=[]
    inputs, _ = next(am_batch)
    x = inputs['the_inputs']
    x_res_len=inputs['input_length']
    y_list.extend(test_data.pny_lst[begin:end])
    label_list.extend(test_data.han_lst[begin:end])
    result = am.model.predict(x, steps=1)
    _, texts = decode_ctc(result, test_data.am_vocab,x_res_len)
    x_list=[]
    pred_text_list=[]
    for index in range(len(y_list)):
        y=y_list[index]
        pny_error_num += min(len(y), GetEditDistance(y, texts[index]))
        pny_num += len(y)
        
print('phone error rate：', pny_error_num / pny_num)
