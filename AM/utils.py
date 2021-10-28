import os
import difflib
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.fftpack import fft
from python_speech_features import mfcc
from random import shuffle
from keras import backend as K
import random
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def data_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_type='train',
        data_path='data/',
        thchs30=False,
        aishell=False,
        prime=False,
        stcmd=False,
        vndc=False,
        batch_size=16,
        start=None,
        data_length=10,
        shuffle=True,
        vndc_test_file=None,
        vndc_train_file=None)
    return params


class get_data():
    def __init__(self, args):
        self.data_type = args.data_type
        self.data_path = args.data_path
        self.thchs30 = args.thchs30
        self.aishell = args.aishell
        self.prime = args.prime
        self.stcmd = args.stcmd
        self.vndc = args.vndc
        self.start = args.start
        self.data_length = args.data_length
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.vndc_test_file = args.vndc_test_file
        self.vndc_train_file = args.vndc_train_file
        self.source_init()

    def source_init(self):
        print('get source list...')
        read_files = []

        if self.data_type == 'train':
            if self.thchs30 == True:
                read_files.append('thchs_train.txt')
            if self.aishell == True:
                read_files.append('aishell_train.txt')
            if self.prime == True:
                read_files.append('prime.txt')
            if self.stcmd == True:
                read_files.append('stcmd.txt')
            if self.vndc == True:
                read_files.append(self.vndc_train_file)
        elif self.data_type == 'dev':
            if self.thchs30 == True:
                read_files.append('thchs_dev.txt')
            if self.aishell == True:
                read_files.append('aishell_dev.txt')
        elif self.data_type == 'test':
            if self.thchs30 == True:
                read_files.append('thchs_test.txt')
            if self.aishell == True:
                read_files.append('aishell_test.txt')
            if self.vndc == True:
                read_files.append(self.vndc_test_file)
        self.wav_lst = []
        self.pny_lst = []
        self.han_lst = []
        self.place_item_lst = []
        for file in read_files:
            print('load ', file, ' data...')
            sub_file = self.data_path + file
            with open(sub_file, 'r', encoding='utf8') as f:
                data = f.readlines()
            for line in tqdm(data):
                line=line.strip("\n")
                st = line.split('\t')
                wav_file=st[0]
                pny=st[1]
                han=st[2]
                self.wav_lst.append(wav_file)
                self.pny_lst.append(pny.split(' '))
                self.han_lst.append(han.strip('\n'))
                if len(st)>=4:
                    placeitem=st[3]
                    self.place_item_lst.append(placeitem.strip('\n'))
        if self.start:
            self.wav_lst = self.wav_lst[self.start:]
            self.pny_lst = self.pny_lst[self.start:]
            self.han_lst = self.han_lst[self.start:]
            self.place_item_lst=self.place_item_lst[self.start:]
        if self.data_length:
            self.wav_lst = self.wav_lst[:self.data_length]
            self.pny_lst = self.pny_lst[:self.data_length]
            self.han_lst = self.han_lst[:self.data_length]
            self.place_item_lst=self.place_item_lst[:self.data_length]
        print('make am vocab...')
        self.am_vocab = self.mk_am_vocab('voc/voc_pinyin.txt')
        print('make p2s y pinyin vocab')
        self.pinyin_y_dict2id=self.mk_y_pinyin_dict("voc/voc_pinyin_y.txt")

    def get_am_batch(self):
        shuffle_list = [i for i in range(len(self.wav_lst))]
        while 1:
            if self.shuffle == True:
                shuffle(shuffle_list)
            for i in range(len(self.wav_lst) // self.batch_size):
                wav_data_lst = []
                label_data_lst = []
                wav_file_lst = []
                pny_lst = []
                label_list = []
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = shuffle_list[begin:end]
                for index in sub_list:
                    wav_file_lst.append(self.wav_lst[index])
                    fbank = compute_fbank(self.data_path + self.wav_lst[index])
                    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
                    pad_fbank[:fbank.shape[0], :] = fbank
                    label = self.pny2id(self.pny_lst[index], self.am_vocab)
                    label_ctc_len = self.ctc_len(label)
                    if True:
                        wav_data_lst.append(pad_fbank)
                        label_data_lst.append(label)
                pad_wav_data, input_length = self.wav_padding(wav_data_lst)
                pad_label_data, label_length = self.label_padding(label_data_lst)
                inputs = {'the_inputs': pad_wav_data,
                          'the_labels': pad_label_data,
                          'input_length': input_length,
                          'label_length': label_length,
                          'wav_file_lst': wav_file_lst,
                          'pny_lst': pny_lst,
                          'label_list':label_list,
                          }
                outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
                yield inputs, outputs
   
    def get_am_batch_p2s(self):
        shuffle_list = [i for i in range(len(self.wav_lst))]
        while 1:
            if self.shuffle == True:
                shuffle(shuffle_list)
            for i in range(len(self.wav_lst) // self.batch_size):
                wav_data_lst = []
                label_data_lst = []
                y_place_list = []
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = shuffle_list[begin:end]
                for index in sub_list:
                    fbank = compute_fbank(self.data_path + self.wav_lst[index])
                    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
                    pad_fbank[:fbank.shape[0], :] = fbank
                    label = self.pny2id(self.pny_lst[index], self.am_vocab)
                    label_ctc_len = self.ctc_len(label)
                    place_item=self.place_item_lst[index]
                    if pad_fbank.shape[0] // 8 >= label_ctc_len:
                        wav_data_lst.append(pad_fbank)
                        label_data_lst.append(label)
                        y_place_list.append(place_item)
                pad_wav_data, input_length = self.wav_padding(wav_data_lst)
                pad_label_data, label_length = self.label_padding(label_data_lst)
                inputs = {'the_inputs': pad_wav_data,
                          'the_labels': pad_label_data,
                          'input_length': input_length,
                          'label_length': label_length,
                          'place_list':y_place_list
                          }
                outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
                yield inputs, outputs
                
    def pny2id(self, line, vocab):
        idlist=[]
        for pny in line:
            try:
                if pny not in vocab:
                    idlist.append(vocab.index('unknown'))
                else:
                    idlist.append(vocab.index(pny))
            except Exception as e:
                print('error type: ',e.__class__.__name__)
                print('error explain',e)
        return idlist

    def wav_padding(self, wav_data_lst):
        wav_lens = [len(data) for data in wav_data_lst]
        wav_max_len = max(wav_lens)
        wav_lens = np.array([leng // 8 for leng in wav_lens])
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens

    def mk_am_vocab(self, file):
        vocab = []
        with open(file) as f:
            for line in f:
                pinyin = line.strip("\n")
                vocab.append(pinyin)
        vocab.append('_')
        return vocab
      
    def mk_y_pinyin_dict(self,file):
        lines = open(file, "r")
        asr20_pinyin_dict = dict()
        for i, line in enumerate(lines):
            asr20_pinyin_dict[line.strip().strip("\n")] = str(i)
        return asr20_pinyin_dict

    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len


# 对音频文件提取mfcc特征
def compute_mfcc(file):
    fs, audio = wav.read(file)
    mfcc_feat = mfcc(audio, samplerate=fs, numcep=26)
    mfcc_feat = mfcc_feat[::3]
    mfcc_feat = np.transpose(mfcc_feat)
    return mfcc_feat


# 获取信号的时频图
def compute_fbank(file):
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
    fs, wavsignal = wav.read(file)
    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    wav_arr = np.array(wavsignal)
    range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype=np.float)
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]
        # if len(np.shape(data_line))>1:
        #     temp_list=[]
        #     for index_temp in  range(np.shape(data_line)[0]):
        #         temp_list.append(data_line[index_temp][0])
        #     data_line=np.array(temp_list)
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    # data_input = data_input[::]
    return data_input


# word error rate------------------------------------
def GetEditDistance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost

# 定义解码器------------------------------------
def decode_ctc(num_result, num2word,x_res_len):
    result = num_result[:, :, :]
    # in_len = np.zeros((result.shape[0]), dtype = np.int32)
    # # in_len[0] = result.shape[1]
    # for i in range(result.shape[0]):
    #     in_len[i]=result[i,:,:].shape[0]
    in_len=x_res_len
    #print("in_len" + " ".join([str(item) for item in in_len]))
#     print(result.shape)
#     print(np.argmax(result[0], axis=-1))
    r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
    r = r[0][0]
    r1 = r.eval(session=get_my_session())
    tf.reset_default_graph() # 然后重置tf图，这句很关键
    
    # print("r:"+str(np.shape(r)))
    # print("r:"+str(r))
#     r1 = K.get_value(r[0][0])
    #print("r1:" + str(np.shape(r1)))
    textlist=[]
    for index in range(result.shape[0]):
        r2 = r1[index]
        text = []
        for i in r2:
            text.append(num2word[i])
        textlist.append(text)
    return r1, textlist


def get_my_session(gpu_fraction=1):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
      
# K.set_session(get_my_session())