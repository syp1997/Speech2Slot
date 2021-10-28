# -*- coding: utf-8 -*
import os
import numpy as np


# 输入压缩后的x或者y1或者y2文件
# 返回一个list，每个元素对应一个instance的x或者y1或者y2
# 需要手动指定分辨率，默认为256
# 每份数据共分3个文件，分别是x文件、y1文件和y2文件
# 返回的list长度相等，对齐后就是一个train/test instance的[x, y1, y2]
def load_bytes_to_ram(f_path, file_type="x", resolution=256):
    ret = []

    f = open(f_path, "rb")
    iterator = 0

    file_size = os.path.getsize(f_path)
    print('loading file:', f_path)
    print('file size:', file_size)

    if file_type == "x":
        if resolution == 256:
            step = 2
        elif resolution == 65536:
            step = 3
        else:
            print("resolution can only be 256 or 65536")
            return None
    elif file_type == "y":
        step = 2
    else:
        print('file type can only be x or y!')
        return None

    compress_bytes = f.read()

    while iterator < file_size:
        length = (compress_bytes[iterator] << 8) + compress_bytes[iterator + 1]
        iterator += 2
        ret.append(compress_bytes[iterator: iterator + step * length])
        iterator += step * length
    return ret


# 输入一个instance对应的256分辨率的字节特征
# 返回一个dense的ndarray
def get_one_instance_x_256(x_bytes, max_len1=104, phone_vocab_size=124):
    iterator = 0
    ret = np.zeros((max_len1, phone_vocab_size), dtype=float)
    valid_pos = np.zeros(max_len1, dtype=int)
    offset = 0

    while iterator < len(x_bytes):
        rel = x_bytes[iterator]
        val_x = x_bytes[iterator + 1]
        offset += rel
        idx1 = offset // phone_vocab_size
        idx2 = offset % phone_vocab_size

        if idx1 >= max_len1 or idx2 >= phone_vocab_size:
            return None, None

        ret[idx1, idx2] = val_x / 256.0
        if idx2 > 3 and ret[idx1, idx2] > 0.5:
            valid_pos[idx1] = idx1
        iterator += 2
    return ret, valid_pos



def get_one_instance_x_256_xjh(x_bytes, max_len1=104, phone_vocab_size=124):
    iterator = 0
    step = 2

    compress_bytes = x_bytes
    length = (compress_bytes[iterator] << 8) + compress_bytes[iterator + 1]
    iterator += 2
    compress_bytes = compress_bytes[iterator: iterator + step * length]

    iterator = 0
    ret = np.zeros((max_len1, phone_vocab_size), dtype=float)
    valid_pos = np.zeros(max_len1, dtype=int)
    offset = 0

    while iterator < len(compress_bytes):
        rel = compress_bytes[iterator]
        val_x = compress_bytes[iterator + 1]
        offset += rel
        idx1 = offset // phone_vocab_size
        idx2 = offset % phone_vocab_size

        if idx1 >= max_len1 or idx2 >= phone_vocab_size:
            return None, None

        ret[idx1, idx2] = val_x / 256.0
        if idx2 > 3 and ret[idx1, idx2] > 0.5:
            valid_pos[idx1] = idx1
        iterator += 2
    return ret, valid_pos, compress_bytes


# 输入一个instance对应的65536分辨率的字节特征
# 返回一个dense的ndarray
def get_one_instance_x_65536(x_bytes, max_len1=104, phone_vocab_size=124):
    iterator = 0
    ret = np.zeros((max_len1, phone_vocab_size), dtype=float)
    valid_pos = np.zeros(max_len1, dtype=int)

    offset = 0

    while iterator < len(x_bytes):
        rel = x_bytes[iterator]
        val_x = (x_bytes[iterator + 1] << 8) + x_bytes[iterator + 2]
        offset += rel
        idx1 = offset // phone_vocab_size
        idx2 = offset % phone_vocab_size
        if idx1 >= max_len1 or idx2 >= phone_vocab_size:
            return None, None
        ret[idx1, idx2] = val_x / 65536.0
        if idx2 > 3 and ret[idx1, idx2] > 0.5:
            valid_pos[idx1] = idx1
        iterator += 3
    return ret, valid_pos

def get_one_instance_x_asr20(x_bytes, max_len1=52):
    ret_list = []
    valid_pos_list = []
    for line in x_bytes:
        ret = np.zeros(max_len1, dtype=int)
        valid_pos = np.zeros(max_len1, dtype=int)
        line = line.strip().split(" ")
        for i, x in enumerate(line):
            ret[i] = x
            valid_pos[i] = i
        ret_list.append(ret)
        valid_pos_list.append(valid_pos)
    return ret_list, valid_pos_list

# 输入一个instance对应的y字节特征
# 返回一个dense的ndarray
def get_one_instance_y(y_bytes, max_len2=40):
    iterator = 0
    ret = np.zeros(max_len2, dtype=int)

    while iterator < len(y_bytes):
        pos_y = y_bytes[iterator]
        val_y = y_bytes[iterator + 1]
        if pos_y >= max_len2:
            return None
        ret[pos_y] = val_y
        iterator += 2

    return ret

def get_one_instance_y_2(y_bytes,index2phone, max_len2=40):
    iterator = 0
    list_temp=[]
    ret = np.zeros(max_len2, dtype=int)
    flag=False
    while iterator < len(y_bytes):
        pos_y = y_bytes[iterator]
        val_y = y_bytes[iterator + 1]
        if(val_y>3):
            list_temp.append(index2phone[val_y])
        if pos_y >= max_len2:
            flag=True
            iterator += 2
            continue
        ret[pos_y] = val_y
        iterator += 2
    if flag:
        ret=None
    return ret,list_temp

def get_one_instance_y_asr20(y_bytes, max_len2=20):
    ret_list = []
    for line in y_bytes:
        ret = np.zeros(max_len2, dtype=int)
        if line.strip() != "":
            line = line.strip().split(" ")
            for i, x in enumerate(line):
                ret[i] = x
        ret_list.append(ret)
    return ret_list

def get_label(y_bytes):
    ret_list=[]
    for line in y_bytes:
        ret_list.append(int(line.strip()))
    return ret_list

# 使用示例
if __name__ == '__main__':

    ret_val_x = load_bytes_to_ram("coin/data/compress/train_90_sil_256_res_no_norm.x.bin", file_type="x")
    ret_val_y1 = load_bytes_to_ram("coin/data/compress/train_90_sil_256_res_no_norm.y1.bin", file_type="y")
    ret_val_y2 = load_bytes_to_ram("coin/data/compress/train_90_sil_256_res_no_norm.y2.bin", file_type="y")
    print(len(ret_val_x))
    print(len(ret_val_y1))
    print(len(ret_val_y2))
