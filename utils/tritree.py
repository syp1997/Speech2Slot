from odps import ODPS
from pypinyin import pinyin, lazy_pinyin, Style
import pypinyin

class Tritree(object):
  def __init__(self, pad_token=0):
    self.root = {'leaf':False}
    self.pad = pad_token
    
  def add_line_candi(self, seq):
    line = seq
    cur_token = self.root
    for t in line:
      if t in cur_token:
        cur_token = cur_token[t]
      else:
        cur_token[t] = {'leaf':False}
        cur_token = cur_token[t]
    cur_token['leaf'] = True

  def get_next_candis(self, seq):

    if seq[-1] == self.pad:
      return []
    cur_t = self.root
    for i in range(len(seq)):
      t = seq[i]
      if t in cur_t:
        cur_t = cur_t[t]
      else:
        return []

    return [x for x in cur_t.keys() if x != "leaf"]

  # 使用时，在外面 new 一个 {}作为state 传入 ， 只要返回结果不为空，就一直 把得到的cur_state 放进来 ，即可 减少计算量
  def stateful_get(self, cur_token, cur_state=None):
    if cur_state is None or len(cur_state) == 0:
      cur_state = self.root

    if cur_token in cur_state:
      cur_state = cur_state[cur_token]
      return [x for x in cur_state.keys()], cur_state
    else:
      return [], {}

  def del_single_candi(self, seq):
    cur_t = self.root
    for i in range(len(seq) - 1):
      t = seq[i]
      if t in cur_t:
        cur_t = cur_t[t]
      else:
        #  表示不存在 这个词 所以不考虑
        return
    if len(cur_t[seq[-1]]) > 0:
      del cur_t[seq[-1]]


# seq = ['unk','1','2','3','4','end','unk','1','3','4','2','end','unk','1','2','4','3','end']
#
# triTree = Tritree(seq,'end')
#
# triTree.add_line_candi(['a','b','c'],'end')
# triTree.get_next_candis(['a','b','c'])
# triTree.del_single_candi(['a','b','c'])
# triTree.get_next_candis(['a','b','c'])


def build_trie_tree_from_interspeech_file(file,pinyin_dict):
  pad = 0
  start = 1
  end = 2
  trieTree = Tritree(pad_token=pad)
  with open(file) as f:
    for line in f:
      place_word=line.strip("\n")
      pinyin_res = pinyin(place_word, style=pypinyin.TONE3, heteronym=True)
      pinyin_st_pre = []
      for item in pinyin_res:
        pinyin_st_pre.append(item[0])

      res_phones = [start]
      for ele in pinyin_st_pre:
        if pinyin_dict.__contains__(ele):
          res_phones.append(pinyin_dict[ele])
      res_phones.append(end)
      for i in range(len(res_phones)):
        trieTree.add_line_candi(res_phones[i:])
  return trieTree


def build_trie_tree_reverse_from_interspeech_file(file,pinyin_dict):
  pad = 0
  start = 1
  end = 2
  trieTree = Tritree(pad_token=pad)
  with open(file) as f:
    for line in f:
      place_word = line.strip("\n")
      pinyin_res = pinyin(place_word, style=pypinyin.TONE3, heteronym=True)
      pinyin_st_pre = []
      for item in pinyin_res:
        pinyin_st_pre.append(item[0])
      pinyin_st_pre.reverse()

      res_phones = [start]
      for item in pinyin_st_pre:
        if pinyin_dict.__contains__(item):
          res_phones.append(pinyin_dict[item])
      res_phones.append(end)
      for i in range(len(res_phones)):
        trieTree.add_line_candi(res_phones[i:])
  return trieTree



if __name__ == '__main__':
  file = "coin/data/0708songs_ind.txt"
  vocab = "coin/phone.vocab"
  trie = build_trie_tree(file, vocab)
  trie.get_next_candis([1])
