import reader
from comrades.base import CorpusBase

import jieba
import numpy as np
import pandas as pd

DATA_BASE = '../data/'

c = reader.read_excel('multi-labeled.xlsx', 1, 0)
cc = reader.read_excel('1_full.xlsx', 1, 0)
c1 = reader.read_pickle('test_klb_150_raw')
c2 = reader.read_pickle('test_yhwc_150_raw')
c3 = reader.read_pickle('test_yxcd_150_raw')

c_cc = c.extend(cc)
p_c_cc = CorpusBase([i[1] for i in c], labels=[i[2] for i in c], label_names=['F', 'T'])
p_c_cc.split_words(lambda x: list(jieba.cut(x)))
classed_wf = p_c_cc.count_word_frequency_with_label()


# write2file = []
# for word in classed_wf:
#     temp = [word]
#     if 0 in classed_wf[word]:
#         temp.append(round(classed_wf[word][0], 4))
#     else:
#         temp.append(0)
#     if 1 in classed_wf[word]:
#         temp.append(round(classed_wf[word][1], 4))
#     else:
#         temp.append(0)
#     write2file.append(temp)
#
# for i in write2file:
#     if i[1] == 0 and i[2] > 2.5:
#         print('T', i)
#     elif i[2] == 0 and i[1] > 2:
#         print('F', i)
#     elif i[1] != 0 and i[2] != 0:
#         if i[1] / i[2] > 2:
#             print('F', i)
#         elif i[2] / i[1] > 2:
#             print('T', i)
