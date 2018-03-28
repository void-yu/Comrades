import pandas as pd
import numpy as np
import pickle

DATA_BASE = '../data/'

def read_excel(file_name, text_column, label_column=None):
    # Read File with '.xlsx' format
    df = pd.read_excel(DATA_BASE + file_name)
    if label_column is not None:
        corpus = df.iloc[:, [text_column, label_column]]
        corpus = np.array(corpus).tolist()
    else:
        corpus = df.iloc[:, [text_column]]
        corpus = np.array(corpus).tolist()
    corpus = list(filter(lambda x: pd.isnull(x[0]) is False, corpus))
    corpus = [[i]+corpus[i] for i in range(len(corpus))]
    return corpus

def read_pickle(file_name):
    pf = open(DATA_BASE + file_name, 'rb')
    corpus = pickle.load(pf)
    # for i in corpus:
    #     corpus = [0] + corpus[i]
    corpus = [[i]+corpus[i]for i in range(len(corpus))]
    return corpus


# c = read_excel('1_full.xlsx', text_column=1, label_column=0)
# c = read_pickle('test_klb_150_raw')
# print(c[0])