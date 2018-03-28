import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import re
import jieba

def read_glossary():
    with open('../data/glossary', 'rb') as fp:
        id2word = pickle.load(fp)
        word2id = {}
        for index in range(len(id2word)):
            word2id[id2word[index]] = index
    return id2word, word2id

def read_corpus(filepath='', shuffled=False):
    with open('../data/corpus/%s' % filepath, 'rb') as fp:
        corpus = pickle.load(fp)
    if shuffled is True:
        np.random.shuffle(corpus)
    return corpus

def read_initw2v():
    with open('../data/initw2v_use', 'rb') as fp:
        embedding = pickle.load(fp)
    return embedding

def read_wordsim240():
    file_sim = open('../data/words-240/Words-240.txt', 'r', encoding='utf8')
    ws240_list = []
    for iter in file_sim.readlines():
        temp = iter[:-1].split()
        temp[2] = float(temp[2])
        ws240_list.append(temp)
    return ws240_list

def read_wordsim297():
    file_sim = open('../data/words-297/297.txt', 'r', encoding='utf8')
    ws297_list = []
    for iter in file_sim.readlines():
        temp = iter[:-1].split()
        temp[2] = float(temp[2])
        ws297_list.append(temp)
    return ws297_list

def restore_from_checkpoint(sess, saver, dir):
    ckpt = tf.train.get_checkpoint_state(dir)
    # print(ckpt)
    if not ckpt or not ckpt.model_checkpoint_path:
        print('No checkpoint found at {}'.format(dir))
        return False
    saver.restore(sess, ckpt.model_checkpoint_path)
    return True


def read_excel(file_path, text_column, label_column=None):
    # Read File with '.xlsx' format
    df = pd.read_excel(file_path)
    if label_column is not None:
        corpus = df.iloc[:, [text_column, label_column]]
        corpus = np.array(corpus).tolist()
    else:
        corpus = df.iloc[:, [text_column]]
        corpus = np.array(corpus).tolist()
    corpus = filter(lambda x: pd.isnull(x[0]) is False, corpus)
    return corpus

def preprocess(corpus, seq_lenth, seq_num, overlap_lenth, input_label=True, output_index=True, de_duplicated=True, split_func=lambda x:list(jieba.cut(x))):
    # Read Vocabulary
    vocab, word2id = read_glossary()
    if de_duplicated is True:
        corpus_t = []
        set_dd = set()
        for i in corpus:
            if i[0] not in set_dd:
                set_dd.add(i[0])
                corpus_t.append(i)
            else:
                continue
        corpus = corpus_t
        del corpus_t

    # Process corpus
    corpus_t = []
    total_lenth = seq_lenth*seq_num - overlap_lenth*(seq_num - 1)
    for index, item in enumerate(corpus):
        text_t = split_func(str(item[0]))
        if len(text_t) > total_lenth:
            print(len(text_t), '>', total_lenth, 'Sequence lenth is not enough~')
            continue
        for jndex, jtem in enumerate(text_t):
            if jtem in [' ', '\u3000', '    ']:
                text_t.remove(jtem)
        for jndex, jtem in enumerate(text_t):
            word = jtem.lower()
            word = re.sub(r'[0-9]+', '^数', word)
            if word not in vocab:
                word = '^替'
            text_t[jndex] = word
        num = len(text_t)
        while len(text_t) < total_lenth:
            text_t.append('^填')
        item_t = [text_t, num]
        if input_label is True:
            item_t.append(item[1])
        if output_index is True:
            item_t.append(index)
        corpus_t.append(item_t)
    corpus = corpus_t
    del corpus_t

    # Replace Words to IDs
    for index, item in enumerate(corpus):
        corpus[index][0] = [word2id[word] for word in corpus[index][0]]

    # Split corpus
    corpus_t = []
    for item in corpus:
        for jndex in range(seq_num):
            start_dex = jndex*seq_lenth - jndex*overlap_lenth
            end_dex = (jndex + 1)*seq_lenth - jndex*overlap_lenth
            # print(start_dex, end_dex)
            num_t = item[1] - start_dex
            num_t = num_t if num_t > 0 else 0
            item_t = [item[0][start_dex:end_dex], num_t]
            if input_label is True:
                item_t.append(item[2])
            if output_index is True:
                item_t.append(item[-1])
            corpus_t.append(item_t)
    corpus = corpus_t
    del corpus_t

    return corpus
