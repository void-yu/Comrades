import pickle
import re

import jieba
import numpy as np
import pandas as pd
import reader


def only_read_exls():
    df = pd.read_excel('data/corpus/excel/1_full.xlsx')
    text = df.iloc[:, [0, 1, 2]]
    text = [[str(i[1]), str(i[2]), i[0]] for i in np.array(text).tolist()]
    # print(np.shape(text))
    return text
    # df_ = pd.DataFrame()

def read_exls(with_labels=False):
    df = pd.read_excel('data/corpus/check/yhwc.xlsx')
    if with_labels is False:
        text = df.iloc[:, [0]]
        text = [[i[0]] for i in np.array(text).tolist()]
    else:
        # print(df)
        text = df.iloc[:, [0, 1]]
        text = [[i[0], i[1]] for i in np.array(text).tolist()]
        text = np.array(text).tolist()
    return text

# text = read_exls(True)
# with open('data/corpus/check/yhwc', 'wb') as fp:
#     pickle.dump(text, fp)


def process_text(text):
    glossary, word2id = reader.read_glossary()
    for index, item in enumerate(text):
        temp = list(jieba.cut(item[0]))
        for jndex, jtem in enumerate(temp):
            if jtem in [' ', '\u3000', '    ']:
                temp.remove(jtem)
        for jndex, jtem in enumerate(temp):
            word = jtem.lower()
            word = re.sub(r'[0-9]+', '^数', word)
            if word not in glossary:
                word = '^替'
            temp[jndex] = word
        temp.append('^终')
        num = len(temp)
        text[index][0] = [temp, num]

    max_len = max([i[0][1] for i in text])
    for index, item in enumerate(text):
        while(len(item[0][0])) < max_len:
            item[0][0].append('^填')

    for item in text:
        item[0][0] = [word2id[word] for word in item[0][0]]

    with open('data/corpus/train_2', 'wb') as fp:
        pickle.dump(text, fp)



def process_double_text(text):
    glossary, word2id = reader.read_glossary()
    new_text = []
    for index, item in enumerate(text):
        new_temp = [[], item[2]]
        temp = list(jieba.cut(item[0]))
        temp_ = list(jieba.cut(item[1]))
        for jndex, jtem in enumerate(temp):
            if jtem in [' ', '\u3000', '    ']:
                temp.remove(jtem)
        for jndex, jtem in enumerate(temp):
            word = jtem.lower()
            word = re.sub(r'[0-9]+', '^数', word)
            if word not in glossary:
                word = '^替'
            temp[jndex] = word
        temp.append('^终')
        for jndex, jtem in enumerate(temp_):
            if jtem in [' ', '\u3000', '    ']:
                temp_.remove(jtem)
        for jndex, jtem in enumerate(temp_):
            word = jtem.lower()
            word = re.sub(r'[0-9]+', '^数', word)
            if word not in glossary:
                word = '^替'
            temp_[jndex] = word
        temp_.append('^终')
        temp.extend(temp_)
        num = len(temp)
        new_temp[0] = [temp, num]
        new_text.append(new_temp)

    max_len = max([i[0][1] for i in new_text])
    for index, item in enumerate(new_text):
        while (len(item[0][0])) < max_len:
            item[0][0].append('^填')

    with open('data/corpus/train_2_oigin', 'wb') as fp:
        pickle.dump(new_text, fp)

    for item in new_text:
        item[0][0] = [word2id[word] for word in item[0][0]]

    with open('data/corpus/train_2', 'wb') as fp:
        pickle.dump(new_text, fp)

# text = only_read_exls()
# process_double_text(text)

def read_text():
    with open('data/corpus/test_2', 'rb') as fp:
        text = pickle.load(fp)
    text_ = []
    coll = []

    # for i in text:
    #     # print(i)
    #     sent = i[0][0]
    #     num = 0
    #     for j in sent:
    #         if j == 35149:
    #             num += relu
    #     rate = num/i[0][relu]
    #     if rate > 0.3:
    #         print(rate, i)
    #         text_.append(i)

    # print(len(text_))
    # with open('data/corpus/test_1_0.2', 'wb') as fp:
    #     pickle.dump(text_, fp)
    print(np.shape(text))
    print(np.shape(text_))
    print(len(text))
    print(max([i[0][1] for i in text]))


def shuffle_text():
    with open('data/corpus/train_2_origin', 'rb') as fp:
        text = pickle.load(fp)
    import random
    random.shuffle(text)
    piece_lenth = len(text) // 11
    train = text[:piece_lenth*9]
    valid = text[piece_lenth*9:piece_lenth*10]
    test = text[piece_lenth*10:]
    with open('data/corpus/train_2', 'wb') as fp:
        pickle.dump(train, fp)
    with open('data/corpus/valid_2', 'wb') as fp:
        pickle.dump(valid, fp)
    with open('data/corpus/test_2', 'wb') as fp:
        pickle.dump(test, fp)



def read_results():
    file = open('results/test_1_results', 'r', encoding='utf8')
    content = file.readlines()
    for i in content:
        item = i.split()
        if item[1] == '0.0':
            print(item[0], item[1], item[2])




def afterProcess():
    with open('data/corpus/raw', 'rb') as fp:
        text = pickle.load(fp)
    for item in text:
        if item[1][0][0] == '^填':
            item[1] = [[['^填']*120, 0]]
        item[1][0][0] = [WORD2ID[word] for word in item[1][0][0]]
        for jtem in item[2]:
            if jtem == []:
                jtem.append([['^填'] * 120, 0])
            jtem[0] = [WORD2ID[word] for word in jtem[0]]
        if item[2] == []:
            item[2] = [[[WORD2ID['^填']]*120, 0]]
    with open('data/corpus/raw_', 'wb') as fp:
        pickle.dump(text, fp)


def write():
    with open('data/corpus/raw_', 'rb') as fp:
        text = pickle.load(fp)
    text_true = []
    text_false = []
    for item in text:
        if item[0] == 'T':
            text_true.append(item)
        elif item[0] == 'F':
            text_false.append(item)
    with open('data/corpus/raw_true', 'wb') as fp:
        pickle.dump(text_true, fp)
    with open('data/corpus/raw_false', 'wb') as fp:
        pickle.dump(text_false, fp)

def sample_from_raw():
    with open('data/corpus/raw_true', 'rb') as fp:
        text_true = pickle.load(fp)
    with open('data/corpus/raw_false', 'rb') as fp:
        text_false = pickle.load(fp)
