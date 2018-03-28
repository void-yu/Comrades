import numpy as np
import collections
import matplotlib.pyplot as plt
import re
import random
import jieba

class Text(object):
    def __init__(self, sents, language='english'):
        try:
            if not isinstance(sents, list):
                raise TypeError("input 'sents' must be type 'list'")
        except TypeError as e:
            print(e)
            return
        self.sents = sents
        self.error_list = []
        if language in ('english', 'English'):
            self.split_english_text()
        elif language in ('chinese', 'Chinese'):
            self.split_chinese_text()
        elif language in ('chinese_use_jieba'):
            self.split_chinese_text_use_jieba()

    def split_english_text(self):
        for index, sent in enumerate(self.sents):
            if not isinstance(sent, str):
                self.error_list.append(index)
                self.sents[index] = []
                continue
            self.sents[index] = [word for word in sent.split()]

    @staticmethod
    def is_alphabet(uchar):
        if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
            return True
        else:
            return False

    def split_chinese_text(self):
        for index, sent in enumerate(self.sents):
            if not isinstance(sent, str):
                self.error_list.append(index)
                self.sents[index] = []
                continue

            word_temp = None
            last_special_flag = None
            list_temp = []
            for i in range(len(sent)):
                if sent[i].isdigit():
                    special_flag = 'd'
                elif Text.is_alphabet(sent[i]):
                    special_flag = 'a'
                else:
                    special_flag = 'o'
                if last_special_flag is None and special_flag is 'o':
                    list_temp.append(sent[i])
                elif last_special_flag is None and special_flag is not 'o':
                    word_temp = sent[i].lower()
                elif last_special_flag is 'o' and special_flag is 'o':
                    list_temp.append(sent[i])
                elif last_special_flag is 'o' and special_flag is not 'o':
                    word_temp = sent[i].lower()
                elif last_special_flag is 'd' and special_flag is 'd':
                    word_temp += sent[i]
                elif last_special_flag is 'd' and special_flag is 'o':
                    list_temp.append(word_temp)
                    list_temp.append(sent[i])
                elif last_special_flag is 'd' and special_flag is 'a':
                    list_temp.append(word_temp)
                    word_temp = sent[i].lower()
                elif last_special_flag is 'a' and special_flag is 'a':
                    word_temp += sent[i].lower()
                elif last_special_flag is 'a' and special_flag is 'o':
                    list_temp.append(word_temp)
                    list_temp.append(sent[i])
                elif last_special_flag is 'a' and special_flag is 'd':
                    list_temp.append(word_temp)
                    word_temp = sent[i]
                last_special_flag = special_flag
                i += 1
            self.sents[index] = list_temp

    def split_chinese_text_use_jieba(self):
        for index, sent in enumerate(self.sents):
            if not isinstance(sent, str):
                self.error_list.append(index)
                self.sents[index] = []
                continue
            self.sents[index] = jieba.cut(sent)


    def collectWordDict(self):
        words = []
        for sent in self.sents:
            words.extend(sent)
        word_count = collections.Counter(words).most_common()
        return word_count

    def lenthPLTDetect(self, pic_show=True, log_log=False):
        lenth_list = [len(iter) for iter in self.sents]
        global_count = collections.Counter(lenth_list)
        global_count = global_count.most_common()
        global_count.sort()
        print(global_count)
        count = np.array([t[0] for t in global_count])
        number = np.array([t[1] for t in global_count])

        if pic_show is True:
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ax.scatter(count, number, s=50,
                       label='point',
                       alpha=0.3,
                       edgecolors='none')
            ax.set_ylabel('count')
            ax.set_xlabel('frequency')
            ax.legend()
            if log_log is True:
                ax.set_yscale('log')
                ax.set_xscale('log')
            ax.grid(True)
            plt.show()




class TextProcess(object):

    @staticmethod
    def sentencesAlign(text, fill_signal='*', upper_bound=None, set_len=None):
        if not isinstance(text, Text):
            return

        if upper_bound is not None:
            text.sents = [sent for sent in text.sents if len(sent) < upper_bound]

        max_len = max([len(sent) for sent in text.sents])
        if set_len is not None:
            set_len = max(max_len, set_len)

        for sent in text.sents:
            len_sent = len(sent)
            sent.extend([fill_signal] * (set_len-len_sent))
        # text.sents = [sent.ljust(max_len, fill_word) for sent in text.sents]
        return text.sents, set_len


    @staticmethod
    def addStartSignal(text, start_signal='>'):
        if not isinstance(text, Text):
            return
        for sent in text.sents:
            sent.insert(0, start_signal)


    @staticmethod
    def sentencesShuffle(text):
        if not isinstance(text, Text):
            return
        random.shuffle(text.sents)
        return text.sents




class SentProcess(object):

    @staticmethod
    def englishWordRevise(word_list):
        err_list = []
        temp_list = []
        glossary = ['.', ',', '"', '?', ':', ';', '!', '(', ')', '&']
        for word in word_list:
            word = word.strip('\n')
            if re.match(re.compile(r'^[a-z]+$'), word) is None:
                err_list.append(word)
            elif word not in glossary:
                glossary.append(word)

        for word in err_list:
            word = re.sub(r'[\-\/]+', ' ', word)
            word = re.sub(r'[`\']+', '', word)
            word = re.sub(r'[\[\]]+', '', word)
            word = re.sub(r'[0-9]+', '^', word)
            words = word.split(' ')
            for item in words:
                if item not in glossary:
                    temp_list.append(item)
        err_list = temp_list
        temp_list = []

        for word in err_list:
            if re.match(re.compile(r'^[a-z\'\^]+$'), word) is None:
                temp_list.append(word)
            elif word not in glossary:
                glossary.append(word)
        return glossary




    @staticmethod
    def chaosSentSubstitution(sent,
                              rm_url=True, rep_url='*',
                              rm_punc=True, rep_punc='*',
                              rm_numb=True, rep_numb='^'):
        sent = str(sent)
        if rm_url is True:
            pa_url = re.compile(
                r'[a-zA-Z]+://[^\s\u4e00-\u9fa5]*')
            sent = re.sub(pa_url, rep_url, sent)
        if rm_punc is True:
            pa_punctuation = re.compile(
                r'[\s+\.\!\/_\-\\\|,:;$%~@?^*()=+\"\'<>\[\]]+|' \
                r'[+＋——！，：；。？、#￥％……&*（）‘’“”《》【】『』]+')
            sent = re.sub(pa_punctuation, rep_punc, sent)
        if rm_numb is True:
            sent = re.sub(r'[0-9]+', rep_numb, sent)
        return sent
