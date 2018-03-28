import nltk
from nltk.text import TextCollection
from nltk.probability import ConditionalFreqDist

class CorpusBase(object):
    def __init__(self, corpus, state='raw', vocab=None, labels=None, label_names=None):
        """Init function

            Args:
                state: 'raw' for untreated corpus, 'splited' for splited corpus
        """
        self.corpus = corpus
        self.corpus_nums = len(self.corpus)
        self.corpus = [{'index': i, 'doc': corpus[i]} for i in range(self.corpus_nums)]
        self.state = state
        if vocab is not None:
            self.load_vocab(vocab)
        if labels is not None:
            self.load_labels(labels, label_names)


    # def process_pipline(self, processes):
    #     if not isinstance(processes, (list, tuple)):
    #         return
    #     for process in processes:
    #         process()


    def __str__(self):
        return self.corpus

    # def __add__(self, other):
    #     if not isinstance(other, CorpusBase):
    #         return
    #     self.corpus.extend(other.corpus)
    #     return self

    def split_words(self, tokenizer):
        for item in self.corpus:
            item['doc'] = tokenizer(item['doc'])
        self.state = 'splited'


    def load_vocab(self, vocab):
        self.vocab = vocab
        self.vocab_word2index = {}
        for index in range(len(vocab)):
            self.vocab_word2index[vocab[index]] = index


    def load_labels(self, labels, label_names=None):
        labels_num = len(labels)
        if labels_num != self.corpus_nums:
            return

        if label_names is None:
            self.labels_names = list(set(labels))
        else:
            self.label_names = label_names

        self.label_name2index = {}
        for index in range(len(label_names)):
            self.label_name2index[label_names[index]] = index

        for index, item in enumerate(self.corpus):
            item['label'] = self.label_name2index[labels[index]]


    def count_word_frequency(self):
        if self.state is not 'splited':
            return

        temp_c = []
        for i in self.corpus:
            temp_c += i['doc']
        fredist = nltk.FreqDist(temp_c).most_common()
        return fredist


    def count_word_frequency_with_label(self):
        if self.state is not 'splited':
            return

        classed_c = {}
        for item in self.corpus:
            label_index = item['label']
            if label_index in classed_c:
                classed_c[label_index].extend(item['doc'])
            else:
                classed_c[label_index] = item['doc']

        classed_wf = {}
        count_labels = {}
        count_sum_w = 0
        for label_index in classed_c:
            temp_fredist = nltk.FreqDist(classed_c[label_index]).most_common()
            classed_wf[label_index] = temp_fredist
            count_labels[label_index] = len(classed_c[label_index])
            count_sum_w += count_labels[label_index]

        words_f = {}
        for label_index in classed_wf:
            for word_index in classed_wf[label_index]:
                temp_word = word_index[0]
                temp_num = word_index[1]
                if temp_word not in words_f:
                    words_f[temp_word] = {label_index: temp_num}
                else:
                    words_f[temp_word][label_index] = temp_num

        for word in words_f:
            count_word = sum(words_f[word][label] for label in words_f[word])
            for label_index in words_f[word]:
                temp_count_word_label = words_f[word][label_index]
                words_f[word][label_index] = temp_count_word_label * count_sum_w / (count_word * count_labels[label_index])

        return words_f