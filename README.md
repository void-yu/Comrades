# Comrades
一个中文NLP处理工具包
A easy used Chinese NLP toolkit.

## Dependencies
- python >= 3.6
- numpy
- pandas
- nltk


## In a Nutshell
一些API

```python
    class CorpusBase(object):
    # 需要先使用CorpusBase读取文本，文本类型为list(str)
        def __init__(self,
            corpus,
            state='raw',
            vocab=None,
            labels=None,
            label_names=None)

        # CorpusBase.split_words使用任意分词器tokenizer进行分词
        def split_words(self, tokenizer)

        # CorpusBase.load_vocab加载词汇表
        def load_vocab(self, vocab)

        # CorpusBase.load_labels加载文本标签
        def load_labels(self, labels, label_names=None)

        # CorpusBase.count_word_frequency对已分好词的文本统计词频
        def count_word_frequency(self)

        # CorpusBase.count_word_frequency对已分好词并有类别标记的文本分类别统计词频
        def count_word_frequency_with_label(self)
```
