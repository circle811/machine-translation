import bz2
import collections
import os
import torch

__all__ = ['load_parallel_en_zh']

PAD, SOS, EOS = range(3)
PAD_STR, SOS_STR, EOS_STR = '<pad>', '<sos>', '<eos>'


class Language:
    def __init__(self, name, raw_sentences):
        self.name = name
        self.raw_sentences = raw_sentences
        self.n_sentences = len(raw_sentences)

        counter = collections.Counter(w for s in raw_sentences for w in s)
        self.word_count = sorted(counter.items(), key=lambda wc: (-wc[1], wc[0]))
        self.int_to_word = [PAD_STR, SOS_STR, EOS_STR] + [w for w, c in self.word_count]
        self.word_to_int = {w: i for i, w in enumerate(self.int_to_word)}
        self.n_words = len(self.int_to_word)

    def to_sentences(self, raw_sentences, eos=False):
        if eos:
            return [torch.tensor([self.word_to_int[w] for w in s] + [EOS], dtype=torch.int64)
                    for s in raw_sentences]
        else:
            return [torch.tensor([self.word_to_int[w] for w in s], dtype=torch.int64)
                    for s in raw_sentences]

    def to_raw_sentences(self, sentences):
        raw_sentences = []
        for s in sentences:
            r = []
            for w in s:
                if w == EOS:
                    break
                r.append(self.int_to_word[w])
            raw_sentences.append(r)
        return raw_sentences


def load_parallel_en_zh():
    file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/parallel_en_zh.txt.bz2'))

    en_raw_sentences = []
    zh_raw_sentences = []
    with bz2.open(file_name, 'rt') as f:
        for line in f:
            e, z = line.strip().split('\t')
            en_raw_sentences.append(e.split(' '))
            zh_raw_sentences.append(z.split(' '))

    en_lang = Language('en', en_raw_sentences)
    zh_lang = Language('zh', zh_raw_sentences)

    return en_lang, zh_lang
