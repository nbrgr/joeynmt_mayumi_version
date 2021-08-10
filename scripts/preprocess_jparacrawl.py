# coding: utf-8
"""
Preprocess JParaCrawl
"""
from __future__ import unicode_literals
import unicodedata
import re
import os
import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict


# Japanese normalization
# taken from https://colab.research.google.com/github/sonoisa/t5-japanese/blob/main/t5_japanese_article_generation.ipynb
# see also https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja

def _unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s


def _remove_extra_spaces(s):
    s = re.sub('\u200b', '', s)
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s


def normalize_neologd(s):
    s = s.strip()
    s = _unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
                  '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = _remove_extra_spaces(s)
    s = _unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s


def prepare(data_dir, size, seed=None):
    dtype = OrderedDict({'source': str, 'probability': float, 'en': str, 'ja': str})
    df = pd.read_csv(os.path.join(data_dir, 'en-ja', 'en-ja.bicleaner05.txt'), header=None, names=dtype.keys(),
                     sep='\t', encoding='utf8', quoting=3, keep_default_na=False, na_values='', dtype=dtype)
    df = df.drop_duplicates(subset=['en', 'ja'])
    df = df[~df['en'].str.contains('�') & ~df['ja'].str.contains('�')]
    df = df[['en', 'ja']].applymap(lambda x: normalize_neologd(x))
    df = df.dropna(how='any')

    if seed is not None:
        np.random.seed(seed)
    test_index = np.random.choice(df.index, size=size, replace=False)
    train_index = np.setdiff1d(df.index, test_index)
    for lang in ['en', 'ja']:
        for data_set, drop_index in zip(['train', 'dev'], [test_index, train_index]):
            df[lang].drop(index=drop_index, inplace=False).to_csv(os.path.join(data_dir, data_set+'.'+lang),
                          header=False, index=False, sep='\t', encoding='utf8', quoting=3)


def main():
    PATH = os.path.dirname(os.path.abspath('__file__'))

    ap = argparse.ArgumentParser("Preprocess JParaCrawl")
    ap.add_argument("--data_dir", type=str, default=os.path.join(PATH, "../test/data/jparacrawl"),
                    help="path to data dir. default: ../test/data/jparacrawl")
    ap.add_argument("--dev_size", type=int, default=5000, help="development set size")
    ap.add_argument("--seed", type=int, default=12345, help="random seed for train-dev-split")
    args = ap.parse_args()

    prepare(args.data_dir, args.dev_size, args.seed)


if __name__ == "__main__":
    main()

