#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Create input files in cptm format from manifesto project csv files.

Usage: python manifestoproject2cptm_input.py <input dir> <output dir>
"""
import pandas as pd
import logging
import argparse
import os
import glob

from cptm.utils.inputgeneration import Perspective, remove_trailing_digits
from cptm.utils.dutchdata import pos_topic_words, pos_opinion_words, word_types
from cptm.utils.frog import get_frogclient, pos_and_lemmas

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger.setLevel(logging.DEBUG)
logging.getLogger('inputgeneration').setLevel(logging.DEBUG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_in', help='directory containing the data '
                        '(manifesto project csv files)')
    parser.add_argument('dir_out', help='the name of the dir where the '
                        'CPT corpus should be saved.')
    args = parser.parse_args()

    dir_in = args.dir_in
    dir_out = args.dir_out

    frogclient = get_frogclient()

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    data_files = glob.glob('{}/*.csv'.format(dir_in))

    for i, data_file in enumerate(data_files):
        if i % 5 == 0:
            logger.info('Processing text {} of {}'.format(i + 1,
                        len(data_files)))
        p = Perspective('', pos_topic_words(), pos_opinion_words())
        df = pd.read_csv(data_file, encoding='utf-8')
        text = ' '.join([line for line in df['content']])
        try:
            for pos, lemma in pos_and_lemmas(text, frogclient):
                if pos in word_types():
                    p.add(pos, remove_trailing_digits(lemma))
        except Exception, e:
            logger.warn(str(e))
            del frogclient
            frogclient = get_frogclient()
            logger.info('parsing pseudo sentences instead')
            for text in df['content']:
                for pos, lemma in pos_and_lemmas(text, frogclient):
                    if pos in word_types():
                        p.add(pos, remove_trailing_digits(lemma))

        file_name = os.path.basename(data_file).replace('.csv', '.txt')
        p.write2file(dir_out, file_name)
