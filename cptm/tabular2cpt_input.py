"""Script that converts a field in a tabular data file to cptm input files

Used for the CAP vragenuurtje data.

Uses frog to pos-tag and lemmatize the data.

Usage: python tabular2cpt_input.py <csv of excel file> <full text field name>
<dir out>
"""

import pandas as pd
import logging
import sys
import argparse
import numpy as np

from cptm.utils.inputgeneration import Perspective, remove_trailing_digits
from cptm.utils.dutchdata import pos_topic_words, pos_opinion_words, word_types
from cptm.utils.frog import get_frogclient, pos_and_lemmas

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('in_file', help='excel or csv file containing text data')
parser.add_argument('text_field', help='name of the text field')
parser.add_argument('out_dir', help='directory where output should be stored')
args = parser.parse_args()

frogclient = get_frogclient()

number_of_words = []

if args.in_file.endswith('.xls') or args.in_file.endswith('.xlsx'):
    input_data = pd.read_excel(args.in_file)
else:
    input_data = pd.read_csv(args.in_file)

for i, text in enumerate(input_data[args.text_field]):
    p = Perspective('', pos_topic_words(), pos_opinion_words())
    if i % 25 == 0:
        logger.info('Processing text {} of {}'.format(i + 1,
                    len(input_data[args.text_field])))
    if pd.notnull(text):
        n = 0
        for pos, lemma in pos_and_lemmas(text, frogclient):
            n += 1
            if pos in word_types():
                p.add(pos, remove_trailing_digits(lemma))
        try:
            file_name = '{}.txt'.format(input_data['id'][i])
        except:
            file_name = '{}.txt'.format(i)
        p.write2file(args.out_dir, file_name)
        number_of_words.append(n)

print 'mean number of words:', np.mean(number_of_words)
print 'std number of words:', np.std(number_of_words)
