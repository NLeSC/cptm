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
import re

from pynlpl.clients.frogclient import FrogClient

from cptm.utils.inputgeneration import Perspective, remove_trailing_digits
from cptm.utils.dutchdata import pos_topic_words, pos_opinion_words, word_types

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('in_file', help='excel or csv file containing text data')
parser.add_argument('text_field', help='name of the text field')
parser.add_argument('out_dir', help='directory where output should be stored')
args = parser.parse_args()

port = 8020

try:
    frogclient = FrogClient('localhost', port)
except:
    logger.error('Cannot connect to the Frog server. '
                 'Is it running at port {}?'.format(port))
    logger.info('Start the Frog server with "docker run ''-p 127.0.0.1:{}:{} '
                '-t -i proycon/lamachine frog -S {}"'.format(port, port, port))
    sys.exit(1)

regex = re.compile(r'\(.*\)')

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
        for data in frogclient.process(text):
            word, lemma, morph, ext_pos = data[:4]
            if ext_pos:  # ext_pos can be None
                pos = regex.sub('', ext_pos)
                if pos in word_types():
                    p.add(pos, remove_trailing_digits(lemma))
        file_name = '{}.txt'.format(i)
        p.write2file(args.out_dir, file_name)
