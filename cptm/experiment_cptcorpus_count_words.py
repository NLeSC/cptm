"""Count the number of topic and opinion words in the corpus.

Usage: python experiment_cptcorpus_count_words.py <experiment.json>
"""
import logging
import argparse
import sys
import tarfile
import os

from utils.experiment import load_config, get_corpus, get_sampler, \
    thetaFileName, topicFileName, opinionFileName, tarFileName


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
args = parser.parse_args()

config = load_config(args.json)
corpus = get_corpus(config)

num_topic_words = 0
num_opinion_words = 0


for d, persp, d_p, doc in corpus:
    for w_id, i in corpus.words_in_document(doc, 'topic'):
        num_topic_words += i

    for w_id, i in corpus.words_in_document(doc, 'opinion'):
        num_opinion_words += i

print 'Number of topic words in corpus', num_topic_words
print 'Number of opinion words in corpus', num_opinion_words
