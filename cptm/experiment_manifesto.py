"""Script to calculate opinion word perplexity per document for a set of text
documents.

The corpus is not divided in perspectives.

Used to estimate the likihood of party manifestos given opinions for the
different perspectives (party manifestos come from the manifesto project)
"""
import logging
import argparse
import pandas as pd
import os
import sys

from CPTCorpus import CPTCorpus
from cptm.utils.experiment import get_sampler, load_config, topicFileName, \
    get_corpus

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

logging.getLogger('gensim').setLevel(logging.ERROR)
logging.getLogger('CPTCorpus').setLevel(logging.DEBUG)
logging.getLogger('CPT_Gibbs').setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
parser.add_argument('data_dir', help='dir containing the input data.')
parser.add_argument('out_dir', help='dir to write results to.')
args = parser.parse_args()

params = load_config(args.json)

topicDict = params.get('outDir').format('topicDict.dict')
opinionDict = params.get('outDir').format('opinionDict.dict')
phi_topic_file = topicFileName(params)

phi_topic = pd.read_csv(phi_topic_file, index_col=0, encoding='utf-8').values.T

c_perspectives = get_corpus(params)
perspectives = [p.name for p in c_perspectives.perspectives]
logger.info('Perspectives found: {}'.format('; '.join(perspectives)))

input_dirs = [args.data_dir for p in perspectives]

corpus = CPTCorpus(input=input_dirs, topicDict=topicDict,
                   opinionDict=opinionDict, testSplit=100, file_dict=None,
                   topicLines=params.get('topicLines'),
                   opinionLines=params.get('opinionLines'))

# Update perspective names (default name is directory name, which is currently
# the same for all perspectives)
for p, name in zip(corpus.perspectives, perspectives):
    p.name = name

logger.info(str(corpus))

params['outDir'] = args.out_dir
nTopics = params.get('nTopics')

corpus.save(os.path.join(params.get('outDir'), 'corpus.json'))

sampler = get_sampler(params, corpus, nTopics=nTopics, initialize=False)
result = sampler.opinion_word_perplexity_per_document(phi_topic)
result.to_csv(os.path.join(params['outDir'],
                           'opinion_word_perplexity_{}.csv'.format(nTopics)),
              encoding='utf8')
