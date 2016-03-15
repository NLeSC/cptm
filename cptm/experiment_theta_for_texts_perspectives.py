"""Script to extract a document/topic matrix for a set of text documents.

The corpus is not divided in perspectives.

Used to calculate theta for the CAP vragenuurtje data.

Before this script can be run, a cptm corpus should be created. Use the
tabular2cptm_input.py script to create a corpus that can be used
as input.

Usage: python experiment_theta_for_texts_perspectives.py <experiment.json>
<input dir> <output dir>
"""
import logging
import argparse
import pandas as pd
import os

from CPTCorpus import CPTCorpus
from cptm.utils.experiment import get_sampler, thetaFileName, load_config, \
    topicFileName

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
parser.add_argument('data_dir', help='dir containing the input data.')
parser.add_argument('out_dir', help='dir to write results to.')
args = parser.parse_args()

params = load_config(args.json)

input_dir = [args.data_dir]
topicDict = params.get('outDir').format('topicDict.dict')
opinionDict = params.get('outDir').format('opinionDict.dict')
phi_topic_file = topicFileName(params)

phi_topic = pd.read_csv(phi_topic_file, index_col=0, encoding='utf-8').values.T
#print phi_topic.shape
#print phi_topic

corpus = CPTCorpus(input=input_dir, topicDict=topicDict,
                   opinionDict=opinionDict, testSplit=100, file_dict=None,
                   topicLines=params.get('topicLines'),
                   opinionLines=params.get('opinionLines'))
print str(corpus)

params['outDir'] = args.out_dir
nTopics = params.get('nTopics')

for i in range(10):
    sampler = get_sampler(params, corpus, nTopics=nTopics,
                          initialize=False)
    sampler._initialize(phi_topic=phi_topic)
    sampler.run()
    sampler.estimate_parameters(start=params.get('sampleEstimateStart'),
                                end=params.get('sampleEstimateEnd'))

    logger.info('saving files')

    documents = []
    for persp in corpus.perspectives:
        print str(persp)
        for f in persp.testFiles:
            p, b = os.path.split(f)
            documents.append(b)
    theta = sampler.theta_to_df(sampler.theta, documents)
    theta.to_csv(os.path.join(params['outDir'],
                              'theta_{}_{}.csv'.format(nTopics, i)),
                 encoding='utf8')
