import logging
import pandas as pd
import argparse
import sys
import tarfile

from utils.experiment import load_config, get_corpus, get_sampler, \
    thetaFileName, topicFileName, opinionFileName, tarFileName


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
args = parser.parse_args()

config = load_config(args.json)

nTopics = config.get('nTopics')
start = config.get('sampleEstimateStart')
end = config.get('sampleEstimateEnd')

if nTopics is None or start is None or end is None:
    logger.error('nTopics ({}), sampleEstimateStart ({}), and '
                 'sampleEstimateEnd ({}) cannot be None'.
                 format(nTopics, start, end))
    logger.error('Please update {} to include these parameters'.
                 format(args.json))
    sys.exit()

corpus = get_corpus(config)
sampler = get_sampler(config, corpus, nTopics=None, initialize=False)
logger.info('estimating parameters')
sampler.estimate_parameters(start=start, end=end)

logger.info('saving files')

documents = []
for persp in corpus.perspectives:
    for f in persp.corpus().input:
        f = f.replace(config.get('inputData').replace('*', ''), '')
        documents.append(f)
theta = sampler.theta_to_df(sampler.theta, documents)
theta.to_csv(thetaFileName(config), encoding='utf8')

topics = sampler.topics_to_df(phi=sampler.topics, words=corpus.topic_words())
topics.to_csv(topicFileName(config), encoding='utf8')

for i, p in enumerate(sampler.corpus.perspectives):
    opinions = sampler.topics_to_df(phi=sampler.opinions[i],
                                    words=corpus.opinion_words())
    opinions.to_csv(opinionFileName(config, p.name), encoding='utf8')

logger.info('creating .tgz')
with tarfile.open(tarFileName(config), "w:gz") as tar:
    for name in [thetaFileName(config), topicFileName(config)] + \
                [opinionFileName(config, p.name)
                    for p in sampler.corpus.perspectives]:
            tar.add(name)
