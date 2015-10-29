"""Calculate opinion perplexity for different numbers of topics

Calclulate opinion perplexity for the test set as described in [Fang et al.
2012] section 5.1.1.

This script should be run after experiment_number_of_topics.py.

Usage: python cptm/experiment_calculate_perplexity.py /path/to/experiment.json.
"""
import pandas as pd
import logging
from multiprocessing import Pool
import argparse

from cptm.utils.experiment import load_config, get_corpus, get_sampler


def calculate_perplexity(config, corpus, nPerplexity, nTopics):
    sampler = get_sampler(config, corpus, nTopics, initialize=False)

    results = []
    for s in nPerplexity:
        logger.info('doing perplexity calculation ({}, {})'.format(nTopics, s))
        tw_perp, ow_perp = sampler.perplexity(index=s)
        results.append((nTopics, s, tw_perp, ow_perp))
    logger.info('finished perplexity calculation for {} topics'.
                format(nTopics))
    return results


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

logging.getLogger('gensim').setLevel(logging.ERROR)
logging.getLogger('CPTCorpus').setLevel(logging.ERROR)
logging.getLogger('CPT_Gibbs').setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
args = parser.parse_args()

config = load_config(args.json)
corpus = get_corpus(config)

nTopics = config.get('expNumTopics')
nPerplexity = [0] + range(9, config.get('nIter')+1, 10)

# calculate perplexity
pool = Pool(processes=config.get('nProcesses'))
results = [pool.apply_async(calculate_perplexity, args=(config, corpus,
                            nPerplexity, n))
           # reverse list, so longest calculation is started first
           for n in nTopics[::-1]]
pool.close()
pool.join()

# aggrate and save results
data = [p.get() for p in results]

topic_perp = pd.DataFrame(columns=nTopics, index=nPerplexity)
opinion_perp = pd.DataFrame(columns=nTopics, index=nPerplexity)

for result in data:
    for n, s, tw_perp, ow_perp in result:
        topic_perp.set_value(s, n, tw_perp)
        opinion_perp.set_value(s, n, ow_perp)

outDir = config.get('outDir')
logger.info('writing perplexity results to {}'.format(outDir.format('')))
topic_perp.to_csv(outDir.format('perplexity_topic.csv'))
opinion_perp.to_csv(outDir.format('perplexity_opinion.csv'))
