"""Run Gibbs sampler for different # topics

Given a json file containing an experiment configuration a Gibbs sampler is
run for different numbers of topics.

Usage: python cptm/experiment_number_of_topics.py /path/to/experiment.json
"""
import logging
from multiprocessing import Pool
import argparse

from utils.experiment import load_config, get_corpus, get_sampler


def run_sampler(config, corpus, nTopics):
    sampler = get_sampler(config, corpus, nTopics)
    sampler.run()


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
args = parser.parse_args()

config = load_config(args.json)
corpus = get_corpus(config)

nTopics = config.get('expNumTopics')
logger.info('running Gibbs sampler for {} configurations'.format(len(nTopics)))

pool = Pool(processes=config.get('nProcesses'))
results = [pool.apply_async(run_sampler, args=(config, corpus, n))
           for n in nTopics]
pool.close()
pool.join()
