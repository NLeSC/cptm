"""Remove saved parameter samples for certain iterations

Before, the Gibbs sampler saved estimates for all iterations. However, because
this took to much disk space, now the sampler only saves every tenth estimate.
This script removes samples for results generated with an old version of the
sampler.

Usage: python experiment_prune_samples.py /path/to/experiment.json
"""
import logging
import argparse
from os import remove

from cptm.utils.experiment import load_config, get_corpus, get_sampler


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
nIter = config.get('nIter')
outDir = config.get('outDir')
sampleInterval = 10

for nt in nTopics:
    sampler = get_sampler(config, corpus, nTopics=nt, initialize=False)
    logging.info('removing parameter sample files for nTopics = {}'.format(nt))
    for t in range(sampler.nIter):
        if t != 0 and (t+1) % sampleInterval != 0:
            try:
                remove(sampler.get_theta_file_name(t))
            except:
                pass

            try:
                remove(sampler.get_phi_topic_file_name(t))
            except:
                pass

            for persp in range(sampler.nPerspectives):
                try:
                    remove(sampler.get_phi_opinion_file_name(persp, t))
                except:
                    pass
