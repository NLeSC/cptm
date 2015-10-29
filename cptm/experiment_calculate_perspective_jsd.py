"""Calculate the pairwise jsd between perspectives for all topics

The results are saved in the outDir.

Usage: python experiment_calculate_perspective_jsd.py experiment.json
"""
import logging
import argparse
import numpy as np

from utils.experiment import load_config, load_opinions
from utils.controversialissues import perspective_jsd_matrix

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
args = parser.parse_args()

config = load_config(args.json)

opinions = load_opinions(config)
nTopics = config.get('nTopics')

perspective_jsd = perspective_jsd_matrix(opinions, nTopics)

print perspective_jsd
print perspective_jsd.sum(axis=(2, 1))

np.save(config.get('outDir').format('perspective_jsd_{}.npy'.format(nTopics)),
        perspective_jsd)
