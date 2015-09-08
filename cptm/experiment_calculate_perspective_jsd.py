import logging
import argparse

from utils.experiment import load_config, get_corpus
from utils.controversialissues import perspective_jsd_matrix

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
args = parser.parse_args()

config = load_config(args.json)

corpus = get_corpus(config)
nTopics = config.get('nTopics')

perspectives = [p.name for p in corpus.perspectives]
perspective_jsd = perspective_jsd_matrix(config, nTopics, perspectives)

print perspective_jsd
print perspective_jsd.sum(axis=(2, 1))
