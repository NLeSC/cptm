import logging
import argparse
import pandas as pd
import numpy as np

from utils.experiment import load_config, get_corpus, load_opinions
from utils.controversialissues import jsd_opinions

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
args = parser.parse_args()

config = load_config(args.json)
corpus = get_corpus(config)

nTopics = config.get('nTopics')

logger.info('loading opinions')
opinions = load_opinions(config)

logger.info('calculating jsd')
# combine opinions from different perspectives and calculate jsd
co = np.zeros((len(corpus.opinionDictionary), corpus.nPerspectives))
jsd = np.zeros(nTopics)
for t in range(nTopics):
    for i in range(len(opinions)):
        co[:, i] = opinions[i][str(t)].values
    jsd[t] = jsd_opinions(co)

fName = '{}/jsd_{}.csv'.format(config.get('outDir').format(''), nTopics)
logger.info('saving {} to disk'.format(fName))
df = pd.DataFrame({'jsd': jsd})
df.to_csv(fName)
