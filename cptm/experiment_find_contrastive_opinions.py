import logging
import pandas as pd
import argparse

from cptm.utils.experiment import load_config, get_corpus, get_sampler, \
    load_topics, load_opinions, load_nks
from cptm.utils.controversialissues import contrastive_opinions, jsd_opinions

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

sampler = get_sampler(config, corpus, nTopics=None, initialize=False)

words = corpus.topic_words()
topics = load_topics(config)
opinions = load_opinions(config)
nks = load_nks(config)

results = pd.DataFrame(index=words, columns=['jsd'])

for idx, word in enumerate(words):
    co = contrastive_opinions(word, topics, opinions, nks)
    jsd = jsd_opinions(co.values)
    results.set_value(word, 'jsd', jsd)

    if idx % 1000 == 0:
        logger.info('jsd for {}: {} (word {} of {})'.format(word, jsd, idx+1,
                                                            len(words)))

fName = 'co_words_{}.csv'.format(config.get('nTopics'))
results.to_csv(config.get('outDir').format(fName), encoding='utf-8')

print 'top 20 words with most contrastive opinions'
top = pd.Series(results['jsd'])
top.sort(ascending=False)
print top[0:20]
