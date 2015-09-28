import logging
import pandas as pd
import argparse
import ast

from cptm.utils.experiment import load_config, get_corpus, load_topics, \
    load_opinions, load_nks
from cptm.utils.controversialissues import contrastive_opinions, \
    jsd_opinions, filter_opinions

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

logging.getLogger('gensim').setLevel(logging.ERROR)
logging.getLogger('CPTCorpus').setLevel(logging.ERROR)
logging.getLogger('CPT_Gibbs').setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
parser.add_argument('--perspectives', '-p', help='list of perspectives to '
                    'calculate contrastive opinions for')
parser.add_argument('--output', '-o', help='file to save output to')
args = parser.parse_args()

print args.perspectives
print args.output

config = load_config(args.json)

if args.output:
    fName = args.output
else:
    fName = config.get('outDir').format('co_words_{}.csv'.
                                        format(config.get('nTopics')))
logger.info('writing output to {}'.format(fName))

corpus = get_corpus(config)

words = corpus.topic_words()
topics = load_topics(config)
opinions = load_opinions(config)
nks = load_nks(config)

if args.perspectives:
    perspectives = ast.literal_eval(args.perspectives)
    logger.info('filtering opinions to [{}]'.format(', '.join(perspectives)))
    opinions = filter_opinions(perspectives, opinions)

results = pd.DataFrame(index=words, columns=['jsd'])

for idx, word in enumerate(words):
    co = contrastive_opinions(word, topics, opinions, nks)
    jsd = jsd_opinions(co.values)
    results.set_value(word, 'jsd', jsd)

    if idx % 1000 == 0:
        logger.info('jsd for {}: {} (word {} of {})'.format(word, jsd, idx+1,
                                                            len(words)))

fName = 'co_words_{}.csv'.format(config.get('nTopics'))
results.to_csv(fName, encoding='utf-8')

print 'top 20 words with most contrastive opinions'
top = pd.Series(results['jsd'])
top.sort(ascending=False)
print top[0:20]
