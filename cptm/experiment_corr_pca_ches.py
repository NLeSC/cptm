"""Calculate correlations between PCA projections and CHES rankings

The script:
1. calculates contrastive opinions for every topic word (this results in an
opinion for every party)
2. calculates rankings based on projections of the data on the first 10
principal components
3. compares the rankings to rankings based on ches data and calculates
correlations

Usage: python experiment_com_pca_ches.py experiment.json [-o /path/to/output]
"""
import logging
import pandas as pd
import argparse

from sklearn.decomposition import PCA
from scipy.stats import kendalltau, spearmanr

from cptm.utils.experiment import load_config, get_corpus, load_topics, \
    load_opinions, load_nks
from cptm.utils.controversialissues import contrastive_opinions, \
    jsd_opinions, filter_opinions


def do_kendallt(list1, list2, alpha=0.05):
    c, p = kendalltau(list1, list2)

    if p < alpha:
        return c
    return 'n.s.'


def do_spearmanr(list1, list2, alpha=0.05):
    c, p = spearmanr(list1, list2)

    if p < alpha:
        return c
    return 'n.s.'


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

    logging.getLogger('gensim').setLevel(logging.ERROR)
    logging.getLogger('CPTCorpus').setLevel(logging.ERROR)
    logging.getLogger('CPT_Gibbs').setLevel(logging.ERROR)
    logging.getLogger('utils.controversialissues').setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('json', help='json file containing experiment '
                        'configuration.')
    parser.add_argument('input', help='csv file containing ches data')
    parser.add_argument('--output', '-o', help='file to save output to')
    args = parser.parse_args()

    config = load_config(args.json)

    if args.output:
        fName = args.output
    else:
        fName = config.get('outDir').format('pca_correlation_words_{}.csv'.
                                            format(config.get('nTopics')))
    logger.info('writing output to {}'.format(fName))

    topics = load_topics(config)
    opinions = load_opinions(config)
    nks = load_nks(config)

    words = list(topics.index)

    ches_data = pd.read_csv(args.input, index_col=0)

    parties = opinions.keys()
    logger.debug('perspectives found: {}'.format(' - '.join(parties)))

    lrgen = ches_data.groupby('party').mean()['lrgen'].copy().loc[parties].sort_values()
    lrecon = ches_data.groupby('party').mean()['lrecon'].copy().loc[parties].sort_values()

    n_pca_components = 10

    r = {}
    column_names = []

    for idx, word in enumerate(words):
        co = contrastive_opinions(word, topics, opinions, nks)

        pca = PCA(n_pca_components)
        pca_results = pca.fit_transform(co.T.values)
        ranking = pd.DataFrame(pca_results, index=co.T.index, columns=['pca{}'.format(i) for i in range(n_pca_components)])

        #print ranking

        for i in range(n_pca_components):
            for m in ['lrgen', 'lrecon']:
                #print m
                if m == 'lrgen':
                    ches_r = lrgen
                else:
                    ches_r = lrecon

                if word not in r.keys():
                    r[word] = []

                # Kendall's tau
                pca_ranking = ranking['pca{}'.format(i)].copy().sort_values().index

                #print list(ches_r.index)
                #print list(pca_ranking)

                r[word].append(do_kendallt(list(ches_r.index), list(pca_ranking)))
                r[word].append(do_kendallt(list(ches_r[::-1].index), list(pca_ranking)))
                r[word].append(do_spearmanr(ches_r.values, ranking['pca{}'.format(i)].loc[ches_r.index].values))

                if idx == 0:
                    column_name_prefix = 'pca{}_{}'.format(i, m)
                    column_names.append('{}_{}'.format(column_name_prefix, 'kendalltau'))
                    column_names.append('{}_{}_{}'.format(column_name_prefix, 'kendalltau', 'R'))
                    column_names.append('{}_{}'.format(column_name_prefix, 'spearmanr'))

        if idx % 1000 == 0:
            logger.info('finished calculations for {} (word {} of {})'.format(word.encode('utf-8'), idx+1, len(words)))

            # write intermediary results
            results = pd.DataFrame(r, index=column_names)
            results = results.T
            results.to_csv(fName, encoding='utf-8')

    results = pd.DataFrame(r, index=column_names)
    results = results.T
    results.to_csv(fName, encoding='utf-8')
