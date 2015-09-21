"""Functions for experiments."""
import logging
import json
from glob import glob
import os
import re
import pandas as pd
import numpy as np

from cptm import CPTCorpus
from cptm import GibbsSampler

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_config(fName):
    with open(fName) as f:
        config = json.load(f)

    logger.debug('configuration of experiment: ')
    params = ['{}: {}'.format(p, v) for p, v in config.iteritems()]
    for p in params:
        logger.debug(p)

    params = {}
    params['inputData'] = config.get('inputData')
    params['outDir'] = config.get('outDir', '/{}')
    params['testSplit'] = config.get('testSplit', 20)
    params['minFreq'] = config.get('minFreq')
    params['removeTopTF'] = config.get('removeTopTF')
    params['removeTopDF'] = config.get('removeTopDF')
    params['nIter'] = config.get('nIter', 200)
    params['beta'] = config.get('beta', 0.02)
    params['beta_o'] = config.get('beta_o', 0.02)
    params['expNumTopics'] = config.get('expNumTopics', range(20, 201, 20))
    params['nTopics'] = config.get('nTopics')
    params['nProcesses'] = config.get('nProcesses', None)
    params['topicLines'] = config.get('topicLines', [0])
    params['opinionLines'] = config.get('opinionLines', [1])
    params['sampleEstimateStart'] = config.get('sampleEstimateStart')
    params['sampleEstimateEnd'] = config.get('sampleEstimateEnd')

    return params


def add_parameter(name, value, fName):
    with open(fName) as f:
        config = json.load(f)
    config[name] = value
    with open(fName, 'w') as f:
        json.dump(config, f)


def get_corpus(params):
    out_dir = params.get('outDir')
    files = glob(params.get('inputData'))

    if not os.path.isfile(out_dir.format('corpus.json')):
        corpus = CPTCorpus(files,
                           testSplit=params.get('testSplit'),
                           topicLines=params.get('topicLines'),
                           opinionLines=params.get('opinionLines'))
        minFreq = params.get('minFreq')
        removeTopTF = params.get('removeTopTF')
        removeTopDF = params.get('removeTopDF')
        if (not minFreq is None) or (not removeTopTF is None) or \
           (not removeTopDF is None):
            corpus.filter_dictionaries(minFreq=minFreq,
                                       removeTopTF=removeTopTF,
                                       removeTopDF=removeTopDF)
        corpus.save_dictionaries(directory=out_dir.format(''))
        corpus.save(out_dir.format('corpus.json'))
    else:
        corpus = CPTCorpus.load(file_name=out_dir.format('corpus.json'),
                                topicLines=params.get('topicLines'),
                                opinionLines=params.get('opinionLines'),
                                topicDict=out_dir.format('topicDict.dict'),
                                opinionDict=out_dir.format('opinionDict.dict'))
    return corpus


def get_sampler(params, corpus, nTopics=None, initialize=True):
    if nTopics is None:
        nTopics = params.get('nTopics')
    out_dir = params.get('outDir')
    nIter = params.get('nIter')
    alpha = 50.0/nTopics
    beta = params.get('beta')
    beta_o = params.get('beta_o')
    logger.info('creating Gibbs sampler (nTopics: {}, nIter: {}, alpha: {}, '
                'beta: {}, beta_o: {})'.format(nTopics, nIter, alpha, beta,
                                               beta_o))
    sampler = GibbsSampler(corpus, nTopics=nTopics, nIter=nIter,
                           alpha=alpha, beta=beta, beta_o=beta_o,
                           out_dir=out_dir.format(nTopics),
                           initialize=initialize)
    return sampler


def load_topics(params):
    return pd.read_csv(topicFileName(params), index_col=0, encoding='utf-8')


def load_opinions(params):
    nTopics = params.get('nTopics')
    outDir = params.get('outDir')
    opinion_files = glob(outDir.format('/opinions_*_{}.csv'.format(nTopics)))
    print opinion_files
    opinions = {}
    for f in opinion_files:
        m = re.match(r'.+opinions_(.+).csv', f)
        name = m.group(1).replace('_{}'.format(nTopics), '')
        opinions[name] = pd.read_csv(f, index_col=0, encoding='utf-8')
    return opinions


def load_nks(params):
    return np.load(nksFileName(params))


def thetaFileName(params):
    nTopics = params.get('nTopics')
    return os.path.join(params.get('outDir').format(''),
                        'theta_{}.csv'.format(nTopics))


def topicFileName(params):
    nTopics = params.get('nTopics')
    return os.path.join(params.get('outDir').format(''),
                        'topics_{}.csv'.format(nTopics))


def opinionFileName(params, name):
    nTopics = params.get('nTopics')
    return os.path.join(params.get('outDir').format(''),
                        'opinions_{}_{}.csv'.format(name, nTopics))


def nksFileName(params):
    # TODO: fix code duplication in sampler
    nTopics = params.get('nTopics')
    outDir = params.get('outDir').format(nTopics)
    return os.path.join(outDir, 'parameter_samples/nks.npy')


def experimentName(params):
    fName = params.get('outDir')
    fName = fName.replace('/{}', '')
    _p, name = os.path.split(fName)
    return name


def tarFileName(params):
    nTopics = params.get('nTopics')
    name = experimentName(params)
    return os.path.join(params.get('outDir').format(''),
                        '{}_{}.tgz'.format(name, nTopics))
