from nose.tools import assert_equal, assert_false

from os import remove
from os.path import join
from json import dump

from cptm.utils.experiment import load_config, add_parameter, thetaFileName, \
    topicFileName, opinionFileName, tarFileName, experimentName


def setup():
    global jsonFile
    global config
    global nTopics

    jsonFile = 'config.json'
    # create cofig.json
    params = {}
    with open(jsonFile, 'wb') as f:
        dump(params, f, sort_keys=True, indent=4)
    config = load_config(jsonFile)

    nTopics = 100


def teardown():
    remove(jsonFile)


def test_load_config_default_values():
    params = {}
    params['inputData'] = None
    params['outDir'] = '/{}'
    params['testSplit'] = 20
    params['minFreq'] = None
    params['removeTopTF'] = None
    params['removeTopDF'] = None
    params['nIter'] = 200
    params['beta'] = 0.02
    params['beta_o'] = 0.02
    params['expNumTopics'] = range(20, 201, 20)
    params['nTopics'] = None
    params['nProcesses'] = None
    params['topicLines'] = [0]
    params['opinionLines'] = [1]
    params['sampleEstimateStart'] = None
    params['sampleEstimateEnd'] = None

    for p, v in params.iteritems():
        yield assert_equal, v, config[p]


def test_add_parameter():
    pName = 'nTopics'

    yield assert_false, hasattr(config, pName)

    add_parameter(pName, nTopics, jsonFile)
    config2 = load_config(jsonFile)

    yield assert_equal, config2[pName], nTopics


def test_thetaFileName():
    config['nTopics'] = nTopics
    fName = thetaFileName(config)
    assert_equal(fName, '/theta_{}.csv'.format(nTopics))


def test_topicFileName():
    config['nTopics'] = nTopics
    fName = topicFileName(config)
    assert_equal(fName, '/topics_{}.csv'.format(nTopics))


def test_opinionFileName():
    config['nTopics'] = nTopics
    perspectives = ['p0', 'p1', 'p2']
    for p in perspectives:
        fName = opinionFileName(config, p)
        yield assert_equal, fName, '/opinions_{}_{}.csv'.format(p, nTopics)


def test_experimentName():
    config['outDir'] = '/tmp/test/{}'

    assert_equal('test', experimentName(config))


def test_tarFileName():
    config['outDir'] = '/{}'
    config['nTopics'] = nTopics
    name = experimentName(config)

    fName = tarFileName(config)

    assert_equal(fName, '/{}_{}.tgz'.format(name, nTopics))
