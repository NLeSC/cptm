from nose.tools import assert_equal, assert_true
from numpy.testing import assert_almost_equal
from numpy import load, sum, zeros
from pandas import DataFrame
from itertools import combinations, chain
from numpy.random import rand

from cptm.utils.experiment import load_topics, load_opinions
from cptm.utils.controversialissues import jsd_opinions, \
    contrastive_opinions, perspective_jsd_matrix, filter_opinions, \
    average_pairwise_jsd, jsd_for_all_topics


def test_jensen_shannon_divergence_self():
    """Jensen-Shannon divergence of a vector and itself must be 0"""
    v = [0.2, 0.2, 0.2, 0.2, 0.2]
    df = DataFrame({'p0': v, 'p1': v})

    assert_equal(0.0, jsd_opinions(df.values))


def test_jensen_shannon_divergence_symmetric():
    """Jensen-Shannon divergence is symmetric"""
    v1 = [0.2, 0.2, 0.2, 0.2, 0.2]
    v2 = [0.2, 0.2, 0.2, 0.3, 0.1]
    df1 = DataFrame({'p0': v1, 'p1': v2})
    df2 = DataFrame({'p0': v2, 'p1': v1})

    assert_equal(jsd_opinions(df1.values),
                 jsd_opinions(df2.values))


def test_jensen_shannon_divergence_known_value():
    """Jensen-Shannon divergence of v1 and v2 == 0.01352883"""
    v1 = [0.2, 0.2, 0.2, 0.2, 0.2]
    v2 = [0.2, 0.2, 0.2, 0.3, 0.1]
    df1 = DataFrame({'p0': v1, 'p1': v2})

    assert_almost_equal(0.01352883, jsd_opinions(df1.values))


def test_contrastive_opinions_result_shape():
    """Verify the shape of the output of contrastive_opinions"""
    params = {
        "inputData": "/home/jvdzwaan/data/tmp/test/*",
        "outDir": "cptm/tests/data/{}",
        "nTopics": 20
    }
    topics = load_topics(params)
    opinions = load_opinions(params)
    nks = load('cptm/tests/data/nks_20.npy')
    co = contrastive_opinions('carrot', topics, opinions, nks)
    num_opinion_words = len(opinions[opinions.keys()[0]].index)
    assert_equal(co.shape, (num_opinion_words, len(opinions)))


def test_contrastive_opinions_prob_distr():
    """Verify that the sum of all columns == 1.0 (probability distribution)"""
    params = {
        "inputData": "/home/jvdzwaan/data/tmp/test/*",
        "outDir": "cptm/tests/data/{}",
        "nTopics": 20
    }
    topics = load_topics(params)
    opinions = load_opinions(params)
    nks = load('cptm/tests/data/nks_20.npy')
    co = contrastive_opinions('carrot', topics, opinions, nks)

    s = co.sum(axis=0)

    for v in s:
        yield assert_almost_equal, v, 1.0


def test_perspective_jsd_matrix_symmetric():
    nTopics = 20
    params = {'nTopics': nTopics, 'outDir': 'cptm/tests/data/{}'}
    opinions = load_opinions(params)
    jsd_matrix = perspective_jsd_matrix(opinions, params.get('nTopics'))

    for i in range(nTopics):
        jsd = jsd_matrix[i]
        yield assert_true, (jsd.transpose() == jsd).all()


def test_perspective_jsd_matrix_diagonal_zeros():
    nTopics = 20
    params = {'nTopics': nTopics, 'outDir': 'cptm/tests/data/{}'}
    opinions = load_opinions(params)
    jsd_matrix = perspective_jsd_matrix(opinions, params.get('nTopics'))

    for i in range(nTopics):
        jsd = jsd_matrix[i]
        for idx in range(jsd.shape[0]):
            yield assert_equal, jsd[idx, idx], 0.0


def test_filter_opinions():
    params = {
        "inputData": "/home/jvdzwaan/data/tmp/test/*",
        "outDir": "cptm/tests/data/{}",
        "nTopics": 20
    }
    opinions = load_opinions(params)
    for perspectives in chain(combinations(opinions.keys(), 1),
                              [[], ['p0', 'p1']]):
        filtered = filter_opinions(perspectives, opinions)

        for p in perspectives:
            yield assert_true, p in filtered.keys()


def test_pairwise_jsd_equal_to_jsd_for_pairs_of_perspectives():
    nTopics = 4
    OT = 7
    cn = [str(t) for t in range(nTopics)]
    perspectives = ['p0', 'p1', 'p2']

    # generate random opinions
    opinions = {}
    for p in perspectives:
        o = rand(OT, nTopics)
        opinions[p] = DataFrame(o / sum(o, axis=0, keepdims=True), columns=cn)

    perspective_jsd = perspective_jsd_matrix(opinions, nTopics)
    ps = opinions.keys()

    for p1, p2 in combinations(ps, 2):
        op = filter_opinions([p1, p2], opinions)
        jsd = jsd_for_all_topics(op)

        idx1 = ps.index(p1)
        idx2 = ps.index(p2)

        for t in range(nTopics):
            yield assert_equal, jsd[t], perspective_jsd[t, idx1, idx2]


def test_value_of_avg_pw_jsd_equal_to_avg_jsd_of_pairs_of_perspectives():
    nTopics = 10
    OT = 7
    cn = [str(t) for t in range(nTopics)]
    perspectives = ['p0', 'p1', 'p2']

    # generate random opinions
    opinions = {}
    for p in perspectives:
        o = rand(OT, nTopics)
        opinions[p] = DataFrame(o / sum(o, axis=0, keepdims=True), columns=cn)

    perspective_jsd = perspective_jsd_matrix(opinions, nTopics)
    ps = opinions.keys()

    avg_pw_jsd = average_pairwise_jsd(perspective_jsd, opinions, ps)

    pairs = [(p1, p2) for p1, p2 in combinations(ps, 2)]
    jsd = zeros((len(pairs), nTopics))
    for index, (p1, p2) in enumerate(pairs):
        op = filter_opinions([p1, p2], opinions)
        jsd[index] = jsd_for_all_topics(op)

    for index, value in enumerate(jsd.mean(axis=0)):
        yield assert_equal, value, avg_pw_jsd[index]


# average pairwise jsd for all perspectives is the same as jsd as defined in
# Fang2012
# can we use this?
#    co_opinions = {}
#    for persp in opinions.keys():
#        df = pd.DataFrame(co[persp])
#        df.columns = ['0']
#        co_opinions[persp] = df
#    print co_opinions

#    perspective_jsd = perspective_jsd_matrix(co_opinions, 1)
#    pw_jsd = average_pairwise_jsd(perspective_jsd, co_opinions,
#                                  opinions.keys())
#
#
#    results.set_value(word, 'pw_jsd', pw_jsd)
