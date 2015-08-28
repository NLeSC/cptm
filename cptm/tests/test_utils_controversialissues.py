from nose.tools import assert_equal
from numpy.testing import assert_almost_equal
from numpy import load
from pandas import DataFrame, read_csv

from cptm.utils.controversialissues import jsd_opinions, contrastive_opinions


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
    topics = read_csv('cptm/tests/data/topics_20.csv', index_col=0)
    opinions = [read_csv('cptm/tests/data/opinions_p0_20.csv', index_col=0),
                read_csv('cptm/tests/data/opinions_p1_20.csv', index_col=0)]
    nks = load('cptm/tests/data/nks_20.npy')
    co = contrastive_opinions('carrot', topics, opinions, nks)
    assert_equal(co.shape, (len(opinions[0].index), len(opinions)))


def test_contrastive_opinions_prob_distr():
    """Verify that the sum of all columns == 1.0 (probability distribution)"""
    topics = read_csv('cptm/tests/data/topics_20.csv', index_col=0)
    opinions = [read_csv('cptm/tests/data/opinions_p0_20.csv', index_col=0),
                read_csv('cptm/tests/data/opinions_p1_20.csv', index_col=0)]
    nks = load('cptm/tests/data/nks_20.npy')
    co = contrastive_opinions('carrot', topics, opinions, nks)

    s = co.sum(axis=0)

    for v in s:
        yield assert_almost_equal, v, 1.0
