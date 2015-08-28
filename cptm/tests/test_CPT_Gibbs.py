from nose.tools import assert_equal, assert_true, assert_false

from cptm import generateCPTCorpus
from cptm import CPTCorpus
import shutil
from cptm.CPT_Gibbs import GibbsSampler
from pandas import DataFrame
from numpy.testing import assert_almost_equal
from numpy import inf
import os.path


def setup():
    global data_dir
    global out_dir
    global persp_dirs
    global documents
    global corpus
    global sampler

    data_dir = 'test_data/'
    out_dir = 'test_output/'
    persp_dirs = ['{}{}'.format(data_dir, p) for p in ('p0', 'p1')]
    documents = generateCPTCorpus.generate_cpt_corpus(data_dir)
    corpus = CPTCorpus(persp_dirs, topicLines=[0], opinionLines=[1])
    sampler = GibbsSampler(corpus, nTopics=3, nIter=5, out_dir=out_dir,
                           sample_interval=1)
    sampler._initialize()
    sampler.run()


def teardown():
    shutil.rmtree(data_dir)
    shutil.rmtree(out_dir)


def test_get_parameter_dir_name():
    """Test existence of parameter directory"""
    dir_name = sampler.get_parameter_dir_name()

    assert_equal(os.path.exists(dir_name), True)


def test_get_phi_topic_file_name():
    """Test existence of phi topic parameter files"""
    for i in range(sampler.nIter):
        file_name = sampler.get_phi_topic_file_name(i)
        yield assert_equal, os.path.isfile(file_name), True


def test_get_theta_file_name():
    """Test existence of theta parameter files"""
    for i in range(sampler.nIter):
        file_name = sampler.get_theta_file_name(i)
        yield assert_equal, os.path.isfile(file_name), True


def test_get_phi_opinion_file_name():
    """Test existence of phi opinion parameter files"""
    for i in range(sampler.nIter):
        for p, persp in enumerate(sampler.corpus.perspectives):
            file_name = sampler.get_phi_opinion_file_name(p, i)
            yield assert_equal, os.path.isfile(file_name), True


def test_check_index():
    """Return index of nIter-1 for certain inputs of _check_index"""
    values = [None, sampler.nIter]
    for v in values:
        yield assert_equal, sampler._check_index(v), sampler.nIter-1


def test_check_start():
    """Test _check_start_and_end with values for start"""
    values = [(-1, 0), (sampler.nIter+5, 0),
              (sampler.nIter-3, sampler.nIter-3)]
    for input_v, output_v in values:
        start, end = sampler._check_start_and_end(input_v, sampler.nIter)
        yield assert_equal, start, output_v


def test_check_end():
    """Test _check_start_and_end with values for start"""
    values = [(-1, sampler.nIter), (sampler.nIter+5, sampler.nIter)]
    for input_v, output_v in values:
        start, end = sampler._check_start_and_end(0, input_v)
        yield assert_equal, end, output_v


def test_check_start_and_end():
    """Test _check_start_and_end with values for start and end"""
    values = [((None, None), (None, None)), ((3, 2), (0, 2))]
    for (start_in, end_in), (start_out, end_out) in values:
        start, end = sampler._check_start_and_end(start_in, end_in)
        yield assert_equal, start, start_out
        yield assert_equal, end, end_out


def test_get_phi_topic_from_memory():
    sampler2 = GibbsSampler(corpus, nTopics=3, nIter=5)
    sampler2._initialize()
    sampler2.run()

    f = sampler2.get_phi_topic_file_name(0)
    yield assert_false, os.path.isfile(f)

    phi = sampler2.get_phi_topic()
    yield assert_equal, phi.shape, (sampler2.nTopics, sampler2.VT)

    phi = sampler2.get_phi_topic(index=2)
    yield assert_equal, phi.shape, (sampler2.nTopics, sampler2.VT)

    phi = sampler2.get_phi_topic(start=0, end=5)
    yield assert_equal, phi.shape, (sampler2.nTopics, sampler2.VT)


def test_get_theta_from_memory():
    sampler2 = GibbsSampler(corpus, nTopics=3, nIter=5)
    sampler2._initialize()
    sampler2.run()

    f = sampler2.get_theta_file_name(0)
    yield assert_false, os.path.isfile(f)

    r = sampler2.get_theta()
    yield assert_equal, r.shape, (sampler2.DT, sampler2.nTopics)

    r = sampler2.get_theta(index=2)
    yield assert_equal, r.shape, (sampler2.DT, sampler2.nTopics)

    r = sampler2.get_theta(start=0, end=5)
    yield assert_equal, r.shape, (sampler2.DT, sampler2.nTopics)


def test_get_phi_opinion_from_memory():
    sampler2 = GibbsSampler(corpus, nTopics=3, nIter=5)
    sampler2._initialize()
    sampler2.run()

    f = sampler2.get_phi_opinion_file_name(0, 0)
    yield assert_false, os.path.isfile(f)

    phi = sampler2.get_phi_opinion()
    yield assert_equal, len(phi), sampler2.nPerspectives
    yield assert_equal, phi[0].shape, (sampler2.nTopics, sampler2.VO)

    phi = sampler2.get_phi_opinion(index=2)
    yield assert_equal, len(phi), sampler2.nPerspectives
    yield assert_equal, phi[0].shape, (sampler2.nTopics, sampler2.VO)

    phi = sampler2.get_phi_opinion(start=0, end=5)
    yield assert_equal, len(phi), sampler2.nPerspectives
    yield assert_equal, phi[0].shape, (sampler2.nTopics, sampler2.VO)


def test_perplexity():
    """Minimal test of perplexity caluclation"""
    corpus2 = CPTCorpus(persp_dirs, testSplit=20, topicLines=[0],
                        opinionLines=[1])
    sampler2 = GibbsSampler(corpus2, nTopics=3, nIter=5)
    sampler2._initialize()
    sampler2.run()
    tw_perp, ow_perp = sampler2.perplexity()

    yield assert_true, tw_perp > 0.0
    yield assert_true, tw_perp < inf
    yield assert_true, ow_perp > 0.0
    yield assert_true, ow_perp < inf
