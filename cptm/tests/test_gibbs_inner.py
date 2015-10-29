import numpy as np
from numpy import random, zeros
from numpy.testing import assert_array_almost_equal
from gibbs_inner import p_z, p_x


def setup():
    global VO
    global VT
    global D
    global nPerspectives
    global nTopics
    global beta

    global ndk
    global nkw
    global ntd
    global nrs
    global ns

    global p

    VO = 10
    VT = 10
    D = 10
    nPerspectives = 4
    nTopics = random.randint(50, 300)
    beta = random.random()

    ndk = random.randint(1000, size=(D, nTopics))
    nkw = random.randint(5000, size=(nTopics, VT))
    ntd = np.sum(nkw, axis=0)
    nrs = random.randint(5000, size=(nPerspectives, nTopics, VO))
    ns = random.randint(5000, size=(nPerspectives, nTopics))

    p = zeros(nTopics)


def p_z_reference(d, w_id, alpha, beta, nTopics, VT, ndk, nkw, nk):
    """Calculate (normalized) probabilities for p(w|z) (topics).

    The probabilities are normalized, because that makes it easier to
    sample from them.
    """
    f1 = (ndk[d]+alpha) / (np.sum(ndk[d])+nTopics*alpha)
    f2 = (nkw[:, w_id]+beta) / (nk+beta*VT)

    p = f1*f2
    return p / np.sum(p)


def p_x_reference(persp, d, w_id, beta_o, VO, nrs, ns, ndk, ntd):
        """Calculate (normalized) probabilities for p(w|x) (opinions).

        The probabilities are normalized, because that makes it easier to
        sample from them.
        """
        f1 = (nrs[persp, :, w_id]+beta_o) / (ns[persp]+beta_o*VO)
        # The paper says f2 = nsd (the number of times topic s occurs in
        # document d) / Ntd (the number of topic words in document d).
        # 's' is used to refer to opinions. However, f2 makes more sense as the
        # fraction of topic words assigned to a topic.
        # Also in test runs of the Gibbs sampler, the topics and opinions might
        # have different indexes when the number of opinion words per document
        # is used instead of the number of topic words.
        f2 = ndk[d]/(ntd[d]+0.0)

        p = f1*f2
        return p / np.sum(p)


def test_p_z():
    """Compare output from reference p_z to cython p_z"""
    alpha = random.random()

    nk = random.randint(5000, size=(nTopics))

    for w_id in range(VT):
        for d in range(D):
            pz1 = p_z_reference(d, w_id, alpha, beta, nTopics, VT, ndk, nkw, nk)
            pz2 = p_z(ndk[d], nkw[:, w_id], nk, alpha, beta, VT, p)

            yield almost_equal, pz1, pz2


def test_p_x():
    """Compare output from reference p_x to cython p_x"""
    for persp in range(nPerspectives):
        for w_id in range(VO):
            for d in range(D):
                p1 = p_x_reference(persp, d, w_id, beta, VO, nrs, ns, ndk, ntd)
                p2 = p_x(nrs[persp, :, w_id], ns[persp], ndk[d], ntd[d], beta, VO, p)

                yield almost_equal, p1, p2


def almost_equal(ar1, ar2):
    assert_array_almost_equal(ar1, ar2)
