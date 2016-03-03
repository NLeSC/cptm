cimport cython

cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def gibbs_inner(self):
    cdef np.ndarray[long, ndim=2, mode='c'] z = self.z
    cdef np.ndarray[long, ndim=2, mode='c'] ndk = self.ndk
    cdef np.ndarray[long, ndim=2, mode='c'] nkw = self.nkw
    cdef np.ndarray[long, ndim=1, mode='c'] nk = self.nk
    cdef np.ndarray[long, ndim=3, mode='c'] x = self.x
    cdef np.ndarray[long, ndim=2, mode='c'] ns = self.ns
    cdef np.ndarray[long, ndim=3, mode='c'] nrs = self.nrs
    cdef np.ndarray[long, ndim=1, mode='c'] ntd = self.ntd

    cdef Py_ssize_t d, w_id, i, persp, d_p
    cdef long topic, opinion, VT = self.VT, VO = self.VO

    cdef double alpha = self.alpha, beta = self.beta, beta_o = self.beta_o
    cdef np.ndarray[double, ndim=1, mode='c'] p
    p = np.empty(ndk.shape[1], dtype=np.double)

    for d, persp, d_p, doc in self.documents:
        for w_id, i in self.corpus.words_in_document(doc, 'topic'):
            topic = z[d, i]

            ndk[d, topic] -= 1
            nkw[topic, w_id] -= 1
            nk[topic] -= 1

            p = p_z(ndk[d], nkw[:, w_id], nk, alpha, beta, VT, p)
            topic = sample_from(p)

            z[d, i] = topic
            ndk[d, topic] += 1
            nkw[topic, w_id] += 1
            nk[topic] += 1

        for w_id, i in self.corpus.words_in_document(doc, 'opinion'):
            opinion = x[persp, d_p, i]

            nrs[persp, opinion, w_id] -= 1
            ns[persp, opinion] -= 1

            p = p_x(nrs[persp, :, w_id], ns[persp], ndk[d], ntd[d], beta_o, VO, p)
            opinion = sample_from(p)

            x[persp, d_p, i] = opinion
            nrs[persp, opinion, w_id] += 1
            ns[persp, opinion] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t sample_from(np.ndarray[double, ndim=1, mode='c'] p):
    cdef double thresh = np.random.rand()
    cdef double cumsum = 0.
    cdef Py_ssize_t i

    for i in range(p.shape[0]):
        cumsum += p[i]
        if cumsum > thresh:
            return i
    return p.shape[0] - 1


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef p_z(np.ndarray[long, ndim=1, mode='c'] ndk_d,
         np.ndarray[long, ndim=1] nkw_w_id,
         np.ndarray[long, ndim=1, mode='c'] nk,
         double alpha, double beta, long VT,
         np.ndarray[double, ndim=1, mode='c'] p):
    """Calculate (normalized) probabilities for p(w|z) (topics).

    The probabilities are normalized, because that makes it easier to
    sample from them.
    """
    #cdef np.ndarray[double, ndim=1, mode='c'] p
    
    # f1 = (ndk_d+alpha) / (np.sum(ndk_d) + nTopics*alpha)
    #p = np.empty(ndk_d.shape[0], dtype=np.double)
    cdef double total = 0
    for i in range(p.shape[0]):
        p[i] = ndk_d[i] + alpha
        total += ndk_d[i]
    for i in range(p.shape[0]):
        p[i] /= (total + p.shape[0] * alpha)

    # f2 = (nkw_w_id + beta) / (nk+beta*VT)
    total = 0
    for i in range(p.shape[0]):
        p[i] *= (nkw_w_id[i] + beta) / (nk[i] + beta * VT)
        total += p[i]
    # p = (f1*f2) / np.sum(f1*f2)
    for i in range(p.shape[0]):
        p[i] /= total

    return p


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef p_x(np.ndarray[long, ndim=1] nrs_d_wid,
          np.ndarray[long, ndim=1, mode='c'] ns_persp,
          np.ndarray[long, ndim=1, mode='c'] ndk_d,
          double ntd_d, double beta, long VO,
          np.ndarray[double, ndim=1, mode='c'] p):
    """Calculate (normalized) probabilities for p(w|x) (opinions).

    The probabilities are normalized, because that makes it easier to
    sample from them.
    """
    #cdef np.ndarray[double, ndim=1, mode='c'] p
    #p = np.empty(ndk_d.shape[0], dtype=np.double)

    # f1 = (nrs_d_wid+beta) / (ns_persp+beta*VO)
    # f2 = ndk_d/ntd_d
    # The paper says f2 = nsd (the number of times topic s occurs in
    # document d) / Ntd (the number of topic words in document d).
    # 's' is used to refer to opinions. However, f2 makes more sense as the
    # fraction of topic words assigned to a topic.
    # Also in test runs of the Gibbs sampler, the topics and opinions might
    # have different indexes when the number of opinion words per document
    # is used instead of the number of topic words.
    # p = (f1*f2) / np.sum(f1*f2)
    cdef double total = 0
    for i in range(p.shape[0]):
        p[i] = (nrs_d_wid[i] + beta) / (ns_persp[i] + beta * VO) * (ndk_d[i]/ntd_d)
        total += p[i]

    for i in range(p.shape[0]):
        p[i] /= total

    return p
