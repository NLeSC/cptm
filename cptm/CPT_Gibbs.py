"""Do CPT using Gibbs sampling.

Uses Gensim fucntionality.

Papers:
- Finding scientific topics
- A Theoretical and Practical Implementation Tutorial on Topic Modeling and
Gibbs Sampling
- Mining contrastive opinions on political texts using cross-perspective topic
model
"""

from __future__ import division
import numpy as np
import CPTCorpus
import glob
import logging
import time
import pandas as pd
import os

from cptm.gibbs_inner import gibbs_inner


logger = logging.getLogger(__name__)


class GibbsSampler():
    PARAMETER_DIR = '{}/parameter_samples'
    PHI_TOPIC = 'phi_topic'
    PHI_OPINION = 'phi_opinion_{}'
    THETA = 'theta'
    NKS = 'nks'

    def __init__(self, corpus, nTopics=10, alpha=0.02, beta=0.02, beta_o=0.02,
                 nIter=2, out_dir=None, sample_interval=10, initialize=True):
        self.corpus = corpus
        self.nTopics = nTopics
        self.alpha = alpha
        self.beta = beta
        self.nPerspectives = len(self.corpus.perspectives)
        self.beta_o = beta_o
        self.nIter = nIter
        self.sampleInterval = sample_interval

        self.VT = len(self.corpus.topicDictionary)
        self.VO = len(self.corpus.opinionDictionary)

        self.out_dir = out_dir
        if self.out_dir:
            if not os.path.exists(self.out_dir):
                os.makedirs(out_dir)
        logger.debug('output directory: {}'.format(self.out_dir))
        parameter_dir = self.get_parameter_dir_name()
        if not os.path.exists(parameter_dir):
            os.makedirs(parameter_dir)

        if initialize:
            self._initialize()

    def _initialize(self, phi_topic=None):
        """Initializes the Gibbs sampler."""
        logger.info('started initialization ({})'.format(str(self)))

        if not isinstance(phi_topic, np.ndarray):
            logger.debug('working with train set')
            self.documents = self.corpus
        else:
            logger.debug('working with test set')
            self.documents = self.corpus.testSet()

        self.DT = sum([len(p.corpus(phi_topic).topicCorpus)
                       for p in self.corpus.perspectives])
        self.DO = max([len(p.corpus(phi_topic).opinionCorpus)
                       for p in self.corpus.perspectives])
        self.maxDocLengthT = max([p.corpus(phi_topic).topicCorpus.maxDocLength
                                 for p in self.corpus.perspectives])
        self.maxDocLengthO = max([p.corpus(phi_topic).opinionCorpus.
                                  maxDocLength
                                  for p in self.corpus.perspectives])

        logger.debug('{} documents in the corpus (DT)'.format(self.DT))
        msg = '{} documents in the biggest perspective (DO)'.format(self.DO)
        logger.debug(msg)
        msg = 'maximum document lengths found: {} (topic) {} (opinion)'
        logger.debug(msg.format(self.maxDocLengthT, self.maxDocLengthO))

        # topics
        self.z = np.zeros((self.DT, self.maxDocLengthT), dtype=np.int)
        self.ndk = np.zeros((self.DT, self.nTopics), dtype=np.int)
        self.nkw = np.zeros((self.nTopics, self.VT), dtype=np.int)
        self.nk = np.zeros(self.nTopics, dtype=np.int)
        self.ntd = np.zeros(self.DT, dtype=np.int)

        # opinions
        self.x = np.zeros((self.nPerspectives, self.DO, self.maxDocLengthO),
                          dtype=np.int)
        self.nrs = np.zeros((self.nPerspectives, self.nTopics, self.VO),
                            dtype=np.int)
        self.ns = np.zeros((self.nPerspectives, self.nTopics), dtype=np.int)

        if phi_topic is not None:
            phi_topic_sum = np.sum(phi_topic, axis=0)

        logger.debug('Starting loop over documents/words')

        # loop over the words in the corpus
        for d, persp, d_p, doc in self.documents:
            for w_id, i in self.corpus.words_in_document(doc, 'topic'):
                #print d, persp, d_p, w_id, i
                if phi_topic is None:
                    topic = np.random.randint(0, self.nTopics)
                else:
                    p = phi_topic[:, w_id] / phi_topic_sum[w_id]
                    topic = self.sample_from(p)
                    #print len(phi_topic[:, w_id])
                    #print np.sum(phi_topic[:, w_id])
                #print topic
                self.z[d, i] = topic
                self.ndk[d, topic] += 1
                self.nkw[topic, w_id] += 1
                self.nk[topic] += 1
                self.ntd[d] += 1

            for w_id, i in self.corpus.words_in_document(doc, 'opinion'):
                opinion = np.random.randint(0, self.nTopics)
                self.x[persp][d_p, i] = opinion
                self.nrs[persp, opinion, w_id] += 1
                self.ns[persp, opinion] += 1
        logger.info('finished initialization ({})'.format(str(self)))

    def sample_from(self, p):
        """Sample (new) topic from multinomial distribution p.
        Returns a word's the topic index based on p_z.

        The searchsorted method is used instead of
        np.random.multinomial(1,p).argmax(), because despite normalizing the
        probabilities, sometimes the sum of the probabilities > 1.0, which
        causes the multinomial method to crash. This probably has to do with
        machine precision.
        """
        return np.searchsorted(np.cumsum(p), np.random.rand())

    def calc_theta_topic(self):
        """Calculate theta based on the current word/topic assignments.
        """
        f1 = self.ndk+self.alpha
        f2 = np.sum(self.ndk, axis=1, keepdims=True)+self.nTopics*self.alpha
        return f1/f2

    def calc_phi_topic(self):
        """Calculate phi based on the current word/topic assignments.
        """
        f1 = self.nkw+self.beta
        f2 = np.sum(self.nkw, axis=1, keepdims=True)+self.VT*self.beta
        return f1/f2

    def calc_phi_opinion(self, persp):
        """Calculate phi based on the current word/topic assignments.
        """
        f1 = self.nrs[persp]+float(self.beta_o)
        f2 = np.sum(self.nrs[persp], axis=1, keepdims=True)+self.VO*self.beta_o
        return f1/f2

    def run(self):
        logger.info('started sampling ({})'.format(str(self)))
        if not self.out_dir:
            # store all parameter samples in memory
            logger.debug('storing parameter samples in memory')
            self.theta_topic = np.zeros((self.nIter, self.DT, self.nTopics))
            self.phi_topic = np.zeros((self.nIter, self.nTopics, self.VT))

            self.phi_opinion = [np.zeros((self.nIter, self.nTopics, self.VO))
                                for p in self.corpus.perspectives]

        # variable to store nk for each iteration
        # nk is used for Contrastive Opinion Modeling
        self.nks = np.zeros((self.nIter, self.nTopics), dtype=np.int)

        for t in range(self.nIter):
            t1 = time.clock()
            logger.debug('iteration {} of {}'.format(t+1, self.nIter))

            gibbs_inner(self)

            # calculate theta and phi
            logger.debug('calculate theta and phi')
            if not self.out_dir:
                self.theta_topic[t] = self.calc_theta_topic()
                self.phi_topic[t] = self.calc_phi_topic()

                for p in range(self.nPerspectives):
                    self.phi_opinion[p][t] = self.calc_phi_opinion(p)
            else:
                if t == 0 or (t+1) % self.sampleInterval == 0:
                    fname = self.get_theta_file_name(t)
                    np.save(fname, self.calc_theta_topic())
                    logger.debug('writing to disk {}'.format(fname))

                    fname = self.get_phi_topic_file_name(t)
                    np.save(fname, self.calc_phi_topic())
                    logger.debug('writing to disk {}'.format(fname))

                    for p in range(self.nPerspectives):
                        fname = self.get_phi_opinion_file_name(p, t)
                        np.save(fname, self.calc_phi_opinion(p))
                        logger.debug('writing to disk {}'.format(fname))

                # save nk (for Contrastive Opinion Mining)
                self.nks[t] = np.copy(self.nk)
                np.save(self.get_nks_file_name(), self.nks)

            t2 = time.clock()
            logger.debug('time elapsed: {}'.format(t2-t1))

        logger.info('finished sampling ({})'.format(str(self)))

        self.estimate_parameters()

    def estimate_parameters(self, index=None, start=None, end=None):
        """Default: return single point estimate of the last iteration"""
        logger.debug('estimating parameters (index: {}, start: {}, end: {})'
                     .format(index, start, end))
        self.theta = self.get_theta(index, start, end)
        self.topics = self.get_phi_topic(index, start, end)
        self.opinions = self.get_phi_opinion(index, start, end)

    def get_theta(self, index=None, start=None, end=None):
        index = self._check_index(index)
        start, end = self._check_start_and_end(start=start, end=end)

        if hasattr(self, 'theta_topic'):
            logger.debug('retrieving theta from memory')
            if start and end:
                return np.mean(self.theta_topic[start:end], axis=0)
            return self.theta_topic[index]
        elif hasattr(self, 'out_dir'):
            logger.debug('retrieving theta from file')
            return self.load_parameters(self.THETA, index=index, start=start,
                                        end=end)
        else:
            # error: no parameter samples.
            # TODO: properly handle this case
            logger.error('no parameter samples found. Please run the Gibbs '
                         'sampler before trying to retrieve paramters.')

    def get_phi_topic(self, index=None, start=None, end=None):
        index = self._check_index(index)
        start, end = self._check_start_and_end(start=start, end=end)

        if hasattr(self, 'phi_topic'):
            logger.debug('retrieving phi topic from memory')
            if start and end:
                return np.mean(self.phi_topic[start:end], axis=0)
            return self.phi_topic[index]
        elif hasattr(self, 'out_dir'):
            logger.debug('retrieving phi topic from file')
            return self.load_parameters(self.PHI_TOPIC, index=index,
                                        start=start, end=end)
        else:
            # error: no parameter samples.
            # TODO: properly handle this case
            logger.error('no parameter samples found. Please run the Gibbs '
                         'sampler before trying to retrieve paramters.')

    def get_phi_opinion(self, index=None, start=None, end=None):
        index = self._check_index(index)
        start, end = self._check_start_and_end(start=start, end=end)

        phi_opinion = [np.zeros((self.nPerspectives, self.nTopics, self.VO))
                       for p in self.corpus.perspectives]

        if hasattr(self, 'phi_opinion'):
            logger.debug('retrieving phi opinion from memory')
            if start and end:
                for p in range(self.nPerspectives):
                    phi_opinion[p] = np.mean(self.phi_opinion[p][start:end],
                                             axis=0)
                return phi_opinion
            for p in range(self.nPerspectives):
                phi_opinion[p] = self.phi_opinion[p][index]
            return phi_opinion
        elif hasattr(self, 'out_dir'):
            logger.debug('retrieving phi opinion from from file')
            for p in range(self.nPerspectives):
                phi_opinion[p] = self.load_parameters(self.PHI_OPINION.
                                                      format(p),
                                                      index=index, start=start,
                                                      end=end)
            return phi_opinion
        else:
            # error: no parameter samples.
            # TODO: properly handle this case
            logger.error('no parameter samples found. Please run the Gibbs '
                         'sampler before trying to retrieve paramters.')

    def _check_index(self, index=None):
        if index is None:
            index = self.nIter-1
        if index == self.nIter:
            logger.warn('requested index {}; setting it to {}'.format(index,
                        self.nIter-1))
            index = self.nIter-1
        return index

    def _check_start_and_end(self, start=None, end=None):
        if start is None and end is None:
            return start, end
        if start < 0 or start > self.nIter or start > end:
            start = 0

        if end < 0 or end > self.nIter:
            end = self.nIter

        return start, end

    def perplexity(self, index=None, phi_topic=None, phi_opinion=None):
        """Calculate topic word and opinion word perplexity of the test set.

        Parameters:
            index : int (optional)
                index of the parameter sample of phi topic to use for a single
                point parameter estimate of phi topic (default nIter-1)
            phi_topic : ndarray (optional)
                phi_topic is used to initialize the Gibbs sampler to determine
                theta for the test set
            phi_opinion : ndarray (optional)
                phi_opinion provides estimates of p(o_i|z_i=k)

        Returns:
            topic word perplexity : float
            opinion word perplexity : float

        Topic word perplexity is calculated using importance sampling, as in
        [Griffiths and Steyvers, 2004]. However, according to
        [Wallach et al. 2009], this does not always result in an accurate
        estimate of the perplexity.

        Opinion word perplexity is calculated as described in [Fang et al.,
        2012] section 5.1.1.
        """
        # TODO: implement more accurate estimate of perplexity
        logger.info('calculating perplexity ({})'.format(str(self)))

        # load parameters
        if phi_topic is None:
            phi_topic = self.get_phi_topic(index)

        if phi_opinion is None:
            phi_opinion = self.get_phi_opinion(index)

        # run Gibbs sampler to find estimates for theta of the test set
        s = GibbsSampler(self.corpus, nTopics=self.nTopics, nIter=self.nIter,
                         alpha=self.alpha, beta=self.beta, beta_o=self.beta_o,
                         initialize=False)
        s._initialize(phi_topic=phi_topic)
        s.run()

        # topic word perplexity
        total_topic_words_in_test_documents = 0
        log_p_w = 0.0

        # opinion word perplexity
        total_opinion_words_in_test_documents = 0
        log_p_od = 0.0

        for d, persp, d_p, doc in self.corpus.testSet():
            for w_id, freq in doc['topic']:
                total_topic_words_in_test_documents += freq
                log_p_w += freq * np.log(np.sum(s.theta[d]*phi_topic[:, w_id]))

            for w_id, freq in doc['opinion']:
                total_opinion_words_in_test_documents += freq
                f1 = np.log(np.sum(s.theta[d]*phi_opinion[persp][:, w_id]))
                log_p_od += freq * f1

        tw_perp = np.exp(-log_p_w/total_topic_words_in_test_documents)
        ow_perp = np.exp(-log_p_od/total_opinion_words_in_test_documents)

        return tw_perp, ow_perp

    def opinion_word_perplexity_per_document(self, index=None, phi_topic=None,
                                             phi_opinion=None):
        """Calculate opinion word perplexity per document

        Opinion word perplexity is calculated per document in the
        test set.
        """
        logger.info('calculating opinion wor perplexity per document '
                    '({})'.format(str(self)))

        # load parameters
        if phi_topic is None:
            phi_topic = self.get_phi_topic(index)

        if phi_opinion is None:
            phi_opinion = self.get_phi_opinion(index)

        # run Gibbs sampler to find estimates for theta and phi_opinions
        # of the test set
        out_dir = os.path.join(self.out_dir, 'per_document_ow_perplexity')
        s = GibbsSampler(self.corpus, nTopics=self.nTopics, nIter=self.nIter,
                         alpha=self.alpha, beta=self.beta, beta_o=self.beta_o,
                         out_dir=out_dir, initialize=False)
        s._initialize(phi_topic=phi_topic)
        s.run()

        file_dict = s.corpus.get_files_in_train_and_test_sets()

        results = {}
        for d, persp, d_p, doc in self.corpus.testSet():
            opinion_words_in_document = 0
            log_p_od = 0.0

            for w_id, freq in doc['opinion']:
                opinion_words_in_document += freq
                f1 = np.log(np.sum(s.theta[d] * phi_opinion[persp][:, w_id]))
                log_p_od += freq * f1

            ow_perp = np.exp(-log_p_od / opinion_words_in_document)
            p_name = self.corpus.perspectives[persp].name

            if not results.get(p_name):
                results[p_name] = []
            results[p_name].append(ow_perp)

        df_index = [os.path.basename(f) for f in file_dict['0']['test']]

        # get theta
        documents = []
        for persp in self.corpus.perspectives:
            print persp
            documents += df_index
        print documents
        theta = s.theta_to_df(s.theta, documents)

        return pd.DataFrame(results, index=df_index), theta

    def load_parameters(self, name, index=None, start=None, end=None):
        index = self._check_index(index)
        start, end = self._check_start_and_end(start, end)

        if start and end:
            logger.debug('loading parameter files {} - {}'.
                         format(self.get_parameter_file_name(name, start),
                                self.get_parameter_file_name(name, end)))
            data = None
            for i in range(start, end):
                fName = self.get_parameter_file_name(name, i)
                if os.path.isfile(fName):
                    ar = np.load(fName)
                    if data is None:
                        data = np.array([ar])
                    else:
                        data = np.append(data, np.array([ar]), axis=0)
            return np.mean(data, axis=0)

        if not index is None:
            logger.debug('loading parameter file {}'.
                         format(self.get_parameter_file_name(name, index)))
            return np.load(self.get_parameter_file_name(name, index))

    def get_phi_topic_file_name(self, number):
        return self.get_parameter_file_name(self.PHI_TOPIC, number)

    def get_phi_opinion_file_name(self, persp, number):
        return self.get_parameter_file_name(self.PHI_OPINION.format(persp),
                                            number)

    def get_theta_file_name(self, number):
        return self.get_parameter_file_name(self.THETA, number)

    def get_nks_file_name(self):
        return '{}/{}.npy'.format(self.get_parameter_dir_name(), self.NKS)

    def get_parameter_file_name(self, name, number):
        return '{}/{}_{:04d}.npy'.format(self.get_parameter_dir_name(),
                                         name, number)

    def get_parameter_dir_name(self):
        return self.PARAMETER_DIR.format(self.out_dir)

    def print_topics_and_opinions(self, topics, opinions, top=10,
                                  threshold=0.0):
        """Print topics and associated opinions.

        Parameters:
            topics : pandas DataFrame
            opinions : list of pandas DataFrames

        The <top> top words and weights are printed.
        """
        for i in range(self.nTopics):
            print u'Topic #{}: {}'. \
                  format(i, self.print_topic(topics.loc[:, i].copy(), top=top,
                                             threshold=threshold))
            print
            for p in range(self.nPerspectives):
                print u'Opinion {}: {}'. \
                      format(self.corpus.perspectives[p].name,
                             self.print_topic(opinions[p].loc[:, i].copy(),
                                              top=top, threshold=threshold))
            print '-----'
            print

    def print_topic(self, series, top=10, threshold=0.0):
        """Prints the top 10 words in the topic/opinion on a single line."""
        s = series.copy()
        s.sort(ascending=False)
        t = [u'{} ({:.4f})'.format(word, p)
             for word, p in s[0:top].iteritems() if p > threshold]
        return u' - '.join(t)

    def topics_to_df(self, phi, words):
        """Returns a pandas DataFrame containing words x probabilities.

        The DataFrame is used as input for print_topics_and_opinions() and
        print_topic().

        Parameters:
            phi : numpy array
                Array containing the word probabilities for the topics.
            words : list of strings
                One of CPTCorpus.topic_words() and CPTCorpus.opinion_words().

        Returns:
            pandas DataFrame
                containing words x probabilities
        """
        logger.debug('creating dataframe with topics')
        df = pd.DataFrame(phi, columns=words)
        df = df.transpose()
        return df

    def theta_to_df(self, theta, documents):
        """Returns a pandas DataFrame containing doc ids x topic probabilities.

        Parameters:
        theta : numpy array
            Array containing the word probabilities for the topics.
        documents : list of strings
            list of document ids.

        Returns:
        pandas DataFrame
            containing document ids x topic probabilities.
        """
        logger.debug('creating dataframe with theta')
        df = pd.DataFrame(theta, index=documents)
        return df

    def __str__(self):
        return 'CPT GibbsSampler: {} perspectives, {} topics,  {} ' \
               'iterations, alpha: {}, beta: {}, beta_o: {}'. \
               format(self.nPerspectives, self.nTopics, self.nIter,
                      self.alpha, self.beta, self.beta_o)

if __name__ == '__main__':
    #logger.setLevel(logging.DEBUG)
    #logger.setLevel(logging.INFO)

    #files = glob.glob('/home/jvdzwaan/data/tmp/dilipad/gov_opp/*')
    files = glob.glob('/home/jvdzwaan/data/tmp/test/*')
    #files = glob.glob('/home/jvdzwaan/data/dilipad/perspectives/*')
    #out_dir = '/home/jvdzwaan/data/tmp/generated/test/'
    out_dir = '/home/jvdzwaan/data/tmp/dilipad/test_perplexity/'

    corpus = CPTCorpus.CPTCorpus(files, testSplit=20)
    #corpus.filter_dictionaries(minFreq=5, removeTopTF=100, removeTopDF=100)
    #corpus.save_dictionaries(directory=out_dir)
    #corpus.save('{}corpus.json'.format(out_dir))
    #corpus = CPTCorpus.CPTCorpus.load('{}corpus.json'.format(out_dir),
    #                                  topicDict='{}/topicDict.dict'.format(out_dir),
    #                                  opinionDict='{}/opinionDict.dict'.format(out_dir))
    #sampler = GibbsSampler(corpus, nTopics=30, nIter=100, out_dir=out_dir)
    sampler = GibbsSampler(corpus, nTopics=3, nIter=10, out_dir=out_dir)
    #sampler = GibbsSampler(corpus, nTopics=100, nIter=2)
    sampler._initialize()
    sampler.run()

    print sampler

    #topics_df = sampler.topics_to_df(sampler.get_phi_topic(),
    #                                 sampler.corpus.topic_words())
    #opinion_df = [sampler.topics_to_df(sampler.get_phi_opinion()[p],
    #                                   sampler.corpus.opinion_words())
    #              for p in range(sampler.nPerspectives)]
    #sampler.print_topics_and_opinions(topics_df, opinion_df, threshold=0.09)
    ps = []
    for i in range(0, 11):
        ps.append(sampler.perplexity(index=i))
    print ps
