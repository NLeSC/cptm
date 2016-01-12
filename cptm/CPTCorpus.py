"""Class to access CPT corpus."""
import logging
from gensim import corpora
import glob
import codecs
from itertools import izip
from collections import Counter
import os
import random
import numpy as np
import json
import sys


logger = logging.getLogger(__name__)


class CPTCorpus():
    """Class to manage CPT corpus.

    Parameters:
        input : list of str
            A list containing names of directories representing the different
            perspectives. Every directory contains a text file for each
            document in the perspective. A text file contains the topic words
            on the first line and the opinion words on the second line. Words
            are separated by spaces.
        topicDict : str or gensim dictionary
        opinionDict : str or gensim dictionary
        testSplit : int
            Integer specifying the percentage of documents to be used as test
            set (for calculating perplexity).
        topicLines : list of ints
            list of ints specifying the line numbers in the data set containing
            topic words
        opinionLines : list of ints
            list of ints specifying the line numbers in the data set containing
            opinion words
    """
    TOPIC_DICT = 'topicDict.dict'
    OPINION_DICT = 'opinionDict.dict'

    def __init__(self, input=None, topicDict=None, opinionDict=None,
                 testSplit=None, file_dict=None, topicLines=[0],
                 opinionLines=[1]):
        if not file_dict is None:
            logger.info('initialize CPT Corpus with file_dict: {} perspectives'
                        .format(len(file_dict)))
            self.perspectives = [Perspective(file_dict=file_dict.get(str(p)),
                                             topicLines=topicLines,
                                             opinionLines=opinionLines)
                                 for p in range(len(file_dict))]
        else:
            logger.info('initialize CPT Corpus with {} perspectives'
                        .format(len(input)))
            input.sort()
            self.perspectives = [Perspective(input=glob.glob('{}/*.txt'.
                                             format(d)), testSplit=testSplit,
                                             topicLines=topicLines,
                                             opinionLines=opinionLines)
                                 for d in input]
            self.input = input

        if isinstance(topicDict, str) or isinstance(topicDict, unicode):
            self.load_dictionaries(topicDict=topicDict)
        elif isinstance(topicDict, corpora.Dictionary):
            self.topicDictionary = topicDict

        if isinstance(opinionDict, str) or isinstance(opinionDict, unicode):
            self.load_dictionaries(opinionDict=opinionDict)
        elif isinstance(opinionDict, corpora.Dictionary):
            self.opinionDictionary = opinionDict

        if not topicDict or not opinionDict:
            self._create_corpus_wide_dictionaries()

        self.testSplit = testSplit
        self.nPerspectives = len(self.perspectives)

    def __str__(self):
        perspectives = [str(p) for p in self.perspectives]
        return 'CPTCorpus with {} perspectives: {}'.format(
               len(self.perspectives), ', '.join(perspectives))

    def _create_corpus_wide_dictionaries(self):
        """Create dictionaries with all topic and opinion words.

        The created dictionaries contain mappings that can be used with across
        the corpora from different perspectives.
        """
        logger.info('creating corpus wide topic and opinion dictionaries')
        s = self.perspectives[0].trainSet
        self.topicDictionary = s.topicCorpus.dictionary
        self.opinionDictionary = s.opinionCorpus.dictionary
        for p in self.perspectives[1:]:
            s = p.trainSet
            self.topicDictionary.add_documents(s.topicCorpus.get_texts(),
                                               prune_at=None)
            self.opinionDictionary.add_documents(s.opinionCorpus.get_texts(),
                                                 prune_at=None)

    def words_in_document(self, doc, topic_or_opinion):
        """Iterator over the individual word positions in a document."""
        i = 0
        for w_id, freq in doc[topic_or_opinion]:
            for j in range(freq):
                yield w_id, i
                i += 1

    def doc_length(self, doc, topic_or_opinion):
        return sum([freq for w_id, freq in doc[topic_or_opinion]])

    def __iter__(self):
        """Iterator over the documents in the corpus."""
        return self._iterate([p.trainSet for p in self.perspectives])

    def _iterate(self, documentSets):
        doc_id_global = 0
        for i, p in enumerate(documentSets):
            doc_id_perspective = 0
            for doc in p:
                doc['topic'] = self.topicDictionary.doc2bow(doc['topic'])
                doc['opinion'] = self.opinionDictionary.doc2bow(doc['opinion'])

                yield doc_id_global, i, doc_id_perspective, doc

                doc_id_global += 1
                doc_id_perspective += 1

    def __len__(self):
        return sum([len(p) for p in self.perspectives])

    def testSet(self):
        return self._iterate([p.testSet for p in self.perspectives])

    def testSetLength(self):
        return sum([len(p.testSet) for p in self.perspectives])

    def calculate_tf_and_df(self):
        self.topic_tf = Counter()
        self.topic_df = Counter()

        self.opinion_tf = Counter()
        self.opinion_df = Counter()

        for doc_id_global, i, doc_id_perspective, doc in self:
            doc_words_topic = set()
            for w_id, freq in doc['topic']:
                self.topic_tf[w_id] += freq
                doc_words_topic.add(w_id)
            self.topic_df.update(doc_words_topic)

            doc_words_opinion = set()
            for w_id, freq in doc['opinion']:
                self.opinion_tf[w_id] += freq
                doc_words_opinion.add(w_id)
            self.opinion_df.update(doc_words_opinion)

    def filter_dictionaries(self, minFreq, removeTopTF, removeTopDF):
        logger.info('Filtering dictionaries')
        self.calculate_tf_and_df()
        self.filter_min_frequency(minFreq)
        self.filter_top_tf(removeTopTF)
        self.filter_top_df(removeTopDF)

        self.topicDictionary.compactify()
        self.opinionDictionary.compactify()
        logger.info('topic dictionary: {}'.format(self.topicDictionary))
        logger.info('opinion dictionary: {}'.format(self.opinionDictionary))

    def filter_min_frequency(self, minFreq=5):
        logger.info('Removing tokens from dictionaries with frequency < {}'.
                    format(minFreq))

        logger.debug('topic dict. before: {}'.format(self.topicDictionary))
        self._remove_from_dict_min_frequency(self.topicDictionary,
                                             self.topic_tf, minFreq)
        logger.debug('topic dict. after: {}'.format(self.topicDictionary))

        logger.debug('opinion dict. before: {}'.format(self.opinionDictionary))
        self._remove_from_dict_min_frequency(self.opinionDictionary,
                                             self.opinion_tf, minFreq)
        logger.debug('opinion dict. after: {}'.format(self.opinionDictionary))

    def _remove_from_dict_min_frequency(self, dictionary, tf, minFreq):
        remove_ids = []
        for w_id, freq in tf.iteritems():
            if freq < minFreq:
                remove_ids.append(w_id)
        logger.debug('removing {} tokens'.format(len(remove_ids)))
        dictionary.filter_tokens(bad_ids=remove_ids)

    def filter_top_tf(self, removeTop):
        logger.info('Removing {} most frequent tokens (top tf)'.
                    format(removeTop))

        logger.debug('topic dict. before: {}'.format(self.topicDictionary))
        self._remove_from_dict_top(self.topicDictionary, self.topic_tf,
                                   removeTop)
        logger.debug('topic dict. after: {}'.format(self.topicDictionary))

        logger.debug('opinion dict. before: {}'.format(self.opinionDictionary))
        self._remove_from_dict_top(self.opinionDictionary, self.opinion_tf,
                                   removeTop)
        logger.debug('opinion dict. after: {}'.format(self.opinionDictionary))

    def filter_top_df(self, removeTop):
        logger.info('Removing {} most frequent tokens (top df)'.
                    format(removeTop))

        logger.debug('topic dict. before: {}'.format(self.topicDictionary))
        self._remove_from_dict_top(self.topicDictionary, self.topic_df,
                                   removeTop)
        logger.debug('topic dict. after: {}'.format(self.topicDictionary))

        logger.debug('opinion dict. before: {}'.format(self.opinionDictionary))
        self._remove_from_dict_top(self.opinionDictionary, self.opinion_df,
                                   removeTop)
        logger.debug('opinion dict. after: {}'.format(self.opinionDictionary))

    def _remove_from_dict_top(self, dictionary, frequencies, top=100):
        remove_ids = []
        for w_id, freq in frequencies.most_common(top):
            remove_ids.append(w_id)
        dictionary.filter_tokens(bad_ids=remove_ids)
        logger.debug('removing {} tokens'.format(len(remove_ids)))

    def topic_words(self):
        """Return the list of topic words."""
        return self._create_word_list(self.topicDictionary)

    def opinion_words(self):
        """Return the list of opinion words."""
        return self._create_word_list(self.opinionDictionary)

    def _create_word_list(self, dictionary):
        """Return a list of all words in the dictionary.

        The word list is ordered by word id.
        """
        return [dictionary.get(i) for i in range(len(dictionary))]

    def save_dictionaries(self, directory=None):
        if directory:
            if not os.path.exists(directory):
                os.makedirs(directory)
        else:
            directory = ''

        self.topicDictionary.save(self.topic_dict_file_name(directory))
        self.opinionDictionary.save(self.opinion_dict_file_name(directory))

    def load_dictionaries(self, topicDict=None, opinionDict=None):
        if topicDict:
            self.topicDictionary = corpora.Dictionary.load(topicDict)
            logger.info('topic dictionary {}'.format(self.topicDictionary))
        if opinionDict:
            self.opinionDictionary = corpora.Dictionary.load(opinionDict)
            logger.info('opinion dictionary {}'.format(self.opinionDictionary))

    def topic_dict_file_name(self, directory=''):
        return os.path.join(directory, self.TOPIC_DICT)

    def opinion_dict_file_name(self, directory=''):
        return os.path.join(directory, self.OPINION_DICT)

    def get_files_in_train_and_test_sets(self):
        file_dict = {}
        for i, p in enumerate(self.perspectives):
            file_dict[str(i)] = {'train': p.trainFiles, 'test': p.testFiles}
        return file_dict

    def save(self, file_name):
        logger.info('saving corpus under {}'.format(file_name))
        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_dict = self.get_files_in_train_and_test_sets()
        with open(file_name, 'wb') as f:
            json.dump(file_dict, f, sort_keys=True, indent=4)

    @classmethod
    def load(self, file_name, topicLines, opinionLines, topicDict=None,
             opinionDict=None):
        logger.info('loading corpus from {}'.format(file_name))
        logger.debug('topicDict: {}'.format(topicDict))
        logger.debug('opinionDict: {}'.format(opinionDict))
        logger.debug('topicLines: {}'.format(topicLines))
        logger.debug('opinionLines: {}'.format(opinionLines))
        with open(file_name, 'rb') as f:
            file_dict = json.load(f)
        return self(file_dict=file_dict, topicDict=topicDict,
                    opinionDict=opinionDict, topicLines=topicLines,
                    opinionLines=opinionLines)


class Perspective():
    """Class representing a perspective in cross perspective topic modeling.
    This class contains two text corpora, one for the topic words and one for
    the opinion words. It is used by the class CTPCorpus.

    Parameters:
        input : list of strings
            List containing the file names of the documents in the corpus
            (.txt). A text file contains the topic words on the first line and
            opinion words on the second line.
    """
    def __init__(self, input=None, testSplit=None, file_dict=None,
                 topicLines=[0], opinionLines=[1]):
        if not file_dict is None:
            self.testFiles = file_dict.get('test', [])
            self.testSet = Corpus(self.testFiles, topicLines=topicLines,
                                  opinionLines=opinionLines)

            self.trainFiles = file_dict.get('train')
            self.trainSet = Corpus(self.trainFiles, topicLines=topicLines,
                                   opinionLines=opinionLines)

            if len(self.trainSet.input) > 0:
                self.name = self.persp_name(self.trainSet.input[0])
            elif len(self.testSet.input) > 0:
                self.name = self.persp_name(self.testSet.input[0])
            else:
                self.name = 'UNKNOWN'

            logger.info('initialize perspective "{}" from file_dict'
                        .format(self.name))
        else:
            if len(input) > 0:
                self.name = self.persp_name(input[0])
            else:
                self.name = 'UNKNOWN'
            logger.info('initialize perspective "{}" ({} documents)'
                        .format(self.name, len(input)))
            self.input = input[:]
            self.testFiles = []

            if testSplit and (testSplit < 1 or testSplit > 100):
                testSplit = None
                logger.warn('illegal value for testSplit ({}); ' +
                            'not creating test set'.format(testSplit))

            if testSplit:
                splitIndex = int(len(input)/100.0*testSplit)
                logger.info('saving {} of {} documents for testing'.
                            format(splitIndex, len(input)))
                random.shuffle(input)
                self.testFiles = input[:splitIndex]
                input = input[splitIndex:]
            self.testSet = Corpus(self.testFiles, topicLines=topicLines,
                                  opinionLines=opinionLines)

            self.trainFiles = input
            self.trainSet = Corpus(self.trainFiles, topicLines=topicLines,
                                   opinionLines=opinionLines)

    def persp_name(self, fName):
        if os.path.isfile(fName):
            p, f = os.path.split(fName)
        if os.path.isdir(fName):
            if fName.endswith('/'):
                # remove trailing /
                p = fName[:-1]
            else:
                p = fName
        _p, name = os.path.split(p)
        return name

    def __len__(self):
        return len(self.trainSet)

    def __str__(self):
        return '{} (train set: {} documents, test set: {} documents)'. \
               format(self.name, len(self.trainSet), len(self.testSet))

    def corpus(self, testSet=None):
        if isinstance(testSet, np.ndarray):
            return self.testSet
        return self.trainSet


class Corpus():
    """Wrapper representing a Corpus of a perspective (train set or test set).
    A Corpus consists of two partial corpora (PartialCorpus): one for topic
    words and one for opinion words. This class is used by the Perspective
    class.
    """
    def __init__(self, input, topicLines, opinionLines):
        self.input = input

        self.topicCorpus = PartialCorpus(input, lineNumbers=topicLines)
        self.opinionCorpus = PartialCorpus(input, lineNumbers=opinionLines)

    def __iter__(self):
        # topic_words and opinion_words are lists of actual words
        for topic_words, opinion_words in izip(self.topicCorpus.get_texts(),
                                               self.opinionCorpus.get_texts()):
            yield {'topic': topic_words, 'opinion': opinion_words}

    def __len__(self):
        return len(self.topicCorpus)


class PartialCorpus(corpora.TextCorpus):
    """Gensim TextCorpus containing either topic or opinion words.
    Used by the Corpus class.
    """
    def __init__(self, input, lineNumbers):
        self.lineNumbers = lineNumbers
        self.maxDocLength = 0
        self.minDocLength = sys.maxint
        input.sort()
        super(PartialCorpus, self).__init__(input)

        self.input = input

    def get_texts(self):
        for txt in self.input:
            with codecs.open(txt, 'rb', 'utf8') as f:
                lines = f.readlines()
                words = []
                for lineNumber in self.lineNumbers:
                    if len(lines) >= (lineNumber+1):
                        words = words + lines[lineNumber].split()

                # keep track of the maximum and minimum document length
                if len(words) > self.maxDocLength:
                    self.maxDocLength = len(words)
                if len(words) < self.minDocLength:
                        self.minDocLength = len(words)
            yield words

    def __len__(self):
        return len(self.input)


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    files = glob.glob('/home/jvdzwaan/data/tmp/test/p*')
    #files = glob.glob('/home/jvdzwaan/data/dilipad/perspectives/*')
    #files = glob.glob('/home/jvdzwaan/data/tmp/dilipad/gov_opp/*')
    files.sort()
    #print '\n'.join(files)
    out_dir = '/home/jvdzwaan/data/tmp/dilipad/test_parameters'

    corpus = CPTCorpus(files)
    #corpus.save_dictionaries(directory=out_dir)
    c = os.path.join(out_dir, 'corpus.json')
    corpus.save(os.path.join(out_dir, 'corpus.json'))
    print corpus.get_files_in_train_and_test_sets()
    corpus2 = CPTCorpus.load(c)
    print corpus2.get_files_in_train_and_test_sets()
    for p in corpus.perspectives:
        print p
    print corpus
    print corpus2
    #print corpus.topicDictionary
    #print corpus.opinionDictionary
    #print len(corpus.perspectives[0].opinionCorpus)
    #print len(corpus.perspectives[0].opinionTestCorpus)
    #for d in corpus.testSet():
    #    print d
    #corpus2 = CPTCorpus(files, topicDict=corpus.topic_dict_file_name(out_dir),
    #                    opinionDict=corpus.opinion_dict_file_name(out_dir))
    #print corpus2.topicDictionary
    #print corpus2.opinionDictionary
    #corpus.filter_dictionaries(minFreq=5, removeTopTF=100, removeTopDF=100)
    #d = '/home/jvdzwaan/data/dilipad/dictionaries'
    #corpus.save_dictionaries(directory=d)
    #corpus.save_dictionaries(None)
    #corpus.load_dictionaries(topic_dict=corpus.topic_dict_file_name(d),
    #                         opinion_dict=corpus.opinion_dict_file_name(d))
