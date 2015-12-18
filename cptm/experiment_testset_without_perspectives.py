"""Script to extract a document/topic matrix for a set of text documents.

The corpus is not divided in perspectives.

Used to calculate theta for the CAP vragenuurtje data.
"""
import logging
import pandas as pd
import os

from CPTCorpus import CPTCorpus
from cptm.utils.experiment import get_sampler, thetaFileName

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

input_dir = ['/home/jvdzwaan/data/dilipad/CAP/vragenuurtje']
topicDict = '/home/jvdzwaan/data/dilipad/experiment/topicDict.dict'
opinionDict = '/home/jvdzwaan/data/dilipad/experiment/opinionDict.dict'
outDir = '/home/jvdzwaan/data/dilipad/CAP/vragenuurtje_results/{}'
nTopics = 60
phi_topic_file = '/home/jvdzwaan/data/dilipad/experiment/topics_60.csv'
start = 100
end = 199

phi_topic = pd.read_csv(phi_topic_file, index_col=0, encoding='utf-8').values.T
#print phi_topic.shape
#print phi_topic

corpus = CPTCorpus(input=input_dir, topicDict=topicDict,
                   opinionDict=opinionDict, testSplit=100, file_dict=None,
                   topicLines=[0], opinionLines=[1])
print str(corpus)

params = {
    'outDir': outDir,
    'nIter': 200,
    'beta': 0.02,
    'beta_o': 0.02,
    'nTopics': nTopics
}
sampler = get_sampler(params, corpus, nTopics=nTopics, initialize=False)
sampler._initialize(phi_topic=phi_topic)
sampler.run()
sampler.estimate_parameters(start=start, end=end)

logger.info('saving files')

documents = []
for persp in corpus.perspectives:
    print str(persp)
    for f in persp.testFiles:
        p, b = os.path.split(f)
        documents.append(b)
theta = sampler.theta_to_df(sampler.theta, documents)
theta.to_csv(thetaFileName(params), encoding='utf8')
