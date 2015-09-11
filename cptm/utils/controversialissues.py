import numpy as np
import pandas as pd
from scipy.stats import entropy
import logging
from itertools import combinations
from cptm.utils.experiment import opinionFileName

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(time)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def contrastive_opinions(query, topics, opinions, nks):
    """Returns a DataFrame containing contrastive opinions for the query.

    Implements contrastive opinion modeling as specified in [Fang et al., 2012]
    equation 1. The resulting probability distributions over words are
    normalized, in order to facilitate mutual comparisons.

    Example usage:
        co = contrastive_opinions('mishandeling')
        print print_topic(co[0])

    Parameters:
        query : str
            The word contrastive opinions should be calculated for.
        topics : pandas DataFrame
            DataFrame containg the topics
        opinions : list of pandas DataFrames
            List containing a pandas DataFrame for every perspective
        nks : numpy ndarray
            numpy array containing nks counts

    Returns:
        pandas DataFrame
            The index of the DataFrame contains the topic words and the columns
            represent the perspectives.
    """
    # TODO: fix case when word not in topicDictionary
    logger.debug('calculating contrastive opinions')

    nPerspectives = len(opinions)
    opinion_words = list(opinions[0].index)

    result = pd.DataFrame(np.zeros((len(opinion_words), nPerspectives)),
                          opinion_words)

    for p, opinion in enumerate(opinions):
        print opinion.shape
        c_opinion = opinion * topics.loc[query] * nks[-1]
        c_opinion = np.sum(c_opinion, axis=1)
        c_opinion /= np.sum(c_opinion)

        result[p] = pd.Series(c_opinion, index=opinion_words)

    return result


def jsd_opinions(co):
    """Calculate Jensen-Shannon divergence between (contrastive) opinions.

    Implements Jensen-Shannon divergence between (contrastive) opinions as
    described in [Fang et al., 2012] section 3.2.

    Parameter:
        co : numpy ndarray
        A numpy ndarray containing (contrastive) opinions (see
        contrastive_opinions(query, topics, opinions, nks))

    Returns:
        float
        The Jensen-Shannon divergence between the contrastive opinions.
    """
    logger.debug('calculate Jensen-Shannon divergence between (contrastive) '
                 'opinions')

    nPerspectives = co.shape[1]

    result = np.zeros(nPerspectives, dtype=np.float)
    p_avg = np.mean(co, axis=1)
    for persp in range(nPerspectives):
        result[persp] = entropy(co[:, persp], p_avg)
    return np.mean(result)


def perspective_jsd_matrix(params, nTopics, perspectives):
    """Return the perspective jsd matrix.

    Returns:
        nTopics x #perspectives x #perspectives matrix containing pairwise
        jsd between perspectives for each topic. The #perspectives x
        #perspectives matrix for each topic is symmetric. The diagonal contains
        zeros.
    """
    logger.debug('calculate matrix containing pairwise JSD between '
                 'perspectives')
    nP = len(perspectives)
    perspective_jsd_matrix = np.zeros((nTopics, nP, nP), np.float)

    for persp1, persp2 in combinations(perspectives, 2):
        opinions1 = pd.read_csv(opinionFileName(params, persp1))
        opinions2 = pd.read_csv(opinionFileName(params, persp2))

        for t in range(nTopics):
            co = np.column_stack((opinions1[str(t)].values,
                                  opinions2[str(t)].values))
            index1 = perspectives.index(persp1)
            index2 = perspectives.index(persp2)
            jsd = jsd_opinions(co)
            perspective_jsd_matrix[t, index1, index2] = jsd
            perspective_jsd_matrix[t, index2, index1] = jsd

    return perspective_jsd_matrix
