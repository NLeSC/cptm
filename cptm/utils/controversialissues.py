import numpy as np
import pandas as pd
from scipy.stats import entropy
import logging

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
