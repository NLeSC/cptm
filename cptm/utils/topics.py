import pandas as pd


def get_top_topic_words(topics, opinions, t, top=10):
    """Return dataframe containing top topics and opinions.

    Parameters
        t : str - index of topic number
        top : int - the number of words to store in the dataframe

    Returns Pandas DataFrame
        The DataFrame contains top topic words, weights of topic words and for
        each perspective opinion words and weigths of opinion words.
    """
    t = str(t)
    topic = topics[t].copy()
    topic.sort(ascending=False)
    topic = topic[0:top]
    df_t = pd.DataFrame(topic)
    df_t.reset_index(level=0, inplace=True)
    df_t.columns = ['topic_{}'.format(t), 'weights_topic_{}'.format(t)]
    dfs = [df_t]

    for p, o in opinions.iteritems():
        opinion = o[t].copy()
        opinion.sort(ascending=False)
        opinion = opinion[0:top]
        df_o = pd.DataFrame(opinion)
        df_o.reset_index(level=0, inplace=True)
        df_o.columns = ['opinion_{}_{}'.format(t, p),
                        'weights_opinion_{}_{}'.format(t, p)]
        dfs.append(df_o)
    return pd.concat(dfs, axis=1)


def topic_str(df, single_line=False, weights=False):
    return str(df)
