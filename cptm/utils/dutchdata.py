"""Utils for the Dutch parliamentary data."""
import pandas as pd
import os
import datetime

known_parties = ['CDA', 'D66', 'GPV', 'GroenLinks', 'OSF', 'PvdA', 'RPF',
                 'SGP', 'SP', 'VVD', '50PLUS', 'AVP', 'ChristenUnie',
                 'Leefbaar Nederland', 'LPF', 'PvdD', 'PVV']


def pos_topic_words():
    return ['N']


def pos_opinion_words():
    return ['ADJ', 'BW', 'WW']


def word_types():
    return pos_topic_words() + pos_opinion_words()


def pos2lineNumber(pos):
    data = {'N': 0, 'ADJ': 1, 'BW': 2, 'WW': 3}
    return data[pos]


def read_coalitions():
    this_dir, this_filename = os.path.split(__file__)
    data_file = os.path.join(this_dir, '../../data/dutch_coalitions.csv')
    df = pd.read_csv(data_file, header=None,
                     names=['Date', 'Name', '1', '2', '3', '4'],
                     index_col=0, parse_dates=True)

    df.sort_index(inplace=True)
    return df


def get_coalition_parties(date=None, cabinet=None):
    coalitions = read_coalitions()
    if date is not None:
        coalitions = coalitions[['1', '2', '3', '4']]
        d = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        print d
        i = coalitions.index.searchsorted(d)
        print i
        c = coalitions.ix[coalitions.index[i-1]].tolist()
        return [p for p in c if str(p) != 'nan']
    if cabinet is not None:
        parties = coalitions[coalitions['Name'] == cabinet]
        parties = parties[['1', '2', '3', '4']].values[0].tolist()
        return [p for p in parties if not str(p) == 'nan']
    return []
