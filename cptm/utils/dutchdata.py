"""Utils for the Dutch parliamentary data."""
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
