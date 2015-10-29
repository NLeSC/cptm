from nose.tools import assert_equal

from cptm.utils.dutchdata import get_coalition_parties


def test_get_coalition_parties_empty():
    assert_equal(get_coalition_parties(), [])


def test_get_coalition_parties_name():
    assert_equal(get_coalition_parties(cabinet='Balkenende I'),
                 ['CDA', 'LPF', 'VVD'])


def test_get_coalition_parties_date():
    assert_equal(get_coalition_parties(date='2007-10-22'),
                 ['CDA', 'PvdA', 'ChristenUnie'])
