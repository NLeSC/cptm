#!/usr/bin/python
# -*- coding: utf-8 -*-
from nose.tools import assert_equal

from cptm.utils.inputgeneration import remove_trailing_digits


def test_remove_training_digits():
    cases = {u'd66': u'd',
             u'f16': u'f',
             u'é33': u'é'}

    for i, o in cases.iteritems():
        r = remove_trailing_digits(i)
        yield assert_equal, r, o
