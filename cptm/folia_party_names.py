"""Extract names of political parties from the Folia files

Usage: python folia_party_names.py <path to raw data files>
"""
import gzip
from lxml import etree
import logging
import argparse
import glob
from collections import Counter

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s : %(message)s',
                        level=logging.INFO)
    logger.setLevel(logging.DEBUG)
    logging.getLogger('inputgeneration').setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('dir_in', help='directory containing the data '
                        '(gzipped FoLiA XML files)')
    args = parser.parse_args()

    data_files = glob.glob('{}/*/data_folia/*.xml.gz'.format(args.dir_in))

    num_speech = 0
    num_speech_without_party = 0
    parties = Counter()
    for data_file in data_files:

        word_tag = '{http://ilk.uvt.nl/FoLiA}w'
        pos_tag = '{http://ilk.uvt.nl/FoLiA}pos'
        lemma_tag = '{http://ilk.uvt.nl/FoLiA}lemma'
        speech_tag = '{http://www.politicalmashup.nl}speech'
        party_tag = '{http://www.politicalmashup.nl}party'
        date_tag = '{http://purl.org/dc/elements/1.1/}date'

        f = gzip.open(data_file)
        context = etree.iterparse(f, events=('end',), tag=(speech_tag, date_tag),
                                  huge_tree=True)
        for event, elem in context:
            if elem.tag == date_tag:
                pass
            if elem.tag == speech_tag:
                num_speech += 1
                party = elem.attrib.get(party_tag)
                if party:
                    # prevent unwanted subdirectories to be created (happens
                    # when there is a / in the party name)
                    party = party.replace('/', '-')

                    parties[party] += 1
                else:
                    num_speech_without_party += 1
        del context
        f.close()

    print 'num speech,', num_speech
    print 'num speech without party,', num_speech_without_party
    for p, f in parties.most_common():
        print p, f
    speeches_found = sum(parties.values())
    print 'speeches found,', speeches_found
    print 'speeches without parties + speeches_found,', num_speech_without_party+speeches_found
