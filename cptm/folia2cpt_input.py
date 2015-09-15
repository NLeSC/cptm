import os
import gzip
from lxml import etree
import logging
import codecs
from fuzzywuzzy import process, fuzz
import argparse
import glob
import datetime
import pandas as pd
from multiprocessing import Pool
from utils.dutchdata import pos_topic_words, pos_opinion_words, word_types, \
    known_parties
from utils.inputgeneration import Perspective


NUMBER = 100


def extract_words(data_file, nFile, nFiles, coalitions, cabinets):
    logger.debug('{} ({} of {})'.format(data_file, nFile, nFiles))
    if nFile % NUMBER == 0:
        logger.info('{} ({} of {})'.format(data_file, nFile, nFiles))

    word_tag = '{http://ilk.uvt.nl/FoLiA}w'
    pos_tag = '{http://ilk.uvt.nl/FoLiA}pos'
    lemma_tag = '{http://ilk.uvt.nl/FoLiA}lemma'
    speech_tag = '{http://www.politicalmashup.nl}speech'
    party_tag = '{http://www.politicalmashup.nl}party'
    date_tag = '{http://purl.org/dc/elements/1.1/}date'

    f = gzip.open(data_file)
    context = etree.iterparse(f, events=('end',), tag=(speech_tag, date_tag),
                              huge_tree=True)

    tWords = pos_topic_words()
    oWords = pos_opinion_words()

    # parties
    data = {}
    for party in known_parties:
        data[party] = Perspective(party, tWords, oWords)
    num_speech = 0
    num_speech_without_party = 0

    # Government vs. opposition
    go_data = {}
    go_data['g'] = Perspective('Government', tWords, oWords)
    go_data['o'] = Perspective('Opposition', tWords, oWords)

    # Cabinets
    # And government vs. opposition divided into cabinets
    # And parties divided into cabinets
    ca_data = {}
    ca_go_data = {}
    ca_parties_data = {}
    for ca in cabinets.tolist():
        ca_data[ca] = Perspective(ca, tWords, oWords)
        ca_go_data[ca] = {}
        ca_go_data[ca]['g'] = Perspective('{}-Government'.format(ca), tWords,
                                          oWords)
        ca_go_data[ca]['o'] = Perspective('{}-Opposition'.format(ca), tWords,
                                          oWords)
        ca_parties_data[ca] = {}
        for party in known_parties:
            ca_parties_data[ca][party] = Perspective('{}-{}'.format(ca, party),
                                                     tWords, oWords)

    for event, elem in context:
        if elem.tag == date_tag:
            d = datetime.datetime.strptime(elem.text, "%Y-%m-%d").date()
            i = coalitions.index.searchsorted(d)
            c = coalitions.ix[coalitions.index[i-1]].tolist()
            coalition_parties = [p for p in c if str(p) != 'nan']
            ca = cabinets[i-1]
        if elem.tag == speech_tag:
            num_speech += 1
            party = elem.attrib.get(party_tag)
            if party:
                # prevent unwanted subdirectories to be created (happens
                # when there is a / in the party name)
                party = party.replace('/', '-')

                if not data.get(party):
                    p, score1 = process.extractOne(party, known_parties)
                    score2 = fuzz.ratio(party, p)
                    logger.debug('Found match for "{}" to known party "{}" '
                                 '(scores: {}, {})'.format(party, p, score1,
                                                           score2))
                    if score1 >= 90 and score2 >= 90:
                        # change party to known party
                        logger.debug('Change party "{}" to known party "{}"'.
                                     format(party, p))
                        party = p
                        if not data.get(party):
                            data[party] = Perspective(party)
                    else:
                        # add new Perspective
                        logger.debug('Add new perspective for party "{}"'.
                                     format(party))
                        data[party] = Perspective(party)

                if party in coalition_parties:
                    go_perspective = 'g'
                else:
                    go_perspective = 'o'
                logger.debug('date: {}, party: {}, government or opposition: '
                             '{}, cabinet: {}'.format(str(d), party,
                                                      go_perspective, ca))

                # find all words
                word_elems = elem.findall('.//{}'.format(word_tag))
                for w in word_elems:
                    pos = w.find(pos_tag).attrib.get('class')
                    l = w.find(lemma_tag).attrib.get('class')
                    if pos in word_types():
                        data[party].add(pos, l)
                        go_data[go_perspective].add(pos, l)
                        ca_data[ca].add(pos, l)
                        ca_go_data[ca][go_perspective].add(pos, l)
                        ca_parties_data[ca][party].add(pos, l)
            else:
                num_speech_without_party += 1
    del context
    f.close()

    return data, go_data, ca_data, ca_go_data, ca_parties_data


def write_data(data, name, data_file):
    min_words = 1
    for p, persp in data.iteritems():
        len_topic_words, len_opinion_words = persp.word_lengths()
        if len_topic_words >= min_words and len_opinion_words >= min_words:
            fpath, fname = os.path.split(data_file)
            persp.write2file('{}/{}'.format(dir_out, name),
                             '{}.txt'.format(fname))


def process_file(data_file, nFile, nFiles, coalitions, cabinets):
    data, go_data, ca_data, ca_go_data, ca_parties_data = extract_words(
        data_file, nFile, nFiles, coalitions, cabinets)
    write_data(data, 'parties', data_file)
    write_data(go_data, 'gov_opp', data_file)
    write_data(ca_data, 'cabinets', data_file)
    for c in ca_go_data:
        write_data(ca_go_data[c], 'cabinets-gov_opp', data_file)
    for c in ca_parties_data:
        write_data(ca_parties_data[c], 'cabinets-parties', data_file)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s : %(message)s',
                        level=logging.INFO)
    logger.setLevel(logging.DEBUG)
    logging.getLogger('inputgeneration').setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('dir_in', help='directory containing the data '
                        '(gzipped FoLiA XML files)')
    parser.add_argument('dir_out', help='the name of the dir where the '
                        'CPT corpus should be saved.')
    args = parser.parse_args()

    coalitions = pd.read_csv('data/dutch_coalitions.csv', header=None,
                             names=['Date', 'Name', '1', '2', '3', '4'],
                             index_col=0, parse_dates=True)
    coalitions.sort_index(inplace=True)

    cabinets = coalitions['Name']
    coalitions = coalitions[['1', '2', '3', '4']]

    dir_in = args.dir_in
    dir_out = args.dir_out

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    pool = Pool(1)
    data_files = glob.glob('{}/*/data_folia/*.xml.gz'.format(dir_in))
    results = [pool.apply_async(process_file, args=(data_file, i+1,
                                len(data_files), coalitions, cabinets))
               for i, data_file in enumerate(data_files)]
    pool.close()
    pool.join()
