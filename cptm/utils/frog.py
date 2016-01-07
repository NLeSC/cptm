from pynlpl.clients.frogclient import FrogClient
import logging
import re
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


def get_frogclient(port=8020):
    try:
        frogclient = FrogClient('localhost', port)
        return frogclient
    except:
        logger.error('Cannot connect to the Frog server. '
                     'Is it running at port {}?'.format(port))
        logger.info('Start the Frog server with "docker run -p '
                    '127.0.0.1:{}:{} -t -i proycon/lamachine frog '
                    '-S {}"'.format(port, port, port))
        sys.exit(1)


def pos_and_lemmas(text, frogclient):
    regex = re.compile(r'\(.*\)')

    for data in frogclient.process(text):
        word, lemma, morph, ext_pos = data[:4]
        if ext_pos:  # ext_pos can be None
            pos = regex.sub('', ext_pos)
            yield pos, lemma
