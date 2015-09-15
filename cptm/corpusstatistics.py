import logging
import argparse
import glob

from cptm.utils.experiment import load_config, get_corpus

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

logging.getLogger('gensim').setLevel(logging.ERROR)
logging.getLogger('CPTCorpus').setLevel(logging.ERROR)
logging.getLogger('CPT_Gibbs').setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help='directory containing the raw data.')
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
args = parser.parse_args()

print 'Folia files'

y = '{}/*/'.format(args.data_dir)
years = glob.glob(y)
print '# of years,{}'.format(len(years))
for year in years:
    data_files = glob.glob('{}data_folia/*.xml.gz'.format(year))
    print '{},{}'.format(year, len(data_files))

config = load_config(args.json)
input_dir = config.get('inputData')

print '\ntext files'

perspectives = glob.glob(input_dir)
print '# of perspectives,{}'.format(len(perspectives))
total = 0
for persp in perspectives:
    data_files = glob.glob('{}/*.txt'.format(persp))
    total += len(data_files)
    print '{},{}'.format(persp, len(data_files))
print 'total,{}'.format(total)
