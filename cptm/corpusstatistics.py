import logging
import argparse
import glob
import gzip
from lxml import etree
import datetime

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

date_tag = '{http://purl.org/dc/elements/1.1/}date'
min_date = datetime.date(2013, 12, 31)
max_date = datetime.date(2008, 01, 01)

y = '{}/*/'.format(args.data_dir)
years = glob.glob(y)
print '# of years,{}'.format(len(years))
for year in years:
    data_files = glob.glob('{}data_folia/*.xml.gz'.format(year))
    data_files.sort()
    print '{},{}'.format(year, len(data_files))
    for df in [data_files[0], data_files[-1]]:
        f = gzip.open(df)
        context = etree.iterparse(f, events=('end',), tag=date_tag,
                                  huge_tree=True)
        for event, elem in context:
            d = datetime.datetime.strptime(elem.text, "%Y-%m-%d").date()
            if d < min_date:
                min_date = d
            if d > max_date:
                max_date = d

print 'time period: ', min_date, max_date

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
