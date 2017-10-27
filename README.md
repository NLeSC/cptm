[![Build Status](https://travis-ci.org/NLeSC/cptm.svg?branch=develop)](https://travis-ci.org/NLeSC/cptm.svg?branch=develop)

# Cross-Perspective Topic Modeling

A Gibbs sampler to do Cross-Perspective Topic Modeling, as described in

> Fang, Si, Somasundaram, & Yu (2012). Mining Contrastive Opinions on Political Texts using Cross-Perspective Topic Model. In proceedings of the fifth ACM international conference on Web Search and Data Mining. http://dl.acm.org/citation.cfm?id=2124306

## Installation

Install prerequisites.

    sudo apt-get install gfortran libopenblas-dev liblapack-dev

Clone the repository.

    git clone git@github.com:NLeSC/cptm.git
    cd cptm

Install the requirements (in virtual environment if desired).

    pip install -r requirements.txt

Install cptm (compiles Cython code).

    python setup.py install

Add the cptm directory to the `PYTHONPATH` (otherwise the scripts don't work).

    export PYTHONPATH=$PYTHONPATH:.

Tests can be run with `nosetests` (don't forget to `pip install nose` if you're using a virtual environment).

## Getting and preparing the Dutch parliamentary proceedings

Download the data.

    ./get_data.sh /path/to/xml/data/dir

Create sets of text documents for different perspectives.

    python folia2cpt_input.py /path/to/xml/data/dir /path/to/perspectives/dir

The script expects the directory structure generated by the `get_data.sh`
script. When the script finishes, there will be different directories in the
perspectives dir. Each directory is a division of the data using different
perspectives. The following perspectives are generated:

* `gov_opp`: Government vs. Opposition. The division is based on the data in
`data/dutch_coalitions.csv`
* `parties`: a perspective for each political party found in the data (noisy)
* `cabinets`: a perspective for each cabinet (based on the data
`data/dutch_coalitions.csv`)
* `cabinets-gov_opp`: a perspective for each cabinet divided by
Government/Opposition (based on the data in `data/dutch_coalitions.csv`)

## Running experiments

Experiments are configured using a json object. An example json file can be
copied from `data/config.json.example`

    cp data/config.json.example /path/experiment.json

The json object looks like:

    {
        "inputData": "/path/to/input/data/*",
        "outDir": "/path/to/output/directory/{}",
        "testSplit": 20,
        "nIter": 200,
        "beta": 0.02,
        "beta_o": 0.02,
        "expNumTopics": [20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
        "nProcesses": 3,
        "nTopics": 100,
        "topicLines": [0],
        "opinionLines": [1, 2, 3],
        "sampleEstimateStart": 80,
        "sampleEstimateEnd": 199,
        "minFreq": 5,
        "removeTopTF": 100,
        "removeTopDF": 100
    }

Options

* `inputData`: directory containing data separated by perspective (should end with \*),
e.g., `/path/to/perspectives/perspective/*` (where `perspective` is one of `gov_opp`,
`parties`, `cabinets`, `cabinets-gov_opp`)  
* `outDir`: directory where parameter estimates and other results will be saved (should
end with `{}`)  
* `testSplit`: percentage of the data used for calculating perplexity  
* `nIter`: number of sampling iterations  
* `beta`: beta parameter (topics)  
* `beta_o`: beta parameter (opinions)  
* `expNumTopics`: list of numbers of topics (e.g., `[20, 30]` means two experiments
will be run, one for # topics = 20 and one for # topics = 30
(script: `experiment_number_of_topics.py`))
* `nProcesses`: the number of processes the script can use (experiments will be run
in parallel if possible)  
* `nTopics`: the number of topics for which results will be calculated (scripts:
`experiment_calculate_perplexity.py`, `experiment_calculate_perspective_jsd.py`,
`experiment_find_contrastive_opinions.py`, `experiment_generate_results_nb.py`,
`experiment_get_results.py`, `experiment_jsd_opinions.py`, and
`experiment_number_of_topics.py`.  
* `topicLines`: line number(s) in input files containing topic words  
* `opinionLines`: line number(s) in input files containing opinion words  
* `sampleEstimateStart`: the relevant parameters are estimated from the samples that
are saved during each iteration. `sampleEstimateStart` is the iteration number where
to start estimating  
* `sampleEstimateEnd`: the relevant parameters are estimated from the samples that
are saved during each iteration. `sampleEstimateEnd` is the last iteration number
that is used to calculate results (<`nIter`).
* `minFreq`: minimal term frequency (terms occuring less frequently will be
removed from the vocabularies)  
* `removeTopTF`: the number of terms removed from the vocabularies based on term
frequency (terms are ordered by term frequency, next the top X is removed)  
* `removeTopDF`: the number of terms removed from the vocabularies based on
document frequency (terms are ordered by document frequency, next the top X is
removed)  

### Experiment scripts

First, run an experiment with different numbers of topics:

    python cptm/experiment_number_of_topics.py /path/to/experiment.json

Next, calculate opinion perplexity:

    python cptm/experiment_calculate_perplexity.py /path/to/experiment.json

To generate an iPython notebook to inspect the results of an experiment:

    python cptm/experiment_generate_results_nb.py   /path/to/dir/with/results/ experimentName /path/to/resulting/notebook.ipynb

The notebook helps to determine the 'optimal' number of topics for the data and
to choose appropriate `sampleEstimateStart` and `sampleEstimateEnd`. These
parameters are required to generate estimates of `theta`, `phi topics`, and
`phi opionions`.

Set the `nTopics`, `sampleEstimateStart`, and `sampleEstimateEnd` parameters in
the experiment configuration file. Next, generate esitmates of `theta`,
`phi topics`, and `phi opionions`:

    python cptm/experiment_get_results.py /path/to/experiment.json

Now you can go back to the iPython notebook to have a look at the topics and
opinions.

The notebook prints the top 5 topic words for all topics and the top 5 of
corresponding opinion words for each perspective. By default, the topics are
ordered by topic number. They can also be ordered by Jensen-Shannon divergence
of the opinions. That requires calculating the Jensen-Shannon divergences:

    python cptm/experiment_jsd_opinions.py /path/to/experiment.json

[Fang et al. 2012] describes contrastive opinion modeling, a method to
determine opinions for individual topic words. To do contrastive opinion
modeling for all topic words (and save the results on disk), run:

    python cptm/experiment_find_contrastive_opinions.py /path/to/experiment.json
    [-p <list of perspectives>] [-o /path/to/output]

The `<list of perspectives>` should be formatted like: `"['Kok II-ChristenUnie',
'Kok II-CDA', 'Kok II-LPF', 'Kok II-PvdA', 'Kok II-SGP', 'Kok II-D66',
'Kok II-GroenLinks', 'Kok II-VVD', 'Kok II-SP']"` (including the double quotes).

There are some additional scripts:

* `experiment_calculate_perspective_jsd.py`
calculates the pairwise average jsd between perspectives for all topics:

    python experiment_calculate_perspective_jsd.py experiment.json

* `experiment_prune_samples.py` removes saved parameter samples (generated by the
Gibbs sampler) for certain iterations. Before, the Gibbs sampler saved estimates
for all iterations. However, because this took to much disk space, now the
sampler only saves every tenth estimate. The `experiment_prune_samples` script
removes samples for results generated with an old version of the sampler:

    python experiment_prune_samples.py /path/to/experiment.json

* `experiment_manifesto.py` calculates opinion word perplexity per document for
a set of text documents. The corpus is not divided in perspectives. (This script
is used to estimate the likelihood of party manifestos given opinions for the
different perspectives (party manifestos come from the manifesto project)) First
run `manifestoproject2cptm_input.py` to create a cptm corpus that can be used
as input:

    python experiment_manifesto.py <experiment.json> \<input dir> \<output dir>

* `experiment_theta_for_texts_perspectives.py` extracts a document/topic
matrix for a set of text documents. The corpus is not divided in perspectives.
This script is used to calculate theta for the CAP vragenuurtje data. First
run `tabular2cptm_input.py` to create a cptm corpus that can be used
as input:

    python experiment_theta_for_texts_perspectives.py <experiment.json> \<input dir> \<output dir>

* `experiment_corr_pca_ches.py` calculate correlations between PCA projections
and CHES rankings:

    python experiment_corr_pca_ches.py <experiment.json> <inpt ches data> [-o /path/to/output]


* `experiment_cptcorpus_count_words.py` counts the number of topic and opinion
words in the corpus:

    python experiment_cptcorpus_count_words.py <experiment.json>


### Other scripts

* `corpusstatistics.py`

Prints some corpus statistics (such as the number of documents in the dataset).

    python corpusstatistics.py <path to raw data files> <experiment.json>

* `folia_party_names.py`

Extract names of political parties from the Folia files.

    python folia_party_names.py <path to raw data files>

* `generateCPTCorpus.py`

Script that generates a (synthetic) corpus to test the CPT model. This script is
used in the tests.

    Usage: python generateCPTCorpus.py <out dir>

* `manifestoproject2cptm_input.py`

Create input files in cptm format from manifesto project csv files

    python manifestoproject2cptm_input.py <input dir> <output dir>

The input dir should contain the manifesto project cvs files.

* `tabular2cptm_input.py`

Script that converts a field in a tabular data file to cptm input files.

    python tabular2cpt_input.py <csv of excel file> <full text field name>
<dir out>

## cptm functionality

### Saving CPTCorpus to disk

    from CPTCorpus import CPTCorpus

    corpus = CPTCorpus(files, testSplit=20)
    corpus.save('/path/to/corpus.json')

### Loading CPTCorpus from disk

    from CPTCorpus import CPTCorpus

    corpus2 = CPTCorpus.load('/path/to/corpus.json')

---
Copyright Netherlands eScience Center.

Distributed under the terms of the Apache2 license. See LICENSE for details.
