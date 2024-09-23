import yaml
import copy
import spacy
import random
import logging
import itertools
from typing import *
nlp = spacy.load('en_core_web_sm')
random.seed(2)

def setup_logger():
    logger = logging.Logger(__name__)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

def split_sentences(doc: str) -> List[str]:
    """Split the document into sentences.

    Args:
        doc (str): The document text

    Returns:
        List[str]: The list of sentences
    """
    return [text.text.strip() for segment in doc.split("\n") for text in nlp(segment).sents if len(text.text.strip()) > 0]

def get_configs():
    """Get a list of all combinations of hyperparameter configurations for the experiment.

    Returns:
        List[Dict[str, Any]]: The list of configurations.
    """
    cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    temp = {
        'dataset': cfg['datasets'],
        'hyperparams': []
    }
    chunkers = cfg['chunkers']
    for chunker in chunkers:
        if chunker in {'LangchainChunker', 'AbsoluteLangchainChunker'}:
            combinations = [{
                "breakpoint_threshold_type": breakpoint_threshold_type,
                "breakpoint_threshold_amount": breakpoint_threshold_amount
            } for breakpoint_threshold_type in cfg[chunker]['breakpoint_threshold_type'] for breakpoint_threshold_amount in cfg[chunker][breakpoint_threshold_type]]
        elif chunker in cfg:
            keys, values = zip(*cfg[chunker].items())
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        else:
            combinations = [{}]
        for combination in combinations:
            combination['chunker'] = chunker
        temp['hyperparams'] += combinations

    keys, values = zip(*temp.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, combination in enumerate(combinations):
        if combination['hyperparams']['chunker'] in {'SingleLinkageChunker', 'DBSCANChunker', 'LangchainChunker', 'AbsoluteLangchainChunker'}:
            combinations[i] = expand_dataset_name(combination)
    return combinations

def create_hyperparam_str(hyperparams):
    cache = copy.deepcopy(hyperparams)
    chunker_name = cache.pop("chunker")
    hyperparam_str = "|".join([chunker_name]+[f"{k}_{v}" for k, v in cache.items()])
    hyperparam_str = hyperparam_str.replace('/', '_').replace(' ', '_').replace('.', '-')
    return hyperparam_str

def expand_dataset_name(combination):
    d = copy.deepcopy(combination)
    d['hyperparams']['dataset'] = d['dataset']
    return d