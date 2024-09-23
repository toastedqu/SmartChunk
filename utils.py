import os
import yaml
import json
import copy
import spacy
import pickle
import random
import logging
import itertools
from typing import *
nlp = spacy.load('en_core_web_sm')
random.seed(2)

def setup_logger():
    """Set up the logger.

    Returns:
        logging.Logger: The logger object.
    """
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

    # get all combinations of hyperparameters
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

    # zip datasets and hyperparams into combinations
    keys, values = zip(*temp.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # expand dataset name for certain chunkers
    for i, combination in enumerate(combinations):
        if combination['hyperparams']['chunker'] in {'SingleLinkageChunker', 'DBSCANChunker', 'LangchainChunker', 'AbsoluteLangchainChunker'}:
            combinations[i] = expand_dataset_name(combination)
    return combinations

def create_hyperparam_str(hyperparams):
    """Create a string representation of the hyperparameters.

    This is typically used for filenames and result visualization.
    The returned hyperparameter string follows the format:
        "chunker|key1_value1|key2_value2|..."

    Args:
        hyperparams (dict): The hyperparameters to create a string representation of.

    Returns:
        str: The string representation of the hyperparameters.
    """
    cache = copy.deepcopy(hyperparams)
    chunker_name = cache.pop("chunker")
    hyperparam_str = "|".join([chunker_name]+[f"{k}_{v}" for k, v in cache.items()])
    hyperparam_str = hyperparam_str.replace('/', '_').replace(' ', '_').replace('.', '-')   # convert decimal to hyphen for storage
    return hyperparam_str

def expand_dataset_name(combination):
    """Expand the dataset name in the hyperparameters.

    Args:
        combination (dict): The hyperparameter configuration.

    Returns:
        dict: The hyperparameter configuration with the dataset name expanded.
    """
    d = copy.deepcopy(combination)
    d['hyperparams']['dataset'] = d['dataset']
    return d

def read_pkl(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

def write_pkl(fname, d):
    with open(fname, "wb") as f:
        pickle.dump(d, f)

def read_pkls(fname):
    ds = []
    with open(fname, "rb") as f:
        while True:
            try:
                ds.append(pickle.load(f))
            except EOFError:
                break

def append_pkls(fname, d):
    with open(fname, "ab") as f:
        pickle.dump(d, f)

def read_json(fname):
    with open(fname, "r") as f:
        return json.load(f)

def write_json(fname, d):
    with open(fname, "w") as f:
        json.dump(d, f)

def read_jsonl(fname):
    ds = []
    with open(fname, 'r') as f:
        for line in f:
            ds.append(json.loads(line))
    return ds

def write_jsonl(fname, d):
    with open(fname, "a") as f:
        f.write(json.dumps(d) + '\n')

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)