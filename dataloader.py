from typing import *
from utils import *
from datasets import load_dataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader

###### dataloaders ######
def load_beir(dataset: str, split: str = "test"):
    """Load a BEIR dataset by name.

    Args:
        dataset (str): The name of the dataset to load.

    Returns:
        Tuple[
            Dict[str, Dict[str, str]],
            Dict[str, str],
            Dict[str, Dict[str, float]]
        ]: The corpus, queries, and qrels.
    """
    # load BEIR dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    # get a subset because the original data is too large
    corpus_subset, queries_subset, qrels_subset = get_subset(corpus, queries, qrels)

    # augment the corpus by concatenating multiple short documents into a single long document
    corpus_augmented = augment_corpus(corpus_subset)

    # modify the qrels to match the long documents in the augmented corpus
    qrels_modified = modify_qrels(qrels_subset, corpus_augmented)

    return corpus_augmented, queries_subset, qrels_modified


def load_mldr(lang: str = 'en', split: str = 'test'):
    """Load the MLDR dataset, a multilingual dataset for Wikipedia document retrieval.

    Huggingface: https://huggingface.co/datasets/Shitao/MLDR
    Paper: https://arxiv.org/pdf/2402.03216

    Args:
        lang (str): The language of the dataset to load.
        split (str): The split of the dataset to load.

    Returns:
        Tuple[
            Dict[str, Dict[str, str]],
            Dict[str, str],
            Dict[str, Dict[str, float]]
        ]: The corpus, queries, and qrels.
    """
    # load MLDR dataset
    dataset = 'Shitao/MLDR'
    queries = load_dataset(dataset, lang, split=split, trust_remote_code=True)
    corpus = load_dataset(dataset, f'corpus-{lang}', split='corpus', trust_remote_code=True)

    # convert the dataset to the BEIR format
    corpus_beir = {doc['docid']: {'text': doc['text']} for doc in corpus}
    queries_beir = {query['query_id']: query['query'] for query in queries}
    qrels_beir = {}
    for query_info in queries:
        qrels_beir[query_info['query_id']] = {}
        for positive_passage in query_info['positive_passages']:
            qrels_beir[query_info['query_id']][positive_passage['docid']] = 1
        for negative_passage in query_info['negative_passages']:
            qrels_beir[query_info['query_id']][negative_passage['docid']] = 0

    # get a subset because the original data is too large
    corpus_subset, queries_subset, qrels_subset = get_subset(corpus_beir, queries_beir, qrels_beir)

    return corpus_subset, queries_subset, qrels_subset



###### MAIN FUNCTION ######
def load_data(dataset: str):
    if dataset in ['nfcorpus','nq','hotpotqa','fiqa', 'webis-touche2020','dbpedia-entity','scidocs','fever','climate-fever','scifact']:
        return load_beir(dataset)
    elif dataset == 'mldr':
        return load_mldr()
    # elif dataset == 'YOUR_DATASET':
    #     return load_custom()
    else:
        raise ValueError(f"Dataset {dataset} not found.")