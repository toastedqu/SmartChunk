import random
from typing import *
from datasets import load_dataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader
random.seed(2)

def load_beir(dataset: str):
    """Load a BEIR dataset by name.

    Args:
        dataset (str): The name of the dataset to load.
    """
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels

def load_mldr(lang: str = 'en', split: str = 'test'):
    """Load the MLDR dataset, a multilingual dataset for Wikipedia document retrieval.

    Huggingface: https://huggingface.co/datasets/Shitao/MLDR
    Paper: https://arxiv.org/pdf/2402.03216

    Args:
        lang (str): The language of the dataset to load.
        split (str): The split of the dataset to load.
    """
    dataset = 'Shitao/MLDR'
    queries = load_dataset(dataset, lang, split=split, trust_remote_code=True)
    corpus = load_dataset(dataset, f'corpus-{lang}', split='corpus', trust_remote_code=True)

    corpus_beir = {doc['docid']: {'title': '', 'text': doc['text']} for doc in corpus}
    queries_beir = {query['query_id']: query['query'] for query in queries}
    qrels_beir = {}
    for query_info in queries:
        qrels_beir[query_info['query_id']] = {}
        for positive_passage in query_info['positive_passages']:
            qrels_beir[query_info['query_id']][positive_passage['docid']] = 1
        for negative_passage in query_info['negative_passages']:
            qrels_beir[query_info['query_id']][negative_passage['docid']] = 0

    # The original corpus is far too big for the experiment, so we will only use the docs mentioned in qrels.
    corpus_subset = {doc_id: corpus_beir[doc_id] for docs in qrels_beir.values() for doc_id in docs}
    return corpus_subset, queries_beir, qrels_beir

def load_data(dataset: str):
    if dataset == 'scidocs':
        return load_beir("scidocs")
    elif dataset == 'mldr':
        return load_mldr()
    else:
        raise ValueError(f"Dataset {dataset} not found.")

# if __name__ == "__main__":
#     corpus, queries, qrels = load_beir("scidocs")
#     temp_title = ""
#     for doc_id, doc in corpus.items():
#         print(doc_id, doc)
#         break