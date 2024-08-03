import random
from typing import *
from datasets import load_dataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader
random.seed(2)

###### helper functions ######
def join_title_text(doc: Dict[str, str]) -> str:
    """Join the title and text of a document into a single string.

    Args:
        doc (Dict[str, str]): The document with keys "title" and "text".

    Returns:
        str: The title and text joined into a single string.
    """
    return (doc["title"] + ".\n" + doc["text"]).strip() if "title" in doc else doc["text"].strip()

def augment_corpus(corpus, n_short_per_long = 10):
    corpus_augmented = {}
    short_doc_ids, short_docs = [], []
    long_doc_ids, long_docs = [], []
    counter = 0
    keys = list(corpus.keys())
    random.shuffle(keys)
    for doc_id in keys:
        doc = corpus[doc_id]
        counter += 1
        short_doc_ids.append(doc_id)
        short_docs.append(join_title_text(doc))

        if counter == n_short_per_long:
            long_doc_id = "/".join(short_doc_ids)
            long_doc = "\n".join(short_docs)
            long_doc_ids.append(long_doc_id)
            long_docs.append(long_doc)
            short_doc_ids, short_docs = [], []
            counter = 0
    if counter != 0:
        long_doc_id = "/".join(short_doc_ids)
        long_doc = "\n".join(short_docs)
        long_doc_ids.append(long_doc_id)
        long_docs.append(long_doc)
    for long_doc_id, long_doc in zip(long_doc_ids, long_docs):
        corpus_augmented[long_doc_id] = {"text": long_doc}
    return corpus_augmented

def modify_qrels(qrels, corpus_augmented):
    qrels_modified = {}
    for query_id, qrel in qrels.items():
        qrels_modified[query_id] = {}
        for doc_id, score in qrel.items():
            if score == 0: continue
            for long_doc_id in corpus_augmented:
                if doc_id in long_doc_id:
                    if long_doc_id not in qrels_modified[query_id] or qrels_modified[query_id][long_doc_id] == 1:
                        qrels_modified[query_id][long_doc_id] = score
                    if qrels_modified[query_id][long_doc_id] == 2:
                        break
                    break
    return qrels_modified

def get_subset(corpus, queries, qrels, subset_size = 500):
    """Get a random subset of the queries and qrels.

    Args:
        queries (Dict[str, str]): The queries.
        qrels (Dict[str, Dict[str, float]]): The relevance judgements.
        subset_size (int): The size of the subset to get.

    Returns:
        Tuple[Dict[str, str], Dict[str, Dict[str, float]]]: The subset of the queries and qrels.
    """
    pos_doc_ids = set([doc_id for qrel in qrels.values() for doc_id in qrel])
    neg_doc_ids = [doc_id for doc_id in corpus if doc_id not in pos_doc_ids]

    if len(queries) > subset_size:
        query_ids = random.sample(list(queries.keys()), subset_size)
        queries_subset = {query_id: queries[query_id] for query_id in query_ids}
        qrels_subset = {query_id: qrels[query_id] for query_id in query_ids}
    else:
        queries_subset = queries
        qrels_subset = qrels

    pos_doc_ids = set([doc_id for qrel in qrels_subset.values() for doc_id in qrel])
    if len(neg_doc_ids) > len(pos_doc_ids):
        neg_doc_ids = random.sample(neg_doc_ids, len(pos_doc_ids))

    corpus_subset = {doc_id: corpus[doc_id] for doc_id in (list(pos_doc_ids) + neg_doc_ids)}
    return corpus_subset, queries_subset, qrels_subset

###### data loading functions ######
def load_beir(dataset: str, split: str = "test"):
    """Load a BEIR dataset by name.

    Args:
        dataset (str): The name of the dataset to load.

    Returns:
        Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, float]]]: The corpus, queries, and qrels.
    """
    # load BEIR dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    corpus_subset, queries_subset, qrels_subset = get_subset(corpus, queries, qrels)
    corpus_augmented = augment_corpus(corpus_subset)
    qrels_modified = modify_qrels(qrels_subset, corpus_augmented)

    return corpus_augmented, queries_subset, qrels_modified

def load_mldr(lang: str = 'en', split: str = 'test'):
    """Load the MLDR dataset, a multilingual dataset for Wikipedia document retrieval.

    Huggingface: https://huggingface.co/datasets/Shitao/MLDR
    Paper: https://arxiv.org/pdf/2402.03216

    Args:
        lang (str): The language of the dataset to load.
        split (str): The split of the dataset to load.
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

    corpus_subset, queries_subset, qrels_subset = get_subset(corpus_beir, queries_beir, qrels_beir)

    return corpus_subset, queries_subset, qrels_subset

def load_data(dataset: str):
    if dataset in ['nfcorpus','nq','hotpotqa','fiqa', 'webis-touche2020','dbpedia-entity','scidocs','fever','climate-fever','scifact']:
        return load_beir(dataset)
    elif dataset == 'mldr':
        return load_mldr()
    else:
        raise ValueError(f"Dataset {dataset} not found.")