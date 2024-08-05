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

def get_subset(corpus, queries, qrels, subset_size = 500):
    """Get a random subset of corpus, queries and qrels.

    Args:
        corpus (Dict[str, Dict[str, str]]): The corpus.
        queries (Dict[str, str]): The queries.
        qrels (Dict[str, Dict[str, float]]): The relevance judgements.
        subset_size (int): The size of the subset to get.

    Returns:
        Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, float]]]: The subset corpus, queries, and qrels.
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

def augment_corpus(corpus, n_short_per_long = 10):
    """Augment the corpus by concatenating multiple short documents into a single long document.

    Args:
        corpus (Dict[str, Dict[str, str]]): The corpus to augment.
        n_short_per_long (int): The number of short documents to concatenate into a single long document.

    Returns:
        Dict[str, Dict[str, str]]: The augmented corpus.
    """
    corpus_augmented = {}
    short_doc_ids, short_docs = [], []
    long_doc_ids, long_docs = [], []

    # ensure randomness in the order of short documents.
    keys = list(corpus.keys())
    random.shuffle(keys)

    counter = 0
    for doc_id in keys:
        doc = corpus[doc_id]
        counter += 1
        short_doc_ids.append(doc_id)
        short_docs.append(join_title_text(doc))

        # when we have n_short_per_long short documents, we create a long document
        if counter == n_short_per_long:
            long_doc_id = "/".join(short_doc_ids)
            long_doc = "\n".join(short_docs)
            long_doc_ids.append(long_doc_id)
            long_docs.append(long_doc)

            # reset cache
            short_doc_ids, short_docs = [], []
            counter = 0

    # if there are leftover short documents, create a long document with them alone.
    if counter != 0:
        long_doc_id = "/".join(short_doc_ids)
        long_doc = "\n".join(short_docs)
        long_doc_ids.append(long_doc_id)
        long_docs.append(long_doc)

    # create new corpus with long documents
    for long_doc_id, long_doc in zip(long_doc_ids, long_docs):
        corpus_augmented[long_doc_id] = {"text": long_doc}
    return corpus_augmented

def modify_qrels(qrels, corpus_augmented):
    """Modify the qrels to match the long documents in the augmented corpus.

    Args:
        qrels (Dict[str, Dict[str, float]]): The qrels to modify.
        corpus_augmented (Dict[str, Dict[str, str]]): The augmented corpus.

    Returns:
        Dict[str, Dict[str, float]]: The modified qrels.
    """
    qrels_modified = {}
    for query_id, qrel in qrels.items():
        qrels_modified[query_id] = {}
        for doc_id, score in qrel.items():
            # skip irrelevant documents
            if score == 0: continue

            # search for the long document that contains the short document
            for long_doc_id in corpus_augmented:
                if doc_id in long_doc_id:
                    # if the long document is not in the qrels or scored 1, change the score to curr score (either 1 or 2).
                    if long_doc_id not in qrels_modified[query_id] or qrels_modified[query_id][long_doc_id] == 1:
                        qrels_modified[query_id][long_doc_id] = score

                    # if we already have a score-2 document, skip the rest
                    # because the score of the long document is 2 if any of the short documents is 2.
                    if qrels_modified[query_id][long_doc_id] == 2:
                        break

                    # if the short document is in the long document, skip the remaining long documents.
                    break

    return qrels_modified

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

def load_data(dataset: str):
    if dataset in ['nfcorpus','nq','hotpotqa','fiqa', 'webis-touche2020','dbpedia-entity','scidocs','fever','climate-fever','scifact']:
        return load_beir(dataset)
    elif dataset == 'mldr':
        return load_mldr()
    else:
        raise ValueError(f"Dataset {dataset} not found.")