import random
from tqdm import tqdm
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

def get_query_subset(queries, qrels):
    """Get a random subset of the queries and qrels.

    Args:
        queries (Dict[str, str]): The queries.
        qrels (Dict[str, Dict[str, float]]): The relevance judgements.
        subset_size (int): The size of the subset to get.

    Returns:
        Tuple[Dict[str, str], Dict[str, Dict[str, float]]]: The subset of the queries and qrels.
    """
    subset_size = min(int(len(queries)*0.8), 100)
    query_ids = random.sample(list(queries.keys()), subset_size)
    queries_subset = {query_id: queries[query_id] for query_id in query_ids}
    qrels_subset = {query_id: qrels[query_id] for query_id in query_ids}
    return queries_subset, qrels_subset

def get_pos_doc_ids(qrels):
    """Get the set of positive doc IDs in qrels.

    Args:
        qrels (Dict[str, Dict[str, float]]): query_id -> doc_id -> score; The actual relevance scores.

    Returns:
        Set[str]: The set of positive doc IDs.
    """
    pos_doc_ids = set()
    for qrel in qrels.values():
        for doc_id in qrel:
            pos_doc_ids.add(doc_id)
    return pos_doc_ids

def reduce_qrels(corpus, qrels):
    """Modify the qrels because of the following issues:
    1. Some datasets have far too many entries per query in qrels (e.g., trec-covid). It takes too long to chunk such corpus.
       Solution: We limit the number of relevant documents to 10, and we fully get rid of irrelevant entries in qrels
       so that we don't have to face inconsistencies when augmenting the corpus with globally irrelevant documents.
    2. Some datasets have the score 2 in addition to 0 and 1.
       Solution: We take the score 2 into account, and we limit the number of "2" documents to 10.
    3. Some datasets don't have negative documents (0) in qrels. We need negative documents when augmenting corpus.
       Solution: We sample negative documents that are fully irrelevant to all sampled queries in qrels.

    Args:
        corpus (Dict[str, Dict[str, str]]): The corpus.
        qrels (Dict[str, Dict[str, float]]): The relevance judgements.

    Returns:
        Tuple[Dict[str, Dict[str, float]], List[str]]: The reduced qrels and the negative doc IDs.
    """
    # get the list of globally irrelevant docs
    pos_doc_ids = get_pos_doc_ids(qrels)
    neg_doc_ids = [doc_id for doc_id in corpus if doc_id not in pos_doc_ids]

    # reduce the qrels so that
    # 1) It only has a max of 10 "1" docs and 10 "2" docs per query.
    # 2) It has no "0" docs because we don't need them in qrels.
    qrels_reduced = {}
    for query_id, qrel in qrels.items():
        qrels_reduced[query_id] = {}
        qrel_subset_1 = [doc_id for doc_id, score in qrel.items() if score == 1]    # a list of docs with score 1 for curr query
        qrel_subset_2 = [doc_id for doc_id, score in qrel.items() if score == 2]    # a list of docs with score 2 for curr query

        # sample a max of 10 docs with score 1 and a max of 10 docs with score 2
        if len(qrel_subset_1) > 10:
            qrel_subset_1 = random.sample(qrel_subset_1, 10)
        if len(qrel_subset_2) > 10:
            qrel_subset_2 = random.sample(qrel_subset_2, 10)

        # store the sampled qrels
        qrels_reduced[query_id].update({doc_id: 1 for doc_id in qrel_subset_1})
        qrels_reduced[query_id].update({doc_id: 2 for doc_id in qrel_subset_2})

    # sample the negative docs to match the number of positive docs in the new qrels
    pos_doc_ids_reduced = get_pos_doc_ids(qrels_reduced)
    if len(neg_doc_ids) > len(pos_doc_ids_reduced):
        neg_doc_ids = random.sample(neg_doc_ids, len(pos_doc_ids_reduced))

    return qrels_reduced, neg_doc_ids

def augment_corpus(corpus, qrels_reduced, neg_doc_ids):
    """Augment the corpus by creating long docs from short docs.

    Args:
        corpus (Dict[str, Dict[str, str]]): The corpus.
        qrels_reduced (Dict[str, Dict[str, float]]): The reduced relevance judgements.
        neg_doc_ids (List[str]): The negative doc IDs.

    Returns:
        Dict[str, Dict[str, str]]: The augmented corpus.
    """
    corpus_augmented = {}
    for qrel in tqdm(qrels_reduced.values()):
        top_doc_ids = [doc_id for doc_id, score in qrel.items() if score == 2]  # a list of docs with score 2 for curr query
        pos_doc_ids = [doc_id for doc_id, score in qrel.items() if score == 1]  # a list of docs with score 1 for curr query
        for doc_id, score in qrel.items():
            # create and add the doc only if it's not in the augmented corpus already
            if doc_id not in corpus_augmented:
                # concatenate score-2 docs with other score-2 docs only
                # so that the new doc score is not diluted by score-1 docs
                if score == 2:
                    # if there are more than 4 score-2 docs, sample 4 of them
                    if len(top_doc_ids) > 4:
                        new_doc_ids = [doc_id] + random.sample(top_doc_ids, 3) + random.sample(neg_doc_ids, 4)
                    # else, use all of them
                    else:
                        new_doc_ids = top_doc_ids + random.sample(neg_doc_ids, 8-len(top_doc_ids))

                # concatenate score-1 docs with other score-1 docs only
                # so that the new doc score is not diluted by score-2 docs
                elif score == 1:
                    # if there are more than 4 score-2 docs, sample 4 of them
                    if len(pos_doc_ids) > 4:
                        new_doc_ids = [doc_id] + random.sample(pos_doc_ids, 3) + random.sample(neg_doc_ids, 4)
                    # else, use all of them
                    else:
                        new_doc_ids = pos_doc_ids + random.sample(neg_doc_ids, 8-len(pos_doc_ids))

                # shuffle the docs to avoid positional biases in chunking
                random.shuffle(new_doc_ids)

                # concatenate short docs into long doc for the curr doc_id
                new_doc = ' '.join([join_title_text(corpus[doc_id]) for doc_id in new_doc_ids])
                corpus_augmented[doc_id] = {"text": new_doc}

    # sample & concatenate negative docs
    for neg_doc_id in neg_doc_ids:
        sampled_neg_doc_ids = [neg_doc_id] + random.sample(neg_doc_ids, 7)
        random.shuffle(neg_doc_ids)
        new_doc = ' '.join([join_title_text(corpus[doc_id]) for doc_id in sampled_neg_doc_ids])
        corpus_augmented[neg_doc_id] = {"text": new_doc}

    return corpus_augmented

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

    # get a subset of queries and qrels because the full corpus is too big
    queries, qrels = get_query_subset(queries, qrels)

    # reduce qrels and get globally negative doc ids because they have too many entries per query
    qrels_reduced, neg_doc_ids = reduce_qrels(corpus, qrels)

    # augment the corpus with long docs
    corpus_augmented = augment_corpus(corpus, qrels_reduced, neg_doc_ids)

    return corpus_augmented, queries, qrels_reduced

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

    # only use the docs mentioned in qrels as the corpus because the full corpus is too big
    corpus_subset = {doc_id: corpus_beir[doc_id] for docs in qrels_beir.values() for doc_id in docs}

    return corpus_subset, queries_beir, qrels_beir

def load_data(dataset: str):
    if dataset in ['trec-covid','nfcorpus','nq','hotpotqa','fiqa', 'webis-touche2020','dbpedia-entity','scidocs','fever','climate-fever','scifact']:
        return load_beir(dataset)
    elif dataset == 'mldr':
        return load_mldr()
    else:
        raise ValueError(f"Dataset {dataset} not found.")