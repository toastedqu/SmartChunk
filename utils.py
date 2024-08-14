import spacy
import random
from typing import *
nlp = spacy.load('en_core_web_sm')
random.seed(2)

def split_sentences(doc: str) -> List[str]:
    """Split the document into sentences.

    Args:
        doc (str): The document text

    Returns:
        List[str]: The list of sentences
    """
    return [text.text.strip() for segment in doc.split("\n") for text in nlp(segment).sents]

def join_title_text(doc: Dict[str, str]) -> str:
    """Join the title and text of a document into a single string.

    Args:
        doc (Dict[str, str]): The document with keys "title" and "text".

    Returns:
        str: The title and text joined into a single string.
    """
    return (doc["title"] + ".\n" + doc["text"]).strip() if "title" in doc else doc["text"].strip()

def get_subset(corpus, queries, qrels, subset_size = 50):
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