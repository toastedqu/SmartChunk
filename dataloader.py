import os
import typing
import functools
import datasets
import numpy as np
import pandas as pd
from beir.datasets.data_loader import GenericDataLoader


def __set_origin__(row, origin: str):
    row["origin"] = origin
    return row


# read corpus to a dataset
def read_corpus(data_folder: str, num_proc: int = 1) -> datasets.Dataset:
    d: datasets.Dataset = datasets.Dataset.from_json(os.path.join(data_folder, "corpus.jsonl"), num_proc=num_proc)
    d = d.rename_column("_id", "id")
    d = d.select_columns(["id", "text", "title"])
    d = d.map(functools.partial(__set_origin__, origin="corpus"), desc="Set origin for corpus", batched=False, num_proc=num_proc)
    return d


# convert corpus to BEIR dict
def convert_corpus(corpus: datasets.Dataset):
    # if there is not a feature called "origin"
    if "origin" not in corpus.features:
        return {row["id"]: {"text": row["text"], "title": row["title"]} for row in corpus.to_list()}
    return {row["id"]: {"text": row["text"], "title": row["title"], "origin": row["origin"]} for row in corpus.to_list()}


# read queries to a dataset
def read_queries(data_folder: str, num_proc: int = 1) -> datasets.Dataset:
    d: datasets.Dataset = datasets.Dataset.from_json(os.path.join(data_folder, "queries.jsonl"), num_proc=num_proc)
    d = d.rename_column("_id", "id")
    d = d.select_columns(["id", "text"])
    d = d.map(functools.partial(__set_origin__, origin="queries"), desc="Set origin for queries", batched=False, num_proc=num_proc)
    return d


# convert queries to BEIR dict
def convert_queries(queries: datasets.Dataset):
    return {row["id"]: row["text"] for row in queries.to_list()}


# read corpus, queries, and qrels to Pandas DataFrame
def read_pd(data_folder: str, split: str = "test"):
    corpus, queries, qrels = GenericDataLoader(data_folder=data_folder).load(split=split)
    # convert back from BEIR dict format
    corpus_pd = pd.DataFrame.from_dict(corpus).T
    queries_pd = pd.DataFrame(queries, index=["text"]).T
    corpus_pd.index = corpus_pd.index.set_names("id")
    queries_pd.index = queries_pd.index.set_names("id")
    corpus_pd["origin"] = "corpus"
    queries_pd["origin"] = "queries"
    return corpus_pd, queries_pd, qrels


# read qrels to a dataset
def read_qrels(data_folder: str, split: typing.Optional[str] = None, num_proc = 1) -> datasets.Dataset:
    if split is None:
        # get all the splits in the qrels folder
        splits = [f.split(".")[0] for f in os.listdir(os.path.join(data_folder, "qrels"))]
        return {split: datasets.Dataset.from_csv(os.path.join(data_folder, "qrels", "{}.tsv".format(split)), delimiter="\t", num_proc=num_proc) for split in splits}
    return datasets.Dataset.from_csv(os.path.join(data_folder, "qrels", "{}.tsv".format(split)), delimiter="\t", num_proc=num_proc)


# convert qrels to BEIR dict
def convert_qrels(qrels: datasets.Dataset):
    return {row["doc_id"]: {row["query_id"]: row["score"]} for row in qrels.to_list()}


# randomly sample a dataset using qrels for n samples
def sample(corpus: datasets.Dataset, queries: datasets.Dataset, qrels: datasets.Dataset, n_samples: int = 10000, num_proc: int = 1):
    if len(corpus) < n_samples:
        return corpus, queries, qrels
    random_indices = np.random.choice(len(queries), size=n_samples, replace=False)
    qrels = qrels.select(random_indices)
    doc_ids = set(qrels["doc_id"])
    query_ids = set(qrels["query_id"])
    corpus = corpus.filter(lambda row: row["id"] in doc_ids, desc="Filtering corpus", num_proc=num_proc)
    queries = queries.filter(lambda row: row["id"] in query_ids, desc="Filtering queries", num_proc=num_proc)
    return corpus, queries, qrels
