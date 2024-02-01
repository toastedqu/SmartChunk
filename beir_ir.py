import pathlib, os
import functools

from beir import util

import datasets

import dataloader
import chunker
import retrieve

# download and unzip the dataset
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

# load and embed the dataset
corpus = dataloader.read_corpus(data_path)
queries = dataloader.read_queries(data_path)
qrels = dataloader.read_qrels(data_path, split="test")

corpus_len = len(corpus)

# chunk the documents into 4-splits
corpus_chunked = corpus.map(functools.partial(dataloader.beir_process, chunker.cluster_chunker, k=4, continuity=True), batched=True, batch_size=1)

# filter out the original documents, only keep the split documents
corpus_split: datasets.Dataset = corpus_chunked.filter(lambda row: row["origin"] != "corpus")

corpus_split_beir = dataloader.convert_corpus(corpus_split)
queries_beir = dataloader.convert_queries(queries)
qrels_beir = dataloader.convert_qrels(qrels)

retrieve.main(corpus_split_beir, queries_beir, qrels_beir)
