import functools

import datasets

import dataloader
import chunker
import retrieve

dataset: datasets.Dataset = datasets.load_dataset("squad", split="train")  # trivia_qa, natural_questions

dataset.shuffle()
# select 10000 samples
dataset = dataset.select(range(10000))

dataloader.qair_init()  # initialize the records for the chunker
dataset.map(functools.partial(dataloader.qair_process, chunker.cluster_chunker, k=3, mode="k-preserve"), batched=True, batch_size=1)
dataset_chunked = datasets.Dataset.from_dict(dataloader.corpus_dict_qair)
corpus_split = dataset_chunked.filter(lambda row: row["type"] == "context_split")
queries = dataset_chunked.filter(lambda row: row["type"] == "question")
qrels_beir = dataloader.qrels_qair

corpus_split_beir = dataloader.convert_corpus(corpus_split)
queries_beir = dataloader.convert_queries(queries)

retrieve.main(corpus_split_beir, queries_beir, qrels_beir)
