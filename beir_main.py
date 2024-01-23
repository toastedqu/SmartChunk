import logging
import pathlib, os
import functools

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import datasets

import dataloader
import chunker

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

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
corpus_chunked = corpus.map(functools.partial(chunker.beir, chunker.cluster_chunker, k=4, continuity=True), batched=True, batch_size=1)

# filter out the original documents, only keep the split documents
corpus_split: datasets.Dataset = corpus_chunked.filter(lambda row: row["origin"] != "corpus")

corpus_split_beir = dataloader.convert_corpus(corpus_split)
queries_beir = dataloader.convert_queries(queries)
qrels_beir = dataloader.convert_qrels(qrels)

# load model and retrieve
model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16)
retriever = EvaluateRetrieval(model, score_function="dot")
results = retriever.retrieve(corpus_split_beir, queries_beir)

# re-construct the results to map and reweight back to the original documents
original_results = {}
for query_id, query_result in results.items():
    original_results[query_id] = {}
    for doc_id, score in query_result.items():
        original_doc_id = doc_id.split("s")[0]
        if original_doc_id not in original_results[query_id]:
            original_results[query_id][original_doc_id] = score
        else:
            original_results[query_id][original_doc_id] += score
            # original_results[query_id][original_doc_id] = max(original_results[query_id][original_doc_id], score)

# evaluate NDCG@k, MAP@K, Recall@K, and Precision@K where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels_beir, original_results, retriever.k_values)

print("Evaluation Results:")
print("NDCG@k:", ndcg)
print("MAP@k:", _map)
print("Recall@k:", recall)
print("Precision@k:", precision)
