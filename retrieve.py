import logging

from beir import LoggingHandler
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import numpy as np

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.WARN,
                    handlers=[LoggingHandler()])

# load model
model_path = "msmarco-distilbert-base-tas-b"
model = DRES(models.SentenceBERT(model_path), batch_size=16, show_progress_bar=False)
retriever = EvaluateRetrieval(model, score_function="dot")

def retrieve(corpus_split_beir, queries_beir, top_k=20):
    return model.search(corpus_split_beir, queries_beir, top_k=top_k, score_function="dot")

def retrieve_and_evaluate(corpus_split_beir, queries_beir, qrels_beir, k_values=[5, 20]):
    # retrieve
    results = retriever.retrieve(corpus_split_beir, queries_beir)

    # re-construct the results to map and reweight back to the original documents
    original_scores, original_results = {}, {}
    for query_id, query_result in results.items():
        original_scores[query_id] = {}
        for doc_id, score in query_result.items():
            original_doc_id = "s".join(doc_id.split("s")[:-1])  # remove the split index
            if original_doc_id not in original_scores[query_id]:
                original_scores[query_id][original_doc_id] = []
            original_scores[query_id][original_doc_id].append(score)
    for query_id, query_result in original_scores.items():
        for doc_id, score_list in query_result.items():
            if query_id not in original_results:
                original_results[query_id] = {}
            original_results[query_id][doc_id] = np.mean(score_list)

    # evaluate NDCG@k, MAP@K, Recall@K, and Precision@K
    ndcg, _map, recall, precision = retriever.evaluate(qrels_beir, original_results, k_values=k_values)

    print("Evaluation Results:")
    print("NDCG@k:", ndcg)
    print("MAP@k:", _map)
    print("Recall@k:", recall)
    print("Precision@k:", precision)
