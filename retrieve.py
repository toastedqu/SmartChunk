import os
import yaml
import json
import time
import argparse
import numpy as np
from typing import *
from chunkers import chunk_corpus
from dataloader import load_data
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

# load model
cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
model = DRES(models.SentenceBERT(cfg["encoder_model"]), batch_size=16, show_progress_bar=False)
retriever = EvaluateRetrieval(model, score_function="dot")

def timed_retrieve(corpus: Dict[str, Dict[str, str]],
                   queries: Dict[str, str]):
    """Retrieve documents for the given queries and measure the retrieval latency.

    Args:
        corpus (Dict[str, Dict[str, str]]): chunk_id -> {"title": str, "text": str}
        queries (Dict[str, str]): query_id -> query

    Returns:
        chunk_results (Dict[str, Dict[str, float]]): query_id -> {chunk_id -> score}
        latency (float): The time taken to retrieve the chunks.
    """
    print("Retrieving...")
    start = time.time()
    chunk_results = retriever.retrieve(corpus, queries)
    end = time.time()
    return chunk_results, end-start

def chunks_to_docs(chunk_results: Dict[str, Dict[str, float]]):
    """Reconstruct the chunk-level retrieval results to document-level results.

    Args:
        chunk_results (Dict[str, Dict[str, float]]): query_id -> {chunk_id -> score}

    Returns:
        doc_scores (Dict[str, Dict[str, List[float]]): query_id -> {doc_id -> [chunk_scores]}
    """
    print("Reconstructing from chunks to docs...")
    doc_scores = {}
    for query_id, query_result in chunk_results.items():
        doc_scores[query_id] = {}
        for chunk_id, score in query_result.items():
            doc_id = chunk_id.split("|")[0]
            doc_scores[query_id][doc_id] = doc_scores[query_id].get(doc_id, []) + [score]
    return doc_scores

def aggregate_doc_scores(doc_scores: Dict[str, Dict[str, List[float]]],
                         strategy: Literal["max", "mean", "sum"] = "max"):
    """Aggregate the chunk scores for each document using the given strategy.

    "max" makes the most sense because what we want is to find the chunk that answers the query.
    Even if a document contains only one chunk that answers the query,
    if that chunk is of the highest quality, then the document should be considered relevant.

    Args:
        doc_scores (Dict[str, Dict[str, List[float]]): query_id -> {doc_id -> [chunk_scores]}
        strategy (str): The aggregation strategy to use. One of "max", "mean", or "sum".

    Returns:
        doc_results (Dict[str, Dict[str, float]]): query_id -> {doc_id -> doc_score}
    """
    print("Aggregating scores...")
    doc_results = {}
    for query_id, query_result in doc_scores.items():
        doc_results[query_id] = {}
        for doc_id, score_list in query_result.items():
            if strategy == "max":
                doc_results[query_id][doc_id] = np.max(score_list)
            elif strategy == "mean":
                doc_results[query_id][doc_id] = np.mean(score_list)
            elif strategy == "sum":
                doc_results[query_id][doc_id] = np.sum(score_list)
    return doc_results

def evaluate(qrels: Dict[str, Dict[str, float]],
             doc_results: Dict[str, Dict[str, float]],
             latency: float,
             k_values: List[int] = retriever.k_values):
    """Evaluate the retrieval performance of the given doc results.

    Args:
        qrels (Dict[str, Dict[str, float]]): query_id -> {doc_id -> relevance_score}
        doc_results (Dict[str, Dict[str, float]]): query_id -> {doc_id -> doc_score}
        k_values (List[int]): The list of k values for evaluation

    Prints:
        NDCG@k: The ranking quality of retrieved documents up to position k, considering both relevance and position.
        MAP@k: The mean average precision of retrieved documents up to position k.
        Recall@k: The fraction of relevant documents retrieved within top-k results.
        Precision@k: The fraction of retrieved documents that are relevant within top-k results.
        MRR@k: The mean reciprocal rank of the first relevant document within top-k results.
        R_cap@k: The fraction of relevant documents retrieved within top-k results, capped at the number of relevant documents.
        Hole@k: The fraction of irrelevant documents retrieved within top-k results.
        Accuracy@k: The fraction of top-k results that are relevant.
        Latency: The time taken for retrieval.

    Returns:
        eval_results (Dict[str, Dict[str, float]]): metric -> {k -> value}
    """
    # evaluate
    ndcg, _map, recall, precision = retriever.evaluate(qrels, doc_results, k_values)
    mrr = retriever.evaluate_custom(qrels, doc_results, k_values, metric="mrr")
    recall_cap = retriever.evaluate_custom(qrels, doc_results, k_values, metric="recall_cap")
    hole = retriever.evaluate_custom(qrels, doc_results, k_values, metric="hole")
    top_k_accuracy = retriever.evaluate_custom(qrels, doc_results, k_values, metric="top_k_accuracy")

    # log results
    eval_results = {
        "NDCG@k": ndcg,
        "MAP@k": _map,
        "Recall@k": recall,
        "Precision@k": precision,
        "MRR@k": mrr,
        "R_cap@k": recall_cap,
        "Hole@k": hole,
        "Accuracy@k": top_k_accuracy,
        "Latency": latency,
    }
    print("Evaluation Results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value}")
    print()

    return eval_results

def main(dataset: str):
    """Run the experiment for the given dataset.

    Args:
        dataset (str): The name of the dataset.
    """
    # load data
    corpus, queries, qrels = load_data(dataset)

    # load config
    cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    for (chunker_name, kwargs) in cfg["chunker_config"]:
        # chunk corpus
        corpus_chunked = chunk_corpus(corpus, chunker_name, **kwargs)

        # retrieve
        chunk_results, latency = timed_retrieve(corpus_chunked, queries)

        # reconstruct from chunks to docs
        doc_scores = chunks_to_docs(chunk_results)

        # aggregate doc scores
        doc_results = aggregate_doc_scores(doc_scores)

        # evaluate
        eval_results = evaluate(qrels, doc_results, latency)

        # log results
        if not os.path.exists(f"results"): os.mkdir("results")
        with open(f"results/new_eval_results_{dataset}.jsonl", "a") as f:
            f.write(json.dumps(eval_results) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False, help="The dataset to use for evaluation.")
    args = parser.parse_args()

    if args.dataset is None:
        for dataset in ['nfcorpus','nq','hotpotqa','fiqa', 'webis-touche2020','dbpedia-entity','scidocs','fever','climate-fever','scifact', 'mldr']:
            print(f"Evaluating on {dataset}...")
            main(dataset)
    else:
        print(f"Evaluating on {args.dataset}...")
        main(args.dataset)