import os
import yaml
import json
import time
import itertools
import numpy as np
from typing import *
from encoder import *
from chunkers import chunk_corpus
from dataloader import logger, load_data
from joblib import Parallel, delayed
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
k_values = [1,3,5,10,20]

# load model
def timed_retrieve(corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], retriever: EvaluateRetrieval):
    """Retrieve documents for the given queries and measure the retrieval latency.

    Args:
        corpus (Dict[str, Dict[str, str]]): chunk_id -> {"title": str, "text": str}
        queries (Dict[str, str]): query_id -> query

    Returns:
        chunk_results (Dict[str, Dict[str, float]]): query_id -> {chunk_id -> score}
        latency (float): The time taken to retrieve the chunks.
    """
    logger.info("Retrieving...")
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
    logger.info("Reconstructing from chunks to docs...")
    doc_scores = {}
    for query_id, query_result in chunk_results.items():
        doc_scores[query_id] = {}
        for chunk_id, score in query_result.items():
            doc_id = chunk_id.split("~")[0]
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
    logger.info("Aggregating scores...")
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
             retriever: EvaluateRetrieval,
             latency: float):
    """Evaluate the retrieval performance of the given doc results.

    Args:
        qrels (Dict[str, Dict[str, float]]): query_id -> {doc_id -> relevance_score}
        doc_results (Dict[str, Dict[str, float]]): query_id -> {doc_id -> doc_score}

    logs:
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
    # recall_cap = retriever.evaluate_custom(qrels, doc_results, k_values, metric="recall_cap")
    hole = retriever.evaluate_custom(qrels, doc_results, k_values, metric="hole")
    top_k_accuracy = retriever.evaluate_custom(qrels, doc_results, k_values, metric="top_k_accuracy")

    # log results
    eval_results = {
        "NDCG@k": ndcg,
        "MAP@k": _map,
        "Recall@k": recall,
        "Precision@k": precision,
        "MRR@k": mrr,
        # "R_cap@k": recall_cap,
        "Hole@k": hole,
        "Accuracy@k": top_k_accuracy,
        "Latency": latency,
    }
    logger.info("Evaluation Results:")
    for metric, value in eval_results.items():
        logger.info(f"{metric}: {value}")
    logger.info()

    return eval_results


def log_results(eval_results, dataset, encoder_name, chunker, kwargs):
    """Log the evaluation results to a file.

    Args:
        eval_results (Dict[str, Dict[str, float]): The evaluation results.
        dataset (str): The name of the dataset.
        encoder_name (str): The name of the encoder.
        chunker (str): The name of the chunker.
        kwargs (Dict): The hyperparameters.
    """
    if not os.path.exists(f"results"): os.mkdir("results")
    with open(f"results/results.jsonl", "a") as f:
        eval_results.update({
            "dataset": dataset,
            "encoder": encoder_name,
            "chunker": chunker,
            "hyperparams": kwargs
        })
        f.write(json.dumps(eval_results) + "\n")


def run(dataset, encoder_name, chunker_cfg, device):
    """Run the experiment for the given dataset.

    Args:
        dataset (str): The name of the dataset.
    """
    chunker = chunker_cfg.pop("chunker")
    kwargs = chunker_cfg
    if chunker in {"cluster_chunker", "dbscan_chunker", "langchain_chunker", "baseline_chunker"}:
        kwargs["precomputed_embeddings"] = dataset
        kwargs["encoder_name"] = encoder_name

    retriever = EvaluateRetrieval(DRES(CustomEncoder(encoder_name, device), batch_size=16, show_progress_bar=False))

    logger.info(f"Dataset: {dataset}")
    logger.info(f"Encoder: {encoder_name}")
    logger.info(f"Chunker: {chunker}")
    logger.info(f"Hyperparameters: {kwargs}")

    # load data
    logger.info("Loading data...")
    corpus, queries, qrels = load_data(dataset)

    # chunk corpus
    logger.info("Chunking corpus...")
    corpus_chunked = chunk_corpus(corpus, chunker, verbose=False, **kwargs)

    # retrieve
    logger.info("Retrieving...")
    chunk_results, latency = timed_retrieve(corpus_chunked, queries, retriever)

    # reconstruct from chunks to docs
    logger.info("Reconstructing scores from chunks to docs...")
    doc_scores = chunks_to_docs(chunk_results)

    # aggregate doc scores
    logger.info("Aggregating scores...")
    doc_results = aggregate_doc_scores(doc_scores)

    # evaluate
    logger.info("Evaluating...")
    eval_results = evaluate(qrels, doc_results, retriever, latency)

    # log results
    logger.info("Logging results...")
    log_results(eval_results, dataset, encoder_name, chunker, kwargs)


def get_configs():
    """Get a list of all combinations of hyperparameter configurations for the experiment.

    Returns:
        List[Dict[str, Any]]: The list of configurations.
    """
    cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    temp = {
        'dataset': cfg['datasets'],
        'encoder_name': cfg['encoders'],
        'chunker_cfg': []
    }
    chunkers = cfg['chunkers']
    for chunker in chunkers:
        if chunker in {'langchain_chunker', 'baseline_chunker'}:
            combinations = [{
                "breakpoint_threshold_type": breakpoint_threshold_type,
                "breakpoint_threshold_amount": breakpoint_threshold_amount
            } for breakpoint_threshold_type in cfg[chunker]['breakpoint_threshold_type'] for breakpoint_threshold_amount in cfg[chunker][breakpoint_threshold_type]]
        elif chunker in cfg:
            keys, values = zip(*cfg[chunker].items())
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        else:
            combinations = [{}]
        for combination in combinations:
            combination['chunker'] = chunker
        temp['chunker_cfg'] += combinations

    keys, values = zip(*temp.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations



if __name__ == "__main__":
    cfgs = get_configs()
    Parallel(n_jobs=4)(delayed(run)(**cfg, device=f"cuda:{i%4}") for i, cfg in enumerate(cfgs))