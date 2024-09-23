import numpy as np
import pickle
import json
import os
from tqdm import tqdm
from src.utils import get_configs, create_hyperparam_str
from sentence_transformers import util
import copy

###### helper functions ######
def get_f1(r, p):
    """Calculate the F1 score given recall and precision.

    Args:
        r (float): The recall.
        p (float): The precision.

    Returns:
        float: The F1 score."""
    return 2*r*p/(r+p)

def concat_results_pkl(old_results_path, new_results_path):
    """Concatenate the results from two pickle files.

    In case of interrupted experiments, this function can be used to concatenate the results
    and save the combined results in the old results file.

    Args:
        old_results_path (str): The path to the old results pickle file.
        new_results_path (str): The path to the new results pickle file.
    """
    results_old = pickle.load(open(old_results_path, 'rb'))
    results_new = pickle.load(open(new_results_path, 'rb'))
    results = results_old | results_new
    pickle.dump(results, open(old_results_path, 'wb'))



###### main functions ######
def save_sim_scores_per_cfg(dataset, hyperparam_str):
    """Calculate and save similarity scores between queries and chunked documents for a given configuration.

    Args:
        dataset (str): The dataset name.
        hyperparam_str (str): The hyperparameter string.
    """
    # load data
    queries = json.load(open(f"datasets/{dataset}/queries.json"))
    queries_embeddings = np.load(f"embeddings/{dataset}/queries.npy")
    chunked_docs_embeddings = pickle.load(open(f"chunk_embeddings/{dataset}/{hyperparam_str}.pkl", 'rb'))

    # init
    cache_query_doc_scores = {}         # query_id -> "doc_id|chunk_id" -> chunk_scores
    l_queries = list(queries.keys())

    # calculate similarity scores between queries and chunked documents
    for query_idx, query_id in enumerate(l_queries):
        cache_query_doc_scores[query_id] = {}
        for doc_id, chunked_doc_embeddings in enumerate(chunked_docs_embeddings):
            query_chunked_doc_sim_scores = util.cos_sim(queries_embeddings, chunked_doc_embeddings).numpy()
            for chunk_id in range(chunked_doc_embeddings.shape[0]):
                doc_chunk_id_key = f"{doc_id}|{chunk_id}"
                cache_query_doc_scores[query_id][doc_chunk_id_key] = query_chunked_doc_sim_scores[query_idx][chunk_id]

    # save the scores
    if not os.path.exists("results"):               os.mkdir("results")
    if not os.path.exists(f"results/{dataset}"):    os.mkdir(f"results/{dataset}")
    pickle.dump(cache_query_doc_scores, open(f"results/{dataset}/scores_{hyperparam_str}.pkl", 'wb'))

def save_sim_scores():
    """Calculate and save similarity scores between queries and chunked documents for all configurations.
    """
    cfgs = get_configs()
    for cfg in tqdm(cfgs):
        hyperparam_str = create_hyperparam_str(cfg['hyperparams'])
        save_sim_scores_per_cfg(cfg['dataset'], hyperparam_str)

def evaluate(retrieved_sents, evidence):
    """Evaluate the evidence retrieval performance.

    Args:
        retrieved_sents (dict): The retrieved sentences.
        evidence (dict): The ground truth evidence.

    Returns:
        float: The recall.
        float: The precision.
    """
    recalls, precisions = [], []

    for query_id, doc_sent_ids in evidence.items():
        # skip queries with no evidence
        if len(doc_sent_ids) == 0:
            continue

        # get totals for recall and precision
        correct_inclusions = 0
        total_for_recall = sum(len(sent_ids) for sent_ids in doc_sent_ids.values())
        total_for_precision = sum(len(sent_ids) for sent_ids in retrieved_sents[query_id].values())

        # skip queries with no retrieved or actual sentences
        if total_for_recall == 0 or total_for_precision == 0:
            continue

        # calculate correct inclusions
        for doc_id, sent_ids in doc_sent_ids.items():
            if doc_id in retrieved_sents[query_id]:
                for sent_id in sent_ids:
                    if sent_id in retrieved_sents[query_id][doc_id]:
                        correct_inclusions += 1

        # calculate recall and precision
        recall = correct_inclusions / total_for_recall if correct_inclusions <= total_for_recall else 1
        precision = correct_inclusions / total_for_precision if correct_inclusions <= total_for_precision else 1

        # append to lists
        recalls.append(recall)
        precisions.append(precision)

    # return mean
    return np.mean(recalls), np.mean(precisions)

def get_retrieval_results_at_k(dataset, hyperparam_str, k=1):
    """Get the retrieval results at k.

    Args:
        dataset (str): The dataset name.
        hyperparam_str (str): The hyperparameter string.
        k (int): The number of chunks to retrieve.

    Returns:
        float: The recall.
        float: The precision.
    """
    # load data
    evidence = json.load(open(f"datasets/{dataset}/evidence.json"))
    chunks = json.load(open(f"chunks/{dataset}/{hyperparam_str}.json"))
    cache_query_doc_scores = pickle.load(open(f"results/{dataset}/scores_{hyperparam_str}.pkl", 'rb'))

    # sort the scores and get top k chunks
    sorted_scores = {query_id: dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]) for query_id, scores in cache_query_doc_scores.items()}

    # get retrieved sentences
    retrieved_sents = {}
    for query_id, scores in sorted_scores.items():
        retrieved_sents[query_id] = {}
        for doc_chunk_id in scores:
            doc_id = doc_chunk_id.split("|")[0]
            if doc_id not in retrieved_sents[query_id]:
                retrieved_sents[query_id][doc_id] = []
            retrieved_sents[query_id][doc_id] += chunks[int(doc_id)][doc_chunk_id]

    return evaluate(retrieved_sents, evidence)

def save_results():
    """Calculate and save the retrieval results for all configurations.
    """
    results = {}
    cfgs = get_configs()
    for cfg in tqdm(cfgs):
        dataset = copy.deepcopy(cfg['dataset'])     # direct reference didn't work for some reason
        hyperparam_str = create_hyperparam_str(cfg['hyperparams'])

        if dataset not in results:
            results[dataset] = {}
        results[dataset][hyperparam_str] = {}

        # calculate results for each k
        for k in [1, 3, 5, 10]:
            recall, precision = get_retrieval_results_at_k(dataset, hyperparam_str, k)
            results[dataset][hyperparam_str][k] = {"recall": recall, "precision": precision, "f1": get_f1(recall, precision)}

    # save the overall results
    pickle.dump(results, open(f"results/results_evidence.pkl", 'wb'))



# Executing this script will save the similarity scores and retrieval results for all configurations.
if __name__ == "__main__":
    save_sim_scores()
    save_results()