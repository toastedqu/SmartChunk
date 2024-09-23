import numpy as np
import pickle
import json
import os
from tqdm import tqdm
from chunk_data import get_configs, create_hyperparam_str
from sentence_transformers import util
import copy

def save_sim_scores(dataset, hyperparam_str):
    queries = json.load(open(f"datasets/{dataset}/queries.json"))
    queries_embeddings = np.load(f"embeddings/{dataset}/queries.npy")
    chunked_docs_embeddings = pickle.load(open(f"embeddings/{dataset}/chunk_embeddings/{hyperparam_str}.pkl", 'rb'))

    cache_query_doc_scores = {} # query_id -> "doc_id|chunk_id" -> [chunk_scores]
    l_queries = list(queries.keys())

    # calculate similarity scores between queries and chunked documents
    for query_idx, query_id in enumerate(l_queries):
        cache_query_doc_scores[query_id] = {}
        for doc_id, chunked_doc_embeddings in enumerate(chunked_docs_embeddings):
            query_chunked_doc_sim_scores = util.cos_sim(queries_embeddings, chunked_doc_embeddings).numpy()
            for chunk_id in range(chunked_doc_embeddings.shape[0]):
                doc_chunk_id_key = f"{doc_id}|{chunk_id}"
                cache_query_doc_scores[query_id][doc_chunk_id_key] = query_chunked_doc_sim_scores[query_idx][chunk_id]
    if not os.path.exists("results"):               os.mkdir("results")
    if not os.path.exists(f"results/{dataset}"):    os.mkdir(f"results/{dataset}")
    pickle.dump(cache_query_doc_scores, open(f"results/{dataset}/scores_{hyperparam_str}.pkl", 'wb'))

def save_sim_scores_all_cfgs():
    print("Saving similarity scores for all configurations...")
    cfgs = get_configs()
    for cfg in tqdm(cfgs):
        hyperparams = copy.deepcopy(cfg['hyperparams'])
        dataset = copy.deepcopy(cfg['dataset'])
        chunker_name = hyperparams.pop("chunker")
        hyperparam_str = create_hyperparam_str(chunker_name, hyperparams)
        save_sim_scores(dataset, hyperparam_str)

def get_f1(r, p):
    return 2*r*p/(r+p)

def evaluate(retrieved_sents, evidence):
    recalls = []
    precisions = []
    for query_id, doc_sent_ids in evidence.items():
        if len(doc_sent_ids) == 0:
            continue

        correct_inclusions = 0
        total_for_recall =  sum(len(sent_ids) for sent_ids in doc_sent_ids.values())
        total_for_precision = sum(len(sent_ids) for sent_ids in retrieved_sents[query_id].values())
        if total_for_recall == 0 or total_for_precision == 0:
            continue

        for doc_id, sent_ids in doc_sent_ids.items():
            if doc_id in retrieved_sents[query_id]:
                for sent_id in sent_ids:
                    if sent_id in retrieved_sents[query_id][doc_id]:
                        correct_inclusions += 1

        recall = correct_inclusions / total_for_recall if correct_inclusions <= total_for_recall else 1
        precision = correct_inclusions / total_for_precision if correct_inclusions <= total_for_precision else 1
        recalls.append(recall)
        precisions.append(precision)
    return np.mean(recalls), np.mean(precisions)

def get_retrieval_results_at_k(dataset, hyperparam_str, k=1):
    # load data
    evidence = json.load(open(f"datasets/{dataset}/evidence.json"))
    chunks = json.load(open(f"datasets/{dataset}/chunks/{hyperparam_str}.json"))
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
    print("Saving results for all configurations...")
    results = {}
    cfgs = get_configs()
    for cfg in tqdm(cfgs):
        hyperparams = copy.deepcopy(cfg['hyperparams'])
        dataset = copy.deepcopy(cfg['dataset'])
        chunker_name = hyperparams.pop("chunker")
        hyperparam_str = create_hyperparam_str(chunker_name, hyperparams)
        if dataset not in results:
            results[dataset] = {}
        results[dataset][hyperparam_str] = {}
        for k in [1, 3, 5, 10]:
            recall, precision = get_retrieval_results_at_k(dataset, hyperparam_str, k)
            results[dataset][hyperparam_str][k] = {"recall": recall, "precision": precision, "f1": get_f1(recall, precision)}
    pickle.dump(results, open(f"results.pkl", 'wb'))

if __name__ == "__main__":
    save_sim_scores_all_cfgs()
    save_results()