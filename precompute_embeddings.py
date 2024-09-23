import os
import yaml
import numpy as np
from tqdm import tqdm
from encoder import Encoder
from joblib import Parallel, delayed
from dataloader import load_data, setup_logger

def precompute_query_embeddings(queries, dataset, encoder):
    if not os.path.exists('embeddings'):           os.mkdir('embeddings')
    if not os.path.exists(f'embeddings/{dataset}'): os.mkdir(f'embeddings/{dataset}')
    queries_embs = encoder.encode_queries(list(queries.values()))
    np.save(f'embeddings/{dataset}/queries.npy', queries_embs)

def precompute_sentence_embeddings(docs, dataset, encoder):
    if not os.path.exists('embeddings'):           os.mkdir('embeddings')
    if not os.path.exists(f'embeddings/{dataset}'): os.mkdir(f'embeddings/{dataset}')
    for i, doc in tqdm(enumerate(docs)):
        sent_embs = encoder.encode_documents(doc)
        np.save(f'embeddings/{dataset}/doc_{i}.npy', sent_embs)

def precompute_embeddings(dataset, device):
    logger = setup_logger()
    encoder = Encoder(device=device)
    queries, _, docs, _ = load_data(dataset)

    logger.info(f"Precomputing query embeddings for {dataset}...")
    precompute_query_embeddings(queries, dataset, encoder)
    logger.info(f"Query embeddings for {dataset} saved.")

    logger.info(f"Precomputing sentence embeddings for {dataset}...")
    precompute_sentence_embeddings(docs, dataset, encoder)
    logger.info(f"Sentence embeddings for {dataset} saved.")

if __name__ == '__main__':
    cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    Parallel(n_jobs=4)(delayed(precompute_embeddings)(dataset=dataset, device=i) for i,dataset in enumerate(cfg['datasets']))