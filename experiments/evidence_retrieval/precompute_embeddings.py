import os
import yaml
import numpy as np
from tqdm import tqdm
from src.encoder import Encoder
from src.utils import setup_logger
from joblib import Parallel, delayed
from load_data import load_data

def precompute_query_embeddings(queries, dataset, encoder):
    """Precompute and save query embeddings.

    Args:
        queries (dict): The queries dictionary (query_id -> query).
        dataset (str): The dataset name.
        encoder (Encoder): The encoder object.
    """
    if not os.path.exists('embeddings'):            os.mkdir('embeddings')
    if not os.path.exists(f'embeddings/{dataset}'): os.mkdir(f'embeddings/{dataset}')
    queries_embs = encoder.encode_queries(list(queries.values()))
    np.save(f'embeddings/{dataset}/queries.npy', queries_embs)

def precompute_sentence_embeddings(docs, dataset, encoder):
    """Precompute and save sentence embeddings.

    Args:
        docs (List[List[str]]): The list of documents, each splitted into sentences.
        dataset (str): The dataset name.
        encoder (Encoder): The encoder object.
    """
    if not os.path.exists('embeddings'):            os.mkdir('embeddings')
    if not os.path.exists(f'embeddings/{dataset}'): os.mkdir(f'embeddings/{dataset}')
    for i, doc in tqdm(enumerate(docs)):
        sent_embs = encoder.encode_documents(doc)
        np.save(f'embeddings/{dataset}/doc_{i}.npy', sent_embs)

def precompute_embeddings(dataset, device):
    """Precompute and save query and sentence embeddings.

    Args:
        dataset (str): The dataset name.
        device (int): The device number for the encoder.
    """
    logger = setup_logger()
    encoder = Encoder(device=device)

    # load data
    queries, _, docs, _ = load_data(dataset)

    # precompute and save query embeddings
    logger.info(f"Precomputing query embeddings for {dataset}...")
    precompute_query_embeddings(queries, dataset, encoder)

    # precompute and save sentence embeddings
    logger.info(f"Precomputing sentence embeddings for {dataset}...")
    precompute_sentence_embeddings(docs, dataset, encoder)

if __name__ == '__main__':
    cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    Parallel(n_jobs=4)(delayed(precompute_embeddings)(dataset=dataset, device=i) for i,dataset in enumerate(cfg['datasets']))