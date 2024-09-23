import os
import gc
import json
import torch
import pickle
import itertools
from src.chunkers import *
from src.utils import setup_logger, get_configs, create_hyperparam_str
from src.encoder import Encoder
from joblib import Parallel, delayed

# map chunker names to chunker classes
# unfortunately, eval() didn't work for some reason
CHUNKER_MAP = {
    "BaseChunker": BaseChunker,
    "PositionalChunker": PositionalChunker,
    "SingleLinkageChunker": SingleLinkageChunker,
    "DBSCANChunker": DBSCANChunker,
    "LangchainChunker": LangchainChunker,
    "AbsoluteLangchainChunker": AbsoluteLangchainChunker
}

def chunk_data(dataset, hyperparams):
    """Chunk the data using the specified chunker and hyperparameters.

    Args:
        dataset (str): The dataset name.
        hyperparams (dict): The hyperparameters for the chunker.
    """
    # init
    logger = setup_logger()
    logger.info(f"Init chunker: {hyperparams}")
    chunker_name = hyperparams["chunker"]
    chunker = CHUNKER_MAP[chunker_name](**hyperparams)
    fname = create_hyperparam_str(hyperparams)

    # chunk docs
    logger.info(f"Chunking data for {dataset} using {chunker_name} with hyperparams: {hyperparams}")
    corpus = json.load(open(f"datasets/{dataset}/docs.json", "r"))
    corpus_chunked = chunker.chunk_corpus(corpus)

    # save chunks
    logger.info(f"Saving chunks for {dataset} using {chunker_name} with hyperparams: {hyperparams}")
    if not os.path.exists(f"chunks/{dataset}"): os.mkdir(f"chunks/{dataset}")
    json.dump(corpus_chunked, open(f"chunks/{dataset}/{fname}.json", "w"))

def precompute_chunk_embeddings(dataset, hyperparams, device):
    """Precompute and save chunk embeddings.

    Args:
        dataset (str): The dataset name.
        hyperparams (dict): The hyperparameters for the chunker.
        device (int): The device number for the encoder.
    """
    # init
    logger = setup_logger()
    encoder = Encoder(device=device)
    chunker_name = hyperparams["chunker"]
    fname = create_hyperparam_str(hyperparams)

    # load chunks and docs
    chunks = json.load(open(f"datasets/{dataset}/chunks/{fname}.json", "r"))
    docs = json.load(open(f"datasets/{dataset}/docs.json", "r"))

    # precompute chunk embeddings
    logger.info(f"Precomputing chunk embeddings for {dataset} using {chunker_name} with hyperparams: {hyperparams}")
    doc_embs = []
    for chunk in tqdm(chunks):
        chunk_texts = []
        for k, sent_ids in chunk.items():
            doc_id = int(k.split('|')[0])
            chunk_text = '\n'.join([docs[doc_id][sent_id] for sent_id in sorted(sent_ids)])
            chunk_texts.append(chunk_text)
        chunk_text_embs = encoder.encode_documents(chunk_texts)
        doc_embs.append(chunk_text_embs)

    # save chunk embeddings
    if not os.path.exists(f"chunk_embeddings/{dataset}"): os.mkdir(f"chunk_embeddings/{dataset}")
    pickle.dump(doc_embs, open(f"chunk_embeddings/{dataset}/{fname}.pkl", "wb"))

    # avoid OOM
    del encoder
    torch.cuda.empty_cache()
    gc.collect()



if __name__ == "__main__":
    cfgs = get_configs()

    # n_jobs=4 here does not cause OOM
    Parallel(n_jobs=4)(delayed(chunk_data)(dataset=cfg['dataset'], hyperparams=cfg['hyperparams']) for cfg in cfgs)

    # using n_jobs=2 to avoid OOM
    Parallel(n_jobs=2)(delayed(precompute_chunk_embeddings)(dataset=cfg['dataset'], hyperparams=cfg['hyperparams'], device=device) for cfg, device in zip(cfgs, itertools.cycle([0,1,2,3])))