import os
import gc
import json
import torch
import pickle
import itertools
from chunkers import *
from utils import get_configs, create_hyperparam_str
from encoder import Encoder
from joblib import Parallel, delayed
from dataloader import setup_logger
# eval() didn't work for some reason
CHUNKER_MAP = {
    "BaseChunker": BaseChunker,
    "PositionalChunker": PositionalChunker,
    "SingleLinkageChunker": SingleLinkageChunker,
    "DBSCANChunker": DBSCANChunker,
    "LangchainChunker": LangchainChunker,
    "AbsoluteLangchainChunker": AbsoluteLangchainChunker
}

def chunk_data(dataset, hyperparams):
    logger = setup_logger()

    logger.info(f"Init chunker: {hyperparams}")
    chunker_name = hyperparams["chunker"]
    chunker = CHUNKER_MAP[chunker_name](**hyperparams)
    fname = create_hyperparam_str(hyperparams)

    logger.info(f"Chunking data for {dataset} using {chunker_name} with hyperparams: {hyperparams}")
    corpus = json.load(open(f"datasets/{dataset}/docs.json", "r"))
    corpus_chunked = chunker.chunk_corpus(corpus)

    logger.info(f"Saving chunks for {dataset} using {chunker_name} with hyperparams: {hyperparams}")
    if not os.path.exists(f"datasets/{dataset}/chunks"): os.mkdir(f"datasets/{dataset}/chunks")
    json.dump(corpus_chunked, open(f"datasets/{dataset}/chunks/{fname}.json", "w"))

def precompute_chunk_embeddings(dataset, hyperparams, device):
    logger = setup_logger()
    encoder = Encoder(device=device)

    chunker_name = hyperparams["chunker"]
    fname = create_hyperparam_str(hyperparams)
    chunks = json.load(open(f"datasets/{dataset}/chunks/{fname}.json", "r"))
    docs = json.load(open(f"datasets/{dataset}/docs.json", "r"))

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

    if not os.path.exists(f"embeddings/{dataset}/chunk_embeddings"): os.mkdir(f"embeddings/{dataset}/chunk_embeddings")
    pickle.dump(doc_embs, open(f"embeddings/{dataset}/chunk_embeddings/{fname}.pkl", "wb"))

    del encoder
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    cfgs = get_configs()
    Parallel(n_jobs=4)(delayed(chunk_data)(dataset=cfg['dataset'], hyperparams=cfg['hyperparams']) for cfg in cfgs)
    Parallel(n_jobs=2)(delayed(precompute_chunk_embeddings)(dataset=cfg['dataset'], hyperparams=cfg['hyperparams'], device=device) for cfg, device in zip(cfgs, itertools.cycle([0,1,2,3])))