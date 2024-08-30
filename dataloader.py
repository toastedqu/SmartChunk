import os
import sys
import yaml
import json
import random
import logging
import itertools
import pandas as pd
from typing import *
from utils import *
from tqdm import tqdm
from datasets import load_dataset
from bs4 import BeautifulSoup
from encoder import CustomEncoder
from joblib import Parallel, delayed
from beir import util
from beir.datasets.data_loader import GenericDataLoader
random.seed(2)
logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# how many short documents should be combined into a long document for synthetic data
avg_n_short_per_doc = {
    "nfcorpus": 10,
    "nq": 20,
    "webis-touche2020": 5,
    "scidocs": 10,
    "scifact": 10,
    "miracl": 25,
    "bioasq": 10,
}

# the following datasets were unused.
# "hotpotqa": 25,
# "fiqa": 15,
# "quora": 100,
# "dbpedia-entity": 25,
# "fever": 25,
# "climate-fever": 25,



###### helper functions ######
def qrels_tsv_to_dict(qrels: pd.DataFrame):
    """Convert the qrels tsv dataframe to a dictionary.

    Args:
        qrels (pd.DataFrame): The qrels dataframe. It should have the following 3 columns: query_id, doc_id, score.

    Returns:
        Dict: The qrels dictionary.
    """
    qrels_dict = {}
    for query_id in qrels['query_id']:
        rows = qrels[(qrels['query_id'] == query_id) & (qrels['score'] == 1)]
        qrels_dict[query_id] = dict(zip(rows['doc_id'], rows['score']))
    return qrels_dict


def is_processed(dataset: str):
    """Check if the dataset has been processed before.

    A processed dataset satisfies the following conditions:
    - It consists of 3 files in the BEIR format: corpus, queries, and qrels.
    - The corpus meets the long document requirement.
    - All 3 files are subsets with a maximum of 100 queries.

    Args:
        dataset (str): The dataset name.

    Returns
        bool: True if the dataset has been processed before, False otherwise.
    """
    corpus_exists = os.path.exists(f"subsets/{dataset}/corpus.json")
    queries_exists = os.path.exists(f"subsets/{dataset}/queries.json")
    qrels_exists = os.path.exists(f"subsets/{dataset}/qrels.json")
    return corpus_exists and queries_exists and qrels_exists


def get_subset(corpus, queries, qrels, subset_size = 100):
    """Get a subset of the BEIR-formatted dataset.

    Args:
        corpus (Dict): The corpus dictionary.
        queries (Dict): The queries dictionary.
        qrels (Dict): The qrels dictionary.
        subset_size (int): The number of queries in the subset.

    Returns:
        Tuple[Dict, Dict, Dict]: The subset of corpus, queries, and qrels.
    """
    logger.info("Getting subset of queries and qrels...")
    if len(queries) > subset_size:
        query_ids = random.sample(list(queries.keys()), subset_size)
        queries_subset = {query_id: queries[query_id] for query_id in query_ids}
        qrels_subset = {query_id: {doc_id: score for doc_id, score in qrels[query_id].items() if score > 0} for query_id in query_ids}
    else:
        queries_subset = queries
        qrels_subset = {query_id: {doc_id: score for doc_id, score in qrels[query_id].items() if score > 0} for query_id in queries}

    logger.info("Getting subset of corpus...")
    pos_doc_ids = set([doc_id for qrel in qrels_subset.values() for doc_id in qrel if qrel[doc_id] > 0])
    neg_doc_ids = [doc_id for doc_id in corpus if doc_id not in pos_doc_ids]

    # ensure the number of negative docs is at most 3 times the number of positive docs
    if len(neg_doc_ids) > 3*len(pos_doc_ids):
        neg_doc_ids = random.sample(neg_doc_ids, 3*len(pos_doc_ids))

    corpus_subset = {doc_id: corpus[doc_id] for doc_id in (list(pos_doc_ids) + neg_doc_ids)}
    return corpus_subset, queries_subset, qrels_subset


def save_subset(corpus, queries, qrels, dataset: str):
    """Save the subset of the BEIR-formatted dataset.

    Args:
        corpus (Dict): The corpus dictionary.
        queries (Dict): The queries dictionary.
        qrels (Dict): The qrels dictionary.
        dataset (str): The dataset name.
    """
    if not is_processed(dataset):
        corpus_subset, queries_subset, qrels_subset = get_subset(corpus, queries, qrels)
        if not os.path.exists("subsets"): os.mkdir("subsets")
        if not os.path.exists(f"subsets/{dataset}"): os.mkdir(f"subsets/{dataset}")
        json.dump(corpus_subset, open(f"subsets/{dataset}/corpus.json", "w"))
        json.dump(queries_subset, open(f"subsets/{dataset}/queries.json", "w"))
        json.dump(qrels_subset, open(f"subsets/{dataset}/qrels.json", "w"))


def save_corpus_embeddings(corpus, dataset: str, encoder: CustomEncoder):
    """Save the embeddings of the corpus documents.

    Args:
        corpus (Dict): The corpus dictionary.
        dataset (str): The dataset name.
        encoder (CustomEncoder): The encoder used for encoding the documents.
    """
    i = 0       # document index, used as embedding filename
    m = {}      # mapping from doc_id to embedding filename

    logger.info(f"Saving {encoder.name} embeddings for {dataset}...")
    for doc_id, doc_dict in tqdm(corpus.items()):
        doc = join_title_text(doc_dict)
        sents = split_sentences(doc)
        embs = encoder.encode_queries(sents)
        if not os.path.exists("embeddings"): os.mkdir("embeddings")
        if not os.path.exists(f"embeddings/{dataset}"): os.mkdir(f"embeddings/{dataset}")
        if not os.path.exists(f"embeddings/{dataset}/{encoder.name}"): os.mkdir(f"embeddings/{dataset}/{encoder.name}")
        if not os.path.exists(f"embeddings/{dataset}/{encoder.name}/{str(i)}.npy"): np.save(f"embeddings/{dataset}/{encoder.name}/{str(i)}.npy", embs)
        m[doc_id] = i
        i += 1
    json.dump(m, open(f"embeddings/{dataset}/{encoder.name}/docids.json", "w"))


def load_processed(dataset: str, is_subset: bool = True):
    """Load the processed BEIR-formatted dataset.

    Args:
        dataset (str): The dataset name.
        is_subset (bool): Whether to load the subset or the full dataset.

    Returns:
        Tuple[Dict, Dict, Dict]: The processed corpus, queries, and qrels dictionaries.
    """
    folder = "subsets" if is_subset else "datasets"
    return json.load(open(f"{folder}/{dataset}/corpus.json", 'r')), json.load(open(f"{folder}/{dataset}/queries.json", 'r')), json.load(open(f"{folder}/{dataset}/qrels.json", 'r'))


def augment_corpus(corpus, n_short_per_long = 10):
    """Augment the corpus by concatenating multiple short documents into a single long document.

    Args:
        corpus (Dict): The corpus dictionary.
        n_short_per_long (int): The number of short documents to combine into a long document.

    Returns:
        Dict: The augmented corpus dictionary.
    """
    corpus_augmented = {}
    short_doc_ids, short_docs = [], []
    long_doc_ids, long_docs = [], []
    counter = 0

    # ensure randomness in the order of short documents.
    keys = list(corpus.keys())
    random.shuffle(keys)

    logger.info("Augmenting corpus...")
    for doc_id in tqdm(keys):
        try:
            doc = corpus[doc_id]
            counter += 1
            short_doc_ids.append(doc_id)
            short_docs.append(join_title_text(doc))
        except:
            continue

        # when we have n_short_per_long short documents, we create a long document
        if counter == n_short_per_long:
            long_doc_id = "|".join(short_doc_ids)
            long_doc = "\n".join(short_docs)
            long_doc_ids.append(long_doc_id)
            long_docs.append(long_doc)

            # reset cache
            short_doc_ids, short_docs = [], []
            counter = 0

    # if there are leftover short documents, create a long document with them alone.
    if counter != 0:
        long_doc_id = "|".join(short_doc_ids)
        long_doc = "\n".join(short_docs)
        long_doc_ids.append(long_doc_id)
        long_docs.append(long_doc)

    # create new corpus with long documents
    for long_doc_id, long_doc in zip(long_doc_ids, long_docs):
        corpus_augmented[long_doc_id] = {"text": long_doc}
    return corpus_augmented


def modify_qrels(qrels, corpus_augmented):
    """Modify the qrels to match the long documents in the augmented corpus.

    The input qrels matches query_ids to doc_ids and relevance scores, for the short documents.
    The output qrels matches query_ids to doc_ids and relevance scores, for the long documents.

    Args:
        qrels (Dict): The qrels dictionary.
        corpus_augmented (Dict): The augmented corpus dictionary.

    Returns:
        Dict: The modified qrels dictionary.
    """
    logger.info("Generating short-long map...")
    short_long_map = {short_doc_id: long_doc_id for long_doc_id in tqdm(corpus_augmented) for short_doc_id in long_doc_id.split("|")}

    logger.info("Modifying qrels...")
    qrels_modified = {}
    for query_id, qrel in tqdm(qrels.items()):
        qrels_modified[query_id] = {}
        for doc_id, score in qrel.items():
            if short_long_map[doc_id] not in qrels_modified[query_id]:
                qrels_modified[query_id][short_long_map[doc_id]] = score
            else:
                qrels_modified[query_id][short_long_map[doc_id]] = max(qrels_modified[query_id][short_long_map[doc_id]], score)

    return qrels_modified



###### dataloaders ######
def load_beir(dataset: str, split: str = "test"):
    """Load the dataset from BEIR.

    Args:
        dataset (str): The dataset name.
        split (str): The dataset split to load.

    Returns:
        Tuple[Dict, Dict, Dict]: The corpus, queries, and qrels dictionaries.
    """
    # load BEIR dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    # augment the corpus by concatenating multiple short documents into a single long document
    if dataset in avg_n_short_per_doc:
        corpus_augmented = augment_corpus(corpus, n_short_per_long=avg_n_short_per_doc[dataset])
    else:
        corpus_augmented = augment_corpus(corpus)

    # modify the qrels to match the long documents in the augmented corpus
    qrels_modified = modify_qrels(qrels, corpus_augmented)

    return corpus_augmented, queries, qrels_modified


def load_mldr():
    """Load the MLDR dataset.

    Returns:
        Tuple[Dict, Dict, Dict]: The corpus, queries, and qrels dictionaries.
    """
    # load MLDR dataset
    dataset = 'Shitao/MLDR'
    queries = load_dataset(dataset, 'en', split='test', trust_remote_code=True)
    corpus = load_dataset(dataset, f'corpus-en', split='corpus', trust_remote_code=True)

    # convert the dataset to the BEIR format
    corpus_beir = {doc['docid']: {'text': doc['text']} for doc in corpus}
    queries_beir = {query['query_id']: query['query'] for query in queries}
    qrels_beir = {}
    for query_info in queries:
        qrels_beir[query_info['query_id']] = {}
        for positive_passage in query_info['positive_passages']:
            qrels_beir[query_info['query_id']][positive_passage['docid']] = 1
        for negative_passage in query_info['negative_passages']:
            qrels_beir[query_info['query_id']][negative_passage['docid']] = 0

    return corpus_beir, queries_beir, qrels_beir


def load_msmarco():
    """Load the MSMARCO dataset.

    Returns:
        Tuple[Dict, Dict, Dict]: The corpus, queries, and qrels dictionaries.
    """
    logger.info("Processing msmarco queries...")
    queries = pd.read_csv("datasets/msmarco/msmarco-docdev-queries.tsv", sep='\t', names=['query_id', 'query'])
    queries_dict = dict(zip(queries['query_id'], queries['query']))
    json.dump(queries_dict, open("datasets/msmarco/queries.json", "w"))

    logger.info("Processing msmarco qrels...")
    qrels = pd.read_csv("datasets/msmarco/msmarco-docdev-qrels.tsv", sep=' ', names=['query_id', 'iter', 'doc_id', 'score'])
    qrels_dict = qrels_tsv_to_dict(qrels)
    json.dump(qrels_dict, open("datasets/msmarco/qrels.json", "w"))

    logger.info("Processing msmarco corpus...")
    corpus_dict = {}
    with open("datasets/msmarco/msmarco-docs.tsv") as f:
        for line in tqdm(f.readlines()):
            doc_id, _, _, text = line.split('\t')
            doc_id = doc_id.strip()
            text = text.strip()
            corpus_dict[doc_id] = {'text': text}
    json.dump(corpus_dict, open("datasets/msmarco/corpus.json", "w"))

    return corpus_dict, queries_dict, qrels_dict


def load_miracl():
    """Load the MIRACL dataset.

    Returns:
        Tuple[Dict, Dict, Dict]: The corpus, queries, and qrels dictionaries.
    """
    logger.info("Processing miracl queries...")
    queries = pd.read_csv('datasets/miracl/topics.miracl-v1.0-en-train.tsv', sep='\t', names=['query_id', 'query'])
    queries_dict = dict(zip(queries['query_id'], queries['query']))
    json.dump(queries_dict, open("datasets/miracl/queries.json", "w"))

    logger.info("Processing miracl qrels...")
    qrels = pd.read_csv('datasets/miracl/qrels.miracl-v1.0-en-train.tsv', sep='\t', names=['query_id', 'iter', 'doc_id', 'score'])
    qrels_dict = qrels_tsv_to_dict(qrels)
    json.dump(qrels_dict, open("datasets/miracl/qrels.json", "w"))

    logger.info("Processing miracl corpus...")
    corpus = load_dataset('miracl/miracl-corpus', 'en', trust_remote_code=True)
    corpus_dict = {d['docid']: {'title': d['title'], 'text': d['text']} for d in tqdm(corpus['train'])}
    json.dump(corpus_dict, open("datasets/miracl/corpus.json", "w"))

    # augment the corpus by concatenating multiple short documents into a single long document
    corpus_augmented = augment_corpus(corpus_dict, n_short_per_long=avg_n_short_per_doc['miracl'])

    # modify the qrels to match the long documents in the augmented corpus
    qrels_modified = modify_qrels(qrels_dict, corpus_augmented)

    return corpus_augmented, queries, qrels_modified


def load_conditionalqa():
    """Load the ConditionalQA dataset.

    Returns:
        Tuple[Dict, Dict, Dict]: The corpus, queries, and qrels dictionaries.
    """
    logger.info("Processing conditionalqa corpus...")
    documents = json.load(open('datasets/conditionalqa/documents.json','r'))
    corpus = {doc['url'].split('/')[-1]: {
        "title": doc['title'],
        "text": ' '.join([BeautifulSoup(sent, "html.parser").get_text() for sent in doc['contents']])
    } for doc in documents}
    json.dump(corpus, open("datasets/conditionalqa/corpus.json", "w"))

    logger.info("Processing conditionalqa queries and qrels...")
    qas = json.load(open('datasets/conditionalqa/train.json','r'))
    qas += json.load(open('datasets/conditionalqa/dev.json','r'))
    qas += json.load(open('datasets/conditionalqa/test_no_answer.json','r'))
    queries, qrels = {}, {}
    for qa in qas:
        doc_id = qa['url'].split('/')[-1]
        if doc_id in corpus:
            query_id = qa['id']
            queries[query_id] = qa['scenario'] + '\n' + qa['question']
            qrels[query_id] = qrels.get(query_id, {})
            qrels[query_id][doc_id] = 1
    json.dump(queries, open("datasets/conditionalqa/queries.json", "w"))
    json.dump(qrels, open("datasets/conditionalqa/qrels.json", "w"))

    return corpus, queries, qrels


def load_bioasq():
    """Load the BioASQ dataset.

    Returns:
        Tuple[Dict, Dict, Dict]: The corpus, queries, and qrels dictionaries.
    """
    logger.info("Processing bioasq corpus...")
    documents = json.load(open('datasets/bioasq/allMeSH_2020.json', 'r', encoding='Latin-1'))
    corpus = {article['pmid']: {'text': article['abstractText']} for article in tqdm(documents['articles'])}
    json.dump(corpus, open("datasets/bioasq/corpus.json", "w"))

    logger.info("Processing bioasq queries and qrels...")
    data = json.load(open('datasets/bioasq/training8b.json','r'))['questions']
    for i in range(1,6):
        data += json.load(open(f'datasets/bioasq/Task8BGoldenEnriched/8B{i}_golden.json','r'))['questions']
    queries, qrels = {}, {}
    for item in data:
        query_id = item['id']
        queries[query_id] = item['body']
        qrels[query_id] = {}
        for url in item['documents']:
            doc_id = url.split('/')[-1]
            if doc_id in corpus:
                qrels[query_id][doc_id] = 1
    json.dump(queries, open("datasets/bioasq/queries.json", "w"))
    json.dump(qrels, open("datasets/bioasq/qrels.json", "w"))

    # augment the corpus by concatenating multiple short documents into a single long document
    corpus_augmented = augment_corpus(corpus, n_short_per_long=avg_n_short_per_doc['bioasq'])

    # modify the qrels to match the long documents in the augmented corpus
    qrels_modified = modify_qrels(qrels, corpus_augmented)

    return corpus_augmented, queries, qrels_modified


def load_qasper():
    """Load the QASPER dataset.

    Returns:
        Tuple[Dict, Dict, Dict]: The corpus, queries, and qrels dictionaries.
    """
    dataset = load_dataset("allenai/qasper")
    corpus, queries, qrels = {}, {}, {}
    for split in {'train','validation','test'}:
        for sample in tqdm(dataset[split]):
            curr_doc = '\n\n'.join(['\n\n'.join(paragraph) for paragraph in sample['full_text']['paragraphs']]).strip()
            if curr_doc:
                doc_id = sample['id'] + "_d"
                query_id = sample['id'] + "_q"
                corpus[doc_id] = {"text": curr_doc}
                queries[query_id] = sample['title']
                qrels[query_id] = {doc_id: 1}
    return corpus, queries, qrels


def load_lethain():
    """Load the Lethain dataset provided by Ofer.

    Returns:
        Tuple[Dict, Dict, Dict]: The corpus, queries, and qrels dictionaries.
    """
    corpus, queries, qrels = {}, {}, {}
    for i,filename in tqdm(enumerate(os.listdir('datasets/lethain/text-files'))):
        query = filename.replace('https-infraeng-dev-','').replace('https-lethain-com-','').replace('.txt','').replace('-',' ')
        with open(f'datasets/lethain/text-files/{filename}', 'r') as f:
            text = f.read()
        queries[str(i)] = query
        corpus[str(i)] = {'text': text}
        qrels[str(i)] = {str(i): 1}
    return corpus, queries, qrels



###### MAIN FUNCTION ######
def load_data(dataset: str):
    """Load the dataset.

    Args:
        dataset (str): The dataset name.

    Returns:
        Tuple[Dict, Dict, Dict]: The corpus, queries, and qrels dictionaries.
    """
    logger.info(f"Loading {dataset}...")
    # load the processed dataset if it exists
    if is_processed(dataset):
        return load_processed(dataset)

    # load the dataset from BEIR
    elif dataset in {'nfcorpus', 'nq', 'hotpotqa', 'fiqa', 'quora', 'webis-touche2020', 'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact'}:
        return load_beir(dataset)

    # load custom dataset
    else:
        return eval(f"load_{dataset}")()


def precompute_embeddings(dataset, encoder_name, device):
    """Precompute the embeddings for the dataset using the specified encoder.

    Args:
        dataset (str): The dataset name.
        encoder_name (str): The encoder name.
        device (str): The device to use for encoding.
    """
    # if the dataset is not processed yet, process and save the subset
    if not is_processed(dataset):
        corpus, queries, qrels = load_data(dataset)
        save_subset(corpus, queries, qrels, dataset)

    # load the processed dataset
    corpus, _, _ = load_processed(dataset)

    # save the embeddings
    save_corpus_embeddings(corpus, dataset, CustomEncoder(encoder_name, device))



if __name__ == "__main__":
    # If the script is executed, precompute the embeddings for all datasets and encoders in the config file
    cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    cfgs = [dict(zip(["dataset", "encoder_name"], v)) for v in itertools.product(cfg['datasets'], cfg['encoders'])]
    Parallel(n_jobs=4)(delayed(precompute_embeddings)(**cfg, device=f"cuda:{i%4}") for i, cfg in enumerate(cfgs))