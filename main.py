from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from scipy.cluster.hierarchy import linkage
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pandas as pd
import datasets
import spacy

from cluster import linkage_info, linkage_k_split, linkage_expand_cluster

import logging
import pathlib, os

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# download and unzip the dataset
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

# load and embed the dataset
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

if os.path.exists("corpus.parquet") and os.path.exists("queries.parquet"):
    corpus_pd = pd.read_parquet("corpus.parquet")
    queries_pd = pd.read_parquet("queries.parquet")
else:
    corpus_pd = pd.DataFrame.from_dict(corpus).T
    queries_pd = pd.DataFrame(queries, index=["text"]).T
    corpus_pd.index = corpus_pd.index.set_names("id")
    queries_pd.index = queries_pd.index.set_names("id")
    corpus_pd["origin"] = "corpus"
    queries_pd["origin"] = "queries"

    corpus_pd.to_parquet("corpus.parquet")
    queries_pd.to_parquet("queries.parquet")

# , model_kwargs={'device': 'cpu'}  # for CPU usage
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-distilbert-base-tas-b")
nlp = spacy.load("en_core_web_sm")

corpus_ds = datasets.Dataset.from_pandas(corpus_pd)
corpus_len = len(corpus_ds)

if os.path.exists("corpus_ds_chunked.hf"):
    corpus_ds_chunked = datasets.load_from_disk("corpus_ds_chunked.hf")
else:
    def chunker(row):
        doc = row["text"][0]
        title = row["title"][0]
        id = row["id"][0]
        texts = [doc]
        titles = [title]
        origins = [row["origin"][0]]
        ids = [id]
        sents = [sent.text for sent in nlp(doc).sents]
        candidate_emb = embeddings.embed_documents(sents)
        # candidate_emb = np.stack(candidate_emb, axis=0)
        linkage_matrix = linkage(candidate_emb, method="single", metric="cosine")
        linkage_parents, linkage_size, _ = linkage_info(linkage_matrix, len(sents))
        # for every document, use clusters of sentence to find 4-split of the document
        k_split_idx = linkage_k_split(linkage_matrix, len(sents), linkage_size, k=4)  # indices of sentences in the split
        for cluster_id in k_split_idx:
            leaf_ids = linkage_expand_cluster(cluster_id, linkage_matrix, len(sents), linkage_parents)
            cluster = [sents[int(i)] for i in leaf_ids]
            texts.append(" ".join(cluster))
            titles.append(title)
            origins.append("cs" + str(row["id"]))  # cs = corpus split
            ids.append("{}s{}".format(id, int(cluster_id)))  # s = split
        return {"text": texts, "title": titles, "origin": origins, "id": ids}

    corpus_ds_chunked = corpus_ds.map(chunker, batched=True, batch_size=1)  # chunk the documents into 4-splits
    corpus_ds_chunked.save_to_disk("corpus_ds_chunked.hf")

# filter out the original documents, only keep the split documents
corpus_ds_split = corpus_ds_chunked.filter(lambda row: row["origin"] != "corpus")

# # for debug: only use the first 10 documents
# corpus_ds_split = corpus_ds_split.select(list(range(10)))

corpus_split = {row["id"]: {"text": row["text"], "title": row["title"], "origin": row["origin"]} for row in corpus_ds_split.to_list()}

# # for debug: only use the first 10 queries
# queries_pd = queries_pd.head(10)
# queries = {row["id"]: row["text"] for row in queries_pd.reset_index().to_dict("records")}


# load model and retrieve
model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16)
retriever = EvaluateRetrieval(model, score_function="dot")
results = retriever.retrieve(corpus_split, queries)

# re-construct the results to map and reweight back to the original documents
original_results = {}
for query_id, query_result in results.items():
    original_results[query_id] = {}
    for doc_id, score in query_result.items():
        original_doc_id = doc_id.split("s")[0]
        if original_doc_id not in original_results[query_id]:
            original_results[query_id][original_doc_id] = score
        else:
            original_results[query_id][original_doc_id] += score

# evaluate NDCG@k, MAP@K, Recall@K, and Precision@K where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, original_results, retriever.k_values)

print("Evaluation Results:")
print("NDCG@k:", ndcg)
print("MAP@k:", _map)
print("Recall@k:", recall)
print("Precision@k:", precision)
