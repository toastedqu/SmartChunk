import yaml
import spacy
import warnings
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
from functools import partial
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# load config
cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
model = SentenceTransformer(cfg["encoder_model"])
nlp = spacy.load(cfg["spacy_model"])

###### helper functions ######
def __dist__(A: np.ndarray, B: np.ndarray, n_segments: int, lamda: float = 0.5) -> float:
    """Calculate the distance between two embeddings.

    Args:
        A (np.ndarray): Embedding A, with index 0 as the position of the sentence
        B (np.ndarray): Embedding B, with index 0 as the position of the sentence
        n_segments (int): The maximum length of the document
        lamda (float): The weight of position distance between 0 and 1 (0 for cosine distance only, 1 for position distance only)

    Returns:
        float: The distance between two embeddings
    """
    # calculate normalized positional distance
    if lamda > 0:                                               # include position
        A_pos, B_pos = int(A[0]), int(B[0])
        pos_dist = abs(A_pos - B_pos)                           # integer [1, n_segments]
        pos_dist_norm = pos_dist / n_segments                   # float [0, 1]
        A_vec, B_vec = A[1:], B[1:]
    else:                                                       # exclude position
        pos_dist_norm = 0
        A_vec, B_vec = A, B

    # calculate normalized cosine distance
    cos_sim = np.dot(A_vec, B_vec) / (np.linalg.norm(A_vec) * np.linalg.norm(B_vec))
    cos_dist = 1 - cos_sim                                      # float [0, 2]
    cos_dist_norm = cos_dist / 2                                # float [0, 1]

    # calculate weighted average
    return lamda * pos_dist_norm + (1-lamda) * cos_dist_norm    # float [0, 1]

def single_linkage_clustering(data: np.ndarray,
                              lamda: int = 0,
                              mode: Literal["k-split", "k-preserve"] = "k-split",
                              n_clusters: Optional[int] = None,
                              min_samples: Optional[int] = None):
    """
    Perform single-linkage clustering on the given data.

    Args:
        data (np.ndarray): A 2D numpy array where each row represents a data point.
        lamda (float): The weight of position distance between 0 and 1 (0 for cosine distance only, 1 for position distance only)
        mode (str):
            - "k-split": Split the document into k chunks
            - "k-preserve": Split the document into chunks of minimal size k
        n_clusters (int): The number of clusters to form in "k-split" mode
        min_samples (int): The minimum number of samples in a cluster in "k-preserve" mode

    Returns:
        np.ndarray: An array of cluster labels for each data point.
    """
    # validate input
    if mode == "k-split" and n_clusters is None:
        raise ValueError("n_clusters must be specified in 'k-split' mode.")
    if mode == "k-preserve":
        if min_samples is None:
            msg = "'min_samples' is not specified in 'k-preserve' mode. Will use default min_samples instead."
            warnings.warn(msg, UserWarning)
            min_samples = len(data) // 10 + 1
        if min_samples > len(data)/2:
            msg = "'min_samples' must be less than half the number of sentences. Will use default min_samples instead."
            warnings.warn(msg, UserWarning)
            min_samples = len(data) // 10 + 1
        n_clusters = len(data)      # k-preserve initializes with total number of segments as n_clusters

    # perform single-linkage clustering
    metric = partial(__dist__, n_segments=len(data), lamda=lamda)
    distance_matrix = np.abs(pdist(data, metric=metric))
    linkage_matrix = linkage(distance_matrix, method='single')
    cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

    # k-split ends here
    if mode == "k-split":
        return cluster_labels

    # k-preserve continues here
    if mode == "k-preserve":
        # iteratively merge small clusters with the closest cluster
        while True:
            # find the small clusters with less than min_samples
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            small_clusters = unique_labels[counts < min_samples]
            small_counts = counts[counts < min_samples]

            # if no small clusters, break
            if len(small_clusters) == 0:
                break

            # find the smallest cluster to merge
            smallest_cluster = small_clusters[np.argmin(small_counts)]
            smallest_indices = np.where(cluster_labels == smallest_cluster)[0]

            # find the closest cluster to merge with
            min_distance = np.inf
            closest_cluster = None
            distance_square = squareform(distance_matrix)
            for i in smallest_indices:
                distances = distance_square[i]
                for lbl in unique_labels:
                    if lbl != smallest_cluster:
                        cluster_indices = np.where(cluster_labels == lbl)[0]
                        mean_distance = np.mean(distances[cluster_indices])
                        if mean_distance < min_distance:
                            min_distance = mean_distance
                            closest_cluster = lbl

            # merge the smallest cluster with the closest cluster
            cluster_labels[cluster_labels == smallest_cluster] = closest_cluster

        # relabel clusters consecutively
        unique_labels = np.unique(cluster_labels)
        new_labels = np.zeros_like(cluster_labels)
        for new_label, unique_label in enumerate(unique_labels, start=1):
            new_labels[cluster_labels == unique_label] = new_label

        return new_labels

def group_by_cluster(texts, labels):
    """Group the texts by their cluster labels.

    Args:
        texts (List[str]): The list of texts
        labels (List[int]): The list of cluster labels

    Returns:
        List[str]: The list of texts grouped by their cluster labels
        List[int]: The list of cluster labels
    """
    df = pd.DataFrame({"label": labels, "texts": texts})
    df = df.groupby("label")["texts"].apply(lambda x: '\n'.join(x)).reset_index()
    return df["texts"].tolist(), df["label"].tolist()

def log_clusters(text_id_tuples):
    """Log the cluster information.

    Args:
        texts (List[str]): The list of texts
        ids (List[str]): The list of IDs
    """
    texts, ids = text_id_tuples
    for text, id in zip(texts, ids):
        print(id, text)
        print()

###### chunkers ######
def whole_chunker(doc: str, doc_id: str):
    """Chunk the document into a single chunk.

    Args:
        doc (str): The document text
        doc_id (str): The document ID

    Returns:
        List[str]: The list of chunked texts
        List[str]: The list of chunked IDs
    """
    return [doc], [doc_id + "|0"]

def sentence_chunker(doc: str, doc_id: str, k = 1):
    """Chunk the document into chunks with k consecutive sentences.

    Args:
        doc (str): The document text
        doc_id (str): The document ID
        k (int): The number of consecutive sentences in a chunk

    Returns:
        List[str]: The list of chunked texts
        List[str]: The list of chunked IDs
    """
    sents = [sent.text for sent in nlp(doc).sents]

    # if there are less than k sentences, return the whole chunk
    if len(sents) < k:
        return [doc], [doc_id + "|0"]

    # chunk the sentences
    sents = [sents[i:i+k] for i in range(0, len(sents), k)]

    # join the sentences
    texts = [" ".join(chunk) for chunk in sents]

    # get IDs with both doc_id and chunk_id
    ids = ["{}|{}".format(doc_id, chunk_id) for chunk_id,_ in enumerate(sents)]

    return texts, ids

def word_chunker(doc: str, doc_id: str, k = 10):
    """Chunk the document into chunks with k consecutive words.

    Args:
        doc (str): The document text
        doc_id (str): The document ID
        k (int): The number of consecutive words in a chunk

    Returns:
        List[str]: The list of chunked texts
        List[str]: The list of chunked IDs
    """
    words = doc.split()

    # if there are less than k words, return the whole chunk
    if len(words) < k:
        return [doc], [doc_id + "|0"]

    # chunk the words
    words = [words[i:i+k] for i in range(0, len(words), k)]

    # join the words
    texts = [" ".join(chunk) for chunk in words]

    # get IDs with both doc_id and chunk_id
    ids = ["{}|{}".format(doc_id, chunk_id) for chunk_id,_ in enumerate(words)]

    return texts, ids

def cluster_chunker(doc: str,
                    doc_id: str,
                    n_sents_per_segment: int = 1,
                    lamda: float = 0.5,
                    mode: Literal["k-split", "k-preserve"] = "k-split",
                    n_clusters: Optional[int] = None,
                    min_samples: Optional[int] = None):
    """Chunk the document into k clusters.

    Args:
        doc (str): The document text
        doc_id (str): The document ID
        n_sents_per_segment (int): The number of sentences per segment. Clustering is performed on segments
        lamda (float): The weight of position distance between 0 and 1 (0 for cosine distance only, 1 for position distance only)
        mode (str):
            - "k-split": Split the document into k chunks
            - "k-preserve": Split the document into chunks of minimal size k
        n_clusters (int): The number of clusters to form in "k-split" mode
        min_samples (int): The minimum number of samples in a cluster in "k-preserve" mode

    Returns:
        List[str]: The list of chunked texts
        List[str]: The list of chunked IDs
    """
    sents = [sent.text for sent in nlp(doc).sents]
    if n_sents_per_segment > 1:
        sents = [''.join(sents[i:i+n_sents_per_segment]) for i in range(0, len(sents), n_sents_per_segment)]

    # if there is only one segment, return the whole chunk
    if len(sents) == 1:
        return whole_chunker(doc, doc_id)

    # validate input
    if mode == "k-split":
        if n_clusters is None:
            raise ValueError("n_clusters must be specified in 'k-split' mode.")
        if len(sents) < n_clusters:
            msg = "The number of segments is less than the number of clusters. Will use sentence_chunker instead."
            warnings.warn(msg, UserWarning)
            return sentence_chunker(doc, doc_id, k=1)

    # get embeddings
    embs = model.encode(sents)
    if lamda > 0:
        embs = np.insert(embs, 0, range(len(embs)), axis=1)     # insert position in front of semantic embeddings

    # perform single-linkage clustering
    labels = single_linkage_clustering(embs, lamda=lamda, mode=mode, n_clusters=n_clusters, min_samples=min_samples)

    # group texts by cluster labels
    texts, ids = group_by_cluster(sents, labels)
    ids = ["{}|{}".format(doc_id, cluster_id) for cluster_id in ids]

    return texts, ids

def chunk_corpus(corpus: Dict[str, Dict[str, str]],
                 chunker: Literal["whole_chunker", "sentence_chunker", "word_chunker", "cluster_chunker"],
                 **kwargs) -> Dict[str, Dict[str, str]]:
    """Chunk the given corpus using the specified chunker.

    Args:
        corpus (dict): A dictionary of documents.
        chunker (str): The chunker function to use.

    Returns:
        chunked_corpus (dict): A dictionary of chunked documents.
    """
    print(f"Chunking corpus with {chunker}...")
    chunker = eval(chunker)
    chunked_corpus = {}
    for doc_id, doc in tqdm(corpus.items()):
        # process text from BEIR-format data
        text = (doc["title"] + ".\n" + doc["text"]).strip() if "title" in doc else doc["text"].strip()

        # get chunks
        chunks, chunk_ids = chunker(text, doc_id, **kwargs)

        # process chunks into BEIR format
        for chunk_id, chunk in zip(chunk_ids, chunks):
            chunked_corpus[chunk_id] = {"text": chunk}

    return chunked_corpus