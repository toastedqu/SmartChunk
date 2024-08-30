
import numpy as np
import pandas as pd
from typing import *
from utils import *
from functools import partial
from scipy.spatial.distance import pdist, squareform

def get_dist(A: np.ndarray, B: np.ndarray, n_segments: int, lamda: float = 0.5) -> float:
    """Calculate the weighted average of positional and cosine distance between two embeddings.

    Args:
        A (np.ndarray): The first embedding
        B (np.ndarray): The second embedding
        n_segments (int): The number of segments in the document
        lamda (float): The weight of positional distance in the weighted average

    Returns:
        float: The weighted average of positional and cosine distance
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


def get_dist_mat(data: np.ndarray, lamda: float = 0.5) -> np.ndarray:
    """Get the distance matrix of the input data.

    Args:
        data (np.ndarray): The input data of shape (n_samples, n_features)
        lamda (float): The weight of positional distance in the weighted average

    Returns:
        np.ndarray: The upper triangular distance matrix of shape (n_samples, n_samples)
    """
    return np.triu(squareform(np.abs(pdist(data, metric=partial(get_dist, n_segments=len(data), lamda=lamda)))))


def group_by_cluster(texts: List[str], labels: List[int]):
    """Group texts by cluster labels.

    Args:
        texts (List[str]): The list of texts
        labels (List[int]): The list of cluster labels

    Returns:
        List[str]: The list of grouped texts
        List[int]: The list of cluster labels
    """
    # In rare cases, labels are numpy arrays.
    # Convert them to lists for consistency.
    if type(labels) == np.ndarray:
        labels = labels.tolist()

    # In rare cases, the number of labels does not match the number of texts.
    # Pad or truncate the labels to match the number of texts.
    while len(labels) < len(texts):
        labels.append(labels[-1])
    while len(labels) > len(texts):
        labels.pop()

    df = pd.DataFrame({"label": labels, "texts": texts})
    df = df.groupby("label")["texts"].apply(lambda x: '\n'.join(x)).reset_index()
    return df["texts"].tolist(), df["label"].tolist()


def log_clusters(texts, ids):
    """Log the cluster information.

    Args:
        texts (List[str]): The list of texts
        ids (List[str]): The list of IDs
    """
    for text, id in zip(texts, ids):
        text = '\n'.join(split_sentences(text))
        print(id, text)
        print()


def sort_distance_indices(distance_matrix: np.ndarray):
    """Sort the distance matrix indices in ascending order.

    Args:
        distance_matrix (np.ndarray): The distance matrix of shape (n_samples, n_samples)

    Returns:
        List[Tuple[float, Tuple[int, int]]]: The sorted list of distance tuples.
            The first element is the distance.
            The second element is the pair of corresponding sentence indices.
    """
    indices = np.triu_indices_from(distance_matrix, k=1)
    return sorted(list(zip(distance_matrix[indices], zip(indices[0], indices[1]))))


def single_linkage_clustering(embs: np.ndarray,
                              lamda: float = 0.5,
                              max_samples_per_cluster: Optional[int] = None,
                              distance_threshold: Optional[float] = 0.5):
    """Perform single linkage clustering on the input embeddings.

    Args:
        embs (np.ndarray): The input embeddings of shape (n_samples, n_features)
        lamda (float): The weight of positional distance in the weighted average
        max_samples_per_cluster (int): The maximum number of samples per cluster. If None, set to `len(embs) // 10` to ensure a reasonable balance between the number of clusters and the number of samples in each cluster.
        distance_threshold (float): The distance threshold for merging clusters

    Returns:
        List[int]: The list of cluster labels
    """
    # set max_samples_per_cluster
    if not max_samples_per_cluster: max_samples_per_cluster = len(embs) // 10

    # get distance matrix
    dist_mat = get_dist_mat(embs, lamda=lamda)
    sorted_distance_tuples = sort_distance_indices(dist_mat)
    tups = []
    for tup in sorted_distance_tuples:
        if tup[0] <= distance_threshold:    # only consider distances below the threshold
            tups.append(tup)
        else:
            break

    # initialize clusters and parents
    clusters = {i: [i] for i in range(len(embs))}
    parents = {i: i for i in range(len(embs))}

    for _, (row, col) in sorted_distance_tuples:
        parent_row, parent_col = parents[row], parents[col]
        if parent_row != parent_col and len(clusters[parent_row]) + len(clusters[parent_col]) <= max_samples_per_cluster:
            clusters[parent_row].extend(clusters[parent_col])
            for sample in clusters[parent_col]:
                parents[sample] = parent_row
            del clusters[parent_col]

    # convert parents to cluster labels
    parent_to_cluster_label = {}
    temp_lbl = 0
    for parent in parents.values():
        if parent not in parent_to_cluster_label:
            parent_to_cluster_label[parent] = temp_lbl
            temp_lbl += 1

    # get cluster labels for all samples
    cluster_labels = [parent_to_cluster_label[parents[idx]] for idx in range(len(embs))]

    return cluster_labels