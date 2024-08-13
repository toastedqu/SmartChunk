
import numpy as np
import pandas as pd
from typing import *
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

def get_curr_min_idx(arr: np.ndarray):
    """Get the index of the minimum value in the 2d array.

    Args:
        arr (np.ndarray): The input 2d array

    Returns:
        int: The row index of the minimum value
        int: The column index of the minimum value
    """
    return np.argmin(arr)//len(arr), np.argmin(arr)%len(arr)

def group_by_cluster(texts: List[str], labels: List[int]):
    """Group texts by cluster labels.

    Args:
        texts (List[str]): The list of texts
        labels (List[int]): The list of cluster labels

    Returns:
        List[str]: The list of grouped texts
        List[int]: The list of cluster labels
    """
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
        print(id, text)
        print()

def single_linkage_clustering(embs: np.ndarray, lamda: float = 0.5, max_samples_per_cluster: Optional[int] = None):
    """Perform single linkage clustering on the input embeddings.

    Args:
        embs (np.ndarray): The input embeddings of shape (n_samples, n_features)
        lamda (float): The weight of positional distance in the weighted average
        max_samples_per_cluster (int): The maximum number of samples per cluster. If None, set to `len(embs) // 10` to ensure a reasonable balance between the number of clusters and the number of samples in each cluster.

    Returns:
        List[int]: The list of cluster labels
    """
    # get distance matrix
    dist_mat = get_dist_mat(embs, lamda=lamda)
    dist_mat[dist_mat == 0] = float('inf')  # set diagonal to infinity so that the minimum value can be iterated

    # set max_samples_per_cluster
    if not max_samples_per_cluster:
        max_samples_per_cluster = len(embs) // 10

    # initialize clusters and parents
    clusters = {i: [i] for i in range(len(embs))}
    parents = {i: i for i in range(len(embs))}

    # iterate through the distance matrix from min to max for single linkage clustering
    while dist_mat.min() < float('inf'):
        row, col = get_curr_min_idx(dist_mat)
        parent_row, parent_col = parents[row], parents[col]
        if parent_row != parent_col and len(clusters[parent_row]) + len(clusters[parent_col]) <= max_samples_per_cluster:
            clusters[parent_row].extend(clusters[parent_col])
            for sample in clusters[parent_col]:
                parents[sample] = parent_row
            del clusters[parent_col]
        dist_mat[row, col] = float('inf')

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