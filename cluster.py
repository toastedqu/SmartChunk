import numpy as np
import typing


# linkage_matrix: np.ndarray of shape (n, 4) where n is the number of non-leaf clusters
# cluster1, cluster2, distance between two clusters in this merge, size


# return the parents of each cluster, the size of each cluster, and the size of each cluster (leaves with id < filter)
def linkage_info(linkage_matrix: np.ndarray, len_embs: int, filter: int = 0) -> typing.Tuple[typing.Dict[int, int], typing.Dict[int, int], typing.Dict[int, int]]:
    parents = {}  # cluster_id -> parent_cluster_id
    sz = {}  # cluster_id -> size of cluster
    sz_filtered = {}  # cluster_id -> size of cluster (leaves with id < filter)
    for i in range(len_embs):
        sz[i] = 1
    for (i, row) in enumerate(linkage_matrix):
        cluster1, cluster2, _, size = row
        size_filtered = 0

        # if cluster is a leaf, then start to filter, otherwise, add the size of the cluster
        if cluster1 < len_embs:
            if cluster1 < filter:
                size_filtered += 1
        else:
            size_filtered += sz_filtered[cluster1]
        if cluster2 < len_embs:
            if cluster2 < filter:
                size_filtered += 1
        else:
            size_filtered += sz_filtered[cluster2]

        parents[cluster1] = i + len_embs
        parents[cluster2] = i + len_embs
        sz[i + len_embs] = size
        sz_filtered[i + len_embs] = size_filtered
    return parents, sz, sz_filtered


# return the leaves in the cluster with cluster_id
def linkage_expand_cluster(cluster_id: int, linkage_matrix: np.ndarray, len_embs: int) -> typing.Set[int]:
    if cluster_id >= len_embs:
        # expand cluster
        left = linkage_matrix[int(cluster_id) - len_embs][0]
        right = linkage_matrix[int(cluster_id) - len_embs][1]
        return linkage_expand_cluster(left, linkage_matrix, len_embs).union(linkage_expand_cluster(right, linkage_matrix, len_embs))
    else:
        # cluster is a leaf
        return set([cluster_id])


# return the leaves in the largest cluster that contains cluster_id and has at most (<=) k leaves
def linkage_nearest_k_neighborhood(cluster_id: int, linkage_matrix: np.ndarray, len_embs: int, linkage_parents: typing.Dict[int, int], linkage_size: typing.Dict[int, int], k: int = 4) -> typing.Set[int]:
    if k <= 0:
        return set()
    if k == 1:
        return set([cluster_id])
    prev_centroid = -1
    centroid = cluster_id
    while linkage_size[centroid] <= k:
        if centroid in linkage_parents:
            prev_centroid = centroid
            centroid = linkage_parents[centroid]
        else:
            # cluster is root
            prev_centroid = centroid
            break
    # now, centroid is the smallest cluster that contains cluster_id and has more than (>) k leaves
    # prev_centroid is the largest cluster that contains cluster_id and has at most (<=) k leaves
    return linkage_expand_cluster(prev_centroid, linkage_matrix, len_embs)


# return the balanced split of (at most) k clusters
def linkage_k_split(linkage_matrix: np.ndarray, len_embs: int, linkage_size: typing.Dict[int, int], k: int = 4, cluster_id: int = -1) -> typing.Set[int]:
    if cluster_id == -1:  # root
        cluster_id = len(linkage_matrix) + len_embs - 1
    if k == 1:
        return set([cluster_id])
    if k <= 0:
        return set()
    if cluster_id < len_embs:  # leaf but k > 1
        raise ValueError("k must be 1 for leaf clusters")

    left = linkage_matrix[int(cluster_id) - len_embs][0]
    right = linkage_matrix[int(cluster_id) - len_embs][1]
    left_size = linkage_size[left]
    right_size = linkage_size[right]

    # keep left be the larger cluster
    if right_size > left_size:
        left, right = right, left
        left_size, right_size = right_size, left_size

    # if able, keep left and right balanced to give distinction
    right_cluster_k = k // 2  # right cluster has floor(k/2) leaves
    left_cluster_k = k - right_cluster_k  # left cluster has ceil(k/2) leaves

    # otherwise, select all leaves in the smaller cluster
    if right_size < right_cluster_k:
        right_cluster_k = right_size
        left_cluster_k = k - right_cluster_k

    # if not possible at all, select all leaves in both clusters to give less than k splits
    if left_size < left_cluster_k:
        left_cluster_k = left_size

    left_cluster = linkage_k_split(linkage_matrix, len_embs, linkage_size, left_cluster_k, left)
    right_cluster = linkage_k_split(linkage_matrix, len_embs, linkage_size, right_cluster_k, right)
    return left_cluster.union(right_cluster)


# return the balanced split containing at least k elements each
def linkage_k_preserve(linkage_matrix: np.ndarray, len_embs: int, linkage_size: typing.Dict[int, int], k: int = 4, cluster_id: int = -1) -> typing.Set[int]:
    if cluster_id == -1:  # root
        cluster_id = len(linkage_matrix) + len_embs - 1
    if k == 1:
        return set([cluster_id])
    if k <= 0:
        return set()
    if cluster_id < len_embs:  # leaf but k > 1
        raise ValueError("k must be 1 for leaf clusters")

    left = linkage_matrix[int(cluster_id) - len_embs][0]
    right = linkage_matrix[int(cluster_id) - len_embs][1]
    left_size = linkage_size[left]
    right_size = linkage_size[right]

    # keep left be the larger cluster
    if right_size > left_size:
        left, right = right, left
        left_size, right_size = right_size, left_size

    if right_size < k:  # one side has less than k elements, select all elements that the cluster contains >= k elements
        return set([cluster_id])
    else:  # both sides have at least k elements, continue to split
        left_cluster = linkage_k_preserve(linkage_matrix, len_embs, linkage_size, k, left)
        right_cluster = linkage_k_preserve(linkage_matrix, len_embs, linkage_size, k, right)
        return left_cluster.union(right_cluster)
