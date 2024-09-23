import warnings
import spacy
import numpy as np
import pandas as pd
from utils import split_sentences
from typing import *
from tqdm import tqdm
from functools import partial
from encoder import Encoder
from scipy.spatial.distance import pdist, squareform
from langchain_community.utils.math import cosine_similarity
from sklearn.cluster import DBSCAN
nlp = spacy.load("en_core_web_sm")

###### chunkers ######
class BaseChunker:
    """
    BaseChunker is a base class for splitting documents into chunks.
    BaseChunker considers each document as a whole chunk.
    """
    def __init__(self):
        pass

    def chunk(self, doc: str | List[str], doc_id: int = 0):
        """Split the document into chunks.

        It is preferred that a document is splitted into sentences beforehand.
        If the document is not splitted, then spacy is used to split the document into sentences.

        Each chunker will only return the sentence indices in each chunk, NOT the actual sentences,
        for the convenience of storage and computation.

        Args:
            doc (str | List[str]): The document to chunk.

        Returns:
            Dict[str, List[int]]: The dictionary of chunks, where the key is the document ID and the value is the list of sentence indices.
        """
        if isinstance(doc, str):
            doc = split_sentences(doc)
        return self._convert_chunks_to_dict([list(range(len(doc)))], doc_id)

    def chunk_corpus(self, corpus: List[List[str]]):
        """Split the corpus into chunks.

        Args:
            corpus (List[List[str]]): A list of documents to chunk.

        Returns:
            List[Dict[str, List[int]]]: The list of dictionaries of chunks, where the key is the chunk ID and the value is the list of sentence indices.
        """
        return [self.chunk(doc, i) for i, doc in tqdm(enumerate(corpus))]

    def _convert_chunks_to_dict(self, chunks: List[List[int]], doc_id: int = 0):
        """Convert the list of chunks to a dictionary.

        We need a specific doc_chunk_id key for the convenience of evaluation later.

        Args:
            chunks (List[List[int]]): The list of chunks, where each chunk is a list of sentence indices.
            doc_id (int): The document ID.

        Returns:
            Dict[str, List[int]]: The dictionary of chunks, where the key is the chunk ID and the value is the list of sentence indices.
        """
        return {f"{doc_id}|{i}": chunk for i, chunk in enumerate(chunks)}



class PositionalChunker(BaseChunker):
    """
    PositionalChunker splits documents into consecutive chunks of consecutive sentences.

    Attributes:
        n_chunks (Optional[int]): The number of chunks to split the document into.
        n_sents_per_chunk (Optional[int]): The number of sentences per chunk.
            Note: n_sents_per_chunk and n_chunks are mutually exclusive.
            If both are provided, n_sents_per_chunk will be ignored.
            n_chunks is only an approximation and may not be exact.
        n_sents_overlap (int): The number of sentences to overlap between chunks.


    """
    def __init__(self,
                 n_chunks: Optional[int] = None,
                 n_sents_per_chunk: Optional[int] = None,
                 n_sents_overlap: int = 0):
        if not n_sents_per_chunk and not n_chunks:
            raise ValueError("Either n_sents_per_chunk or n_chunks should be provided.")
        self.n_chunks = n_chunks
        self.n_sents_per_chunk = n_sents_per_chunk
        self.n_sents_overlap = n_sents_overlap

    def chunk(self, doc: str | List[str], doc_id: int = 0):
        if self.n_chunks:
            if self.n_chunks >= len(doc):
                warnings.warn(f"n_chunks ({self.n_chunks}) is greater than or equal to the number of sentences in the document ({len(doc)}).\nUsing the number of sentences as the number of chunks.")
                n_sents_per_chunk = 1
            else:
                n_sents_per_chunk = len(doc) // self.n_chunks
                n_sents_overlap = self.n_sents_overlap
        else:
            n_sents_per_chunk = self.n_sents_per_chunk
            n_sents_overlap = self.n_sents_overlap
        if n_sents_per_chunk == 1:
            n_sents_overlap = 0
        assert n_sents_per_chunk > n_sents_overlap, "n_sents_per_chunk should be greater than n_sents_overlap"

        sent_ids = list(range(len(doc)))
        end = len(sent_ids)-n_sents_overlap
        step = n_sents_per_chunk-n_sents_overlap
        chunks = [sent_ids[i:i+n_sents_per_chunk] for i in range(0, end, step)]
        return self._convert_chunks_to_dict(chunks, doc_id)



class SingleLinkageChunker(BaseChunker):
    """
    SingleLinkageChunker splits documents into chunks using single linkage clustering.
    Unlike most other chunkers, SingleLinkageChunker does NOT require consecutive sentences to be in the same chunk.
    Nonetheless, positional distance is considered in the clustering process.

    Attributes:
        lamda (float): The weight of positional distance.
        n_clusters (Optional[int]): The number of clusters to split the document into.
        max_samples_per_cluster (Optional[int]): The maximum number of sentences per cluster.
            Note: max_samples_per_cluster and n_clusters are mutually exclusive.
            If both are provided, max_samples_per_cluster will be ignored.
            n_clusters is only an approximation and may not be exact.
        distance_threshold (float): The distance threshold for merging clusters.
        dataset (str): The name of the dataset for precomputed embeddings.
    """
    def __init__(self,
                 lamda: float = 0.5,
                 n_clusters: Optional[int] = None,
                 max_samples_per_cluster: Optional[int] = None,
                 distance_threshold: Optional[float] = 0.5,
                 dataset: Optional[str] = None):
        if not max_samples_per_cluster and not n_clusters:
            raise ValueError("Either max_samples_per_cluster or n_clusters should be provided.")
        self.lamda = lamda
        self.n_clusters = n_clusters
        self.max_samples_per_cluster = max_samples_per_cluster
        self.distance_threshold = distance_threshold
        self.dataset = dataset
        if self.dataset is None:
            # only load the encoder if the dataset is not provided
            # this avoids unnecessary OOM
            self.encoder = Encoder()

    def chunk(self, doc: List[str], doc_id: int = 0):
        if self.dataset is None:
            embs = self.encoder.encode(doc)
        else:
            embs = np.load(f'embeddings/{self.dataset}/doc_{doc_id}.npy')

        assert len(embs) == len(doc), "The number of embeddings should be equal to the number of sentences in the document."

        labels = self._single_linkage_clustering(embs)
        clusters = self._group_by_cluster(list(range(len(doc))), labels)
        return self._convert_chunks_to_dict(clusters, doc_id)

    def _convert_chunks_to_dict(self, clusters: Dict[int, List[int]], doc_id: int = 0):
        return {f"{doc_id}|{k}": v for k, v in clusters.items()}

    def _get_dist(self,
                  A: np.ndarray,
                  B: np.ndarray,
                  n_segments: int,
                  lamda: float = 0.5) -> float:
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
        cos_sim = np.dot(A_vec, B_vec) / (np.linalg.norm(A_vec) * np.linalg.norm(B_vec))    # float [-1, 1]
        cos_sim = max(0, cos_sim)                                   # float [0, 1]
        cos_dist = 1 - cos_sim                                      # float [0, 1]

        # calculate weighted average
        return lamda * pos_dist_norm + (1-lamda) * cos_dist         # float [0, 1]

    def _get_dist_mat(self, data: np.ndarray, lamda: float = 0.5) -> np.ndarray:
        """Get the distance matrix of the input data.

        Args:
            data (np.ndarray): The input data of shape (n_samples, n_features)
            lamda (float): The weight of positional distance in the weighted average

        Returns:
            np.ndarray: The upper triangular distance matrix of shape (n_samples, n_samples)
        """
        return np.triu(squareform(np.abs(pdist(data, metric=partial(self._get_dist, n_segments=len(data), lamda=lamda)))))

    def _group_by_cluster(self, sent_ids: List[str], labels: List[int]):
        """Group texts by cluster labels.

        Args:
            texts (List[str]): The list of texts
            labels (List[int]): The list of cluster labels

        Returns:
            List[str]: The list of grouped texts
            List[int]: The list of cluster labels
        """
        df = pd.DataFrame({"label": labels, "sent_ids": sent_ids})
        df = df.groupby("label")["sent_ids"].apply(list).reset_index()
        return df.set_index("label")["sent_ids"].to_dict()

    def _sort_distance_indices(self, distance_matrix: np.ndarray):
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

    def _single_linkage_clustering(self, embs: np.ndarray):
        """Perform single linkage clustering on the input embeddings.

        Args:
            embs (np.ndarray): The input embeddings of shape (n_samples, n_features)
            lamda (float): The weight of positional distance in the weighted average
            max_samples_per_cluster (int): The maximum number of samples per cluster. If None, set to `len(embs) // 10` to ensure a reasonable balance between the number of clusters and the number of samples in each cluster.
            distance_threshold (float): The distance threshold for merging clusters

        Returns:
            List[int]: The list of cluster labels
        """
        # confirm max_samples_per_cluster
        if self.n_clusters:
            if self.n_clusters >= len(embs):
                warnings.warn(f"n_clusters ({self.n_clusters}) is greater than or equal to the number of sentences in the document ({len(embs)}).\nUsing the number of sentences as the number of clusters.")
                max_samples_per_cluster = 1
            else:
                max_samples_per_cluster = len(embs) // self.n_clusters
        else:
            max_samples_per_cluster = self.max_samples_per_cluster

        # init consts
        sent_ids = list(range(len(embs)))

        # prepend positional information to the embeddings
        if self.lamda > 0:
            embs = np.concatenate([np.array(sent_ids).reshape(-1, 1), embs], axis=1)

        # get distance matrix
        dist_mat = self._get_dist_mat(embs, lamda=self.lamda)

        # get sorted distance indices as tuples for iteration
        sorted_distance_tuples = self._sort_distance_indices(dist_mat)
        tups = []
        for tup in sorted_distance_tuples:
            if tup[0] <= self.distance_threshold:    # only consider distances below the threshold
                tups.append(tup)
            else:
                break

        # initialize clusters and parents
        clusters = {i: [i] for i in sent_ids}
        parents = {i: i for i in sent_ids}

        # merge two clusters in the ascending order of distance
        for _, (row, col) in sorted_distance_tuples:
            parent_row, parent_col = parents[row], parents[col]
            if parent_row != parent_col and len(clusters[parent_row])+len(clusters[parent_col]) <= max_samples_per_cluster:
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
        cluster_labels = [parent_to_cluster_label[parents[sent_id]] for sent_id in sent_ids]
        return cluster_labels


class DBSCANChunker(SingleLinkageChunker):
    def __init__(self,
                 lamda: float = 0.5,
                 eps: float = 0.2,
                 min_samples_per_cluster: int = 5,
                 dataset: str = None):
        self.lamda = lamda
        self.eps = eps
        self.min_samples_per_cluster = min_samples_per_cluster
        self.dataset = dataset
        if self.dataset is None:
            self.encoder = Encoder()

    def chunk(self, doc: List[str], doc_id: int = 0):
        if self.dataset is None:
            embs = self.encoder.encode(doc)
        else:
            embs = np.load(f'embeddings/{self.dataset}/doc_{doc_id}.npy')

        if self.min_samples_per_cluster > len(embs)//2:
            warnings.warn(f"min_samples_per_cluster ({self.min_samples_per_cluster}) is greater than half of the number of sentences in the document ({len(embs)}).\nReduced it to half.")
            min_samples_per_cluster = len(embs)//2
        else:
            min_samples_per_cluster = self.min_samples_per_cluster

        labels = DBSCAN(eps=self.eps,
                        min_samples=min_samples_per_cluster,
                        metric=partial(self._get_dist, n_segments=len(embs), lamda=self.lamda)).fit(embs).labels_
        if -1 in labels:
            labels += 1
        clusters = self._group_by_cluster(list(range(len(doc))), labels)
        return self._convert_chunks_to_dict(clusters, doc_id)

class LangchainChunker(BaseChunker):
    """
    The following code is slightly modified from Langchain's source code for SemanticChunker:

    https://api.python.langchain.com/en/latest/_modules/langchain_experimental/text_splitter.html#SemanticChunker

    The modifications include:
    - Removing the `number_of_chunks` parameter
    - Removing the `threshold_from_clusters` method
    - Removing the regex-based text splitting method
    - Added precomputed embeddings for the dataset
    - Added presplitted texts as input
    """
    def __init__(self,
                 breakpoint_threshold_type: str = "percentile",
                 breakpoint_threshold_amount: float = 50,
                 dataset: str = None):
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.dataset = dataset
        if self.dataset is None:
            self.encoder = Encoder()

    def _combine_sentences(self, sentences: List[dict], buffer_size: int = 1) -> List[dict]:
        """Combine sentences based_calculate_sentence_distances on buffer size.

        Args:
            sentences: List of sentences to combine.
            buffer_size: Number of sentences to combine. Defaults to 1.

        Returns:
            List of sentences with combined sentences.
        """
        # Go through each sentence dict
        for i in range(len(sentences)):
            # Create a string that will hold the sentences which are joined
            combined_sentence = ""

            # Add sentences before the current one, based on the buffer size.
            for j in range(i - buffer_size, i):
                # Check if the index j is not negative
                # (to avoid index out of range like on the first one)
                if j >= 0:
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += sentences[j]["sentence"] + " "

            # Add the current sentence
            combined_sentence += sentences[i]["sentence"]

            # Add sentences after the current one, based on the buffer size
            for j in range(i + 1, i + 1 + buffer_size):
                # Check if the index j is within the range of the sentences list
                if j < len(sentences):
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += " " + sentences[j]["sentence"]

            # Then add the whole thing to your dict
            # Store the combined sentence in the current sentence dict
            sentences[i]["combined_sentence"] = combined_sentence

        return sentences

    def _calculate_cosine_distances(self, sentences: List[dict]) -> Tuple[List[float], List[dict]]:
        """Calculate cosine distances between sentences.

        Args:
            sentences: List of sentences to calculate distances for.

        Returns:
            Tuple of distances and sentences.
        """
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]["combined_sentence_embedding"]
            embedding_next = sentences[i + 1]["combined_sentence_embedding"]

            # Calculate cosine similarity
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the list
            distances.append(distance)

            # Store distance in the dictionary
            sentences[i]["distance_to_next"] = distance

        # Optionally handle the last sentence
        # sentences[-1]['distance_to_next'] = None  # or a default value

        return distances, sentences

    def _calculate_breakpoint_threshold(
        self, distances: List[float]
    ) -> Tuple[float, List[float]]:
        if self.breakpoint_threshold_type == "percentile":
            return cast(
                float,
                np.percentile(distances, self.breakpoint_threshold_amount),
            ), distances
        elif self.breakpoint_threshold_type == "standard_deviation":
            return cast(
                float,
                np.mean(distances)
                + self.breakpoint_threshold_amount * np.std(distances),
            ), distances
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1
            return np.mean(
                distances
            ) + self.breakpoint_threshold_amount * iqr, distances
        elif self.breakpoint_threshold_type == "gradient":
            # Calculate the threshold based on the distribution of gradient of distance array. # noqa: E501
            if len(distances) == 1:
                return 0.0, distances
            distance_gradient = np.gradient(distances, range(0, len(distances)))
            return cast(
                float,
                np.percentile(distance_gradient, self.breakpoint_threshold_amount),
            ), distance_gradient
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{self.breakpoint_threshold_type}"
            )

    # def _threshold_from_clusters(self, distances: List[float]) -> float:
    #     """
    #     Calculate the threshold based on the number of chunks.
    #     Inverse of percentile method.
    #     """
    #     if self.number_of_chunks is None:
    #         raise ValueError(
    #             "This should never be called if `number_of_chunks` is None."
    #         )
    #     x1, y1 = len(distances), 0.0
    #     x2, y2 = 1.0, 100.0

    #     x = max(min(self.number_of_chunks, x1), x2)

    #     # Linear interpolation formula
    #     y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)
    #     y = min(max(y, 0), 100)

    #     return cast(float, np.percentile(distances, y))

    def _calculate_sentence_distances(
            self, single_sentences_list: List[str]
        ) -> Tuple[List[float], List[dict]]:
        """Split text into multiple components."""

        _sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]
        sentences = self._combine_sentences(_sentences)
        if self.dataset is not None:
            embeddings = self.embs
        else:
            embeddings = self.encoder.encode(
                [x["combined_sentence"] for x in sentences]
            )

        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i]

        return self._calculate_cosine_distances(sentences)

    def split_text(
        self,
        single_sentences_list: List[str],
    ) -> List[str]:
        # having len(single_sentences_list) == 1 would cause the following
        # np.percentile to fail.
        if len(single_sentences_list) == 1:
            return single_sentences_list
        distances, sentences = self._calculate_sentence_distances(single_sentences_list)
        # if self.number_of_chunks is not None:
        #     breakpoint_distance_threshold = self._threshold_from_clusters(distances)
        #     breakpoint_array = distances
        # else:
        (
            breakpoint_distance_threshold,
            breakpoint_array,
        ) = self._calculate_breakpoint_threshold(distances)

        indices_above_thresh = [
            i
            for i, x in enumerate(breakpoint_array)
            if x > breakpoint_distance_threshold
        ]

        chunks = []
        start_index = 0

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index : end_index + 1]
            combined_text = [d["index"] for d in group]
            chunks.append(combined_text)

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = [d["index"] for d in sentences[start_index:]]
            chunks.append(combined_text)
        return chunks

    def chunk(self, doc: List[str], doc_id: int = 0):
        if self.dataset is None:
            self.embs = self.encoder.encode(doc)
        else:
            self.embs = np.load(f'embeddings/{self.dataset}/doc_{doc_id}.npy')

        chunks = self.split_text(doc)
        return self._convert_chunks_to_dict(chunks, doc_id)


class AbsoluteLangchainChunker(LangchainChunker):
    def __init__(self,
                 breakpoint_threshold_type: str = "distance",
                 breakpoint_threshold_amount: float = 0.5,
                 dataset: str = None):
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.dataset = dataset
        if self.dataset is None:
            self.encoder = Encoder()

    def _calculate_breakpoint_threshold(
        self, distances: List[float]
    ) -> Tuple[float, List[float]]:
        if self.breakpoint_threshold_type == "distance":
            return cast(
                float,
                self.breakpoint_threshold_amount,
            ), distances
        elif self.breakpoint_threshold_type == "gradient":
            if len(distances) == 1:
                return 0.0, distances
            distance_gradient = np.abs(np.gradient(distances, range(0, len(distances))))
            return cast(
                float,
                self.breakpoint_threshold_amount,
            ), distance_gradient
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{self.breakpoint_threshold_type}"
            )