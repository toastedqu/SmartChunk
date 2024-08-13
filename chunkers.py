from utils import *
from clusters import *
from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker

###### chunkers ######
def whole_chunker(doc: str, doc_id: str):
    """Chunk the document into a single chunk.

    Args:
        doc (str): The document text
        doc_id (str): The document ID

    Returns:
        List[str]: The list of chunked texts
        List[str]: The list of chunk IDs
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
    sents = split_sentences(doc)

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


def cluster_chunker(doc, doc_id, encoder = None, lamda: float = 0.5, max_samples_per_cluster: Optional[int] = None):
    """Chunk the document into clusters using single linkage clustering.

    Args:
        doc (str): The document text
        doc_id (str): The document ID
        encoder (object): The encoder object
        lamda (float): The weight of positional distance in the weighted average
        max_samples_per_cluster (int): The maximum number of samples per cluster. If None, set to `len(embs) // 10` to ensure a reasonable balance between the number of clusters and the number of samples in each cluster.

    Returns:
        List[str]: The list of chunked texts
        List[str]: The list of chunked IDs
    """
    if not encoder: raise ValueError("Encoder is required for cluster_chunker")

    # split the document into sentences
    sents = split_sentences(doc)

    # to prevent long ingestion latency caused by large number of sentences
    # group sentences together iteratively until the number of segments is less than 500
    while len(sents) > 500:
        sents = [' '.join(sents[i:i+2]) for i in range(0, len(sents), 2)]

    # get embeddings
    embs = encoder.embed_documents(sents)

    # perform single linkage clustering
    labels = single_linkage_clustering(embs, lamda=lamda, max_samples_per_cluster=max_samples_per_cluster)

    # group texts by cluster labels
    texts, ids = group_by_cluster(sents, labels)

    # get IDs with both doc_id and cluster_id
    ids = ["{}|{}".format(doc_id, cluster_id) for cluster_id in ids]

    return texts, ids


def langchain_chunker(doc: str, doc_id: str, encoder = None, breakpoint_threshold_amount: float = 0.5):
    """Chunk the document into clusters using Langchain semantic chunking.

    Args:
        doc (str): The document text
        doc_id (str): The document ID
        encoder (object): The encoder object
        breakpoint_threshold_amount (float): The threshold amount for the breakpoint

    Returns:
        List[str]: The list of chunked texts
        List[str]: The list of chunked IDs
    """
    if not encoder: raise ValueError("Encoder is required for langchain_chunker")

    # get langchain semantic chunker
    text_splitter = SemanticChunker(encoder, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=breakpoint_threshold_amount)

    # split the document into semantic chunks
    texts = text_splitter.create_documents([doc])
    texts = [text.page_content for text in texts]

    # get IDs with both doc_id and cluster_id
    ids = ["{}|{}".format(doc_id, cluster_id) for cluster_id in range(len(texts))]

    return texts, ids



###### MAIN FUNCTION ######
def chunk_corpus(corpus: Dict[str, Dict[str, str]], chunker: str, verbose: bool = False, **kwargs) -> Dict[str, Dict[str, str]]:
    """Chunk the given corpus using the specified chunker.

    Args:
        corpus (Dict[str, Dict[str, str]]): A dictionary of documents.
        chunker (str): The chunker function to use.
        verbose (bool): Whether to display the clusters.
        kwargs: Additional arguments for the chunker.

    Returns:
        Dict[str, Dict[str, str]]: A corpus of chunked documents.
    """
    print(f"Chunking corpus with {chunker}...")
    chunker = eval(chunker)
    chunked_corpus = {}
    for doc_id, doc in tqdm(corpus.items()):
        # process text from BEIR-format data
        text = (doc["title"] + ".\n" + doc["text"]).strip() if "title" in doc else doc["text"].strip()

        # get chunks
        chunks, chunk_ids = chunker(text, doc_id, **kwargs)

        # log clusters
        if verbose: log_clusters(chunks, chunk_ids)

        # process chunks into BEIR format
        for chunk_id, chunk in zip(chunk_ids, chunks):
            chunked_corpus[chunk_id] = {"text": chunk}

    return chunked_corpus