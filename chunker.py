from cluster import linkage_info, linkage_k_split, linkage_expand_cluster
from scipy.cluster.hierarchy import linkage
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import spacy
import numpy as np

# model_kwargs={'device': 'cpu'}  # for CPU usage
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-distilbert-base-tas-b")
nlp = spacy.load("en_core_web_sm")


def __sim_pos_metric__(A: np.ndarray, B: np.ndarray) -> float:
    A_pos = int(A[0])
    B_pos = int(B[0])
    pos_dist = abs(A_pos - B_pos)  # integer >= 1
    A_vec = A[1:]
    B_vec = B[1:]
    cosine_sim = np.dot(A_vec, B_vec) / (np.linalg.norm(A_vec) * np.linalg.norm(B_vec))
    cosine_dist = 1 - cosine_sim  # 0 — identical vectors, 1 — no correlation, 2 — absolutely different
    return pos_dist * cosine_dist


def __plt__(linkage_matrix):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    plt.figure()
    dendrogram(linkage_matrix)
    plt.savefig("dendrogram.png")


# Use a cluster as a chunk, by clustering sentences in the document, then split the document into k chunks
# This chunker does not guarantee but consider the continuity of sentences, but it only preserves the order
def cluster_chunker(doc: str, doc_id: int, k=4, continuity=True):
    texts = []
    ids = []
    sents = [sent.text for sent in nlp(doc).sents]
    if len(sents) == 1:
        return [doc], [doc_id + "s0"]  # s = split
    candidate_emb = embeddings.embed_documents(sents)
    if continuity:
        for i, emb in enumerate(candidate_emb):
            # append i to the beginning of the embedding
            candidate_emb[i] = np.append([i], emb)
        linkage_matrix = linkage(candidate_emb, method="single", metric=__sim_pos_metric__)
    else:
        linkage_matrix = linkage(candidate_emb, method="single", metric="cosine")
    # __plt__(linkage_matrix)
    linkage_parents, linkage_size, _ = linkage_info(linkage_matrix, len(sents))
    # for every document, use clusters of sentence to find 4-split of the document
    k_split_idx = linkage_k_split(linkage_matrix, len(sents), linkage_size, k)  # indices of sentences in the split
    k_split_idx = sorted(list(k_split_idx))
    for cluster_id in k_split_idx:
        leaf_ids = linkage_expand_cluster(cluster_id, linkage_matrix, len(sents), linkage_parents)
        leaf_ids = sorted(list(leaf_ids))
        cluster = [sents[int(i)] for i in leaf_ids]
        texts.append(" ".join(cluster))
        ids.append("{}s{}".format(doc_id, int(cluster_id)))  # s = split
    return texts, ids


# Use a cluster as a chunk, but guarantee every chunk contains at least k sentences
def sent_cluster_chunker(doc: str, doc_id: int, k=4, continuity=True):
    raise NotImplementedError()


# Simple chunker that every chunk contains k consecutive sentences
def sent_cont_chunker(doc: str, doc_id: int, k=1):
    texts = []
    ids = []
    sents = [sent.text for sent in nlp(doc).sents]
    buffer = ""
    for sid, sent in enumerate(sents):
        if sid % k == k - 1:
            texts.append(buffer + " " + sent)
            ids.append("{}s{}".format(id, int(sid)))  # s = split
            buffer = ""
        else:
            buffer = buffer + " " + sent if buffer != "" else sent
    if buffer != "":  # add the last sentence
        texts.append(buffer)
        ids.append("{}s{}".format(doc_id, len(sents) - 1))
    return texts, ids


# Make every k words a chunk, also called Passage
def arbitrary_chunker(doc: str, doc_id: int, k=100):
    texts = []
    ids = []
    words = doc.split(" ")
    buffer = ""
    for wid, word in enumerate(words):
        if wid % k == k - 1:
            texts.append(buffer + " " + word)
            ids.append("{}s{}".format(id, int(wid)))  # s = split
            buffer = ""
        else:
            buffer = buffer + " " + word if buffer != "" else word
    if buffer != "":  # add the last passage
        texts.append(buffer)
        ids.append("{}s{}".format(doc_id, len(words) - 1))
    return texts, ids


if __name__ == "__main__":
    # print(cluster_chunker({
    #     "text": ["Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”\n So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.\n There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, “Oh dear! Oh dear! I shall be late!” (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.\n In another moment down went Alice after it, never once considering how in the world she was to get out again.\n The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well."],
    #     "title": ["Alice in Wonderland"],
    #     "id": ["1"],
    #     "origin": ["corpus"]
    # }, k=4))
    print(cluster_chunker(
        doc="Baron Augustin-Louis Cauchy was a French mathematician, engineer, and physicist who made pioneering contributions to several branches of mathematics, including mathematical analysis and continuum mechanics. He was one of the first to state and rigorously prove theorems of calculus, rejecting the heuristic principle of the generality of algebra of earlier authors. He (nearly) single-handedly founded complex analysis and the study of permutation groups in abstract algebra. A profound mathematician, Cauchy had a great influence over his contemporaries and successors; Hans Freudenthal stated: \"More concepts and theorems have been named for Cauchy than for any other mathematician (in elasticity alone there are sixteen concepts and theorems named for Cauchy).\" Cauchy was a prolific writer; he wrote approximately eight hundred research articles and five complete textbooks on a variety of topics in the fields of mathematics and mathematical physics. \
                Leonhard Euler was a Swiss mathematician, physicist, astronomer, geographer, logician, and engineer who founded the studies of graph theory and topology and made pioneering and influential discoveries in many other branches of mathematics such as analytic number theory, complex analysis, and infinitesimal calculus. He introduced much of modern mathematical terminology and notation, including the notion of a mathematical function. He is also known for his work in mechanics, fluid dynamics, optics, astronomy, and music theory. Euler is held to be one of the greatest mathematicians in history and the greatest of the 18th century. Several great mathematicians who produced their work after Euler's death have recognised his importance in the field as shown by quotes attributed to many of them: Pierre-Simon Laplace expressed Euler's influence on mathematics by stating, \"Read Euler, read Euler, he is the master of us all.\" Carl Friedrich Gauss wrote: \"The study of Euler's works will remain the best school for the different fields of mathematics, and nothing else can replace it.\" Euler is also widely considered to be the most prolific; his 866 publications as well as his correspondences are being collected in the Opera Omnia Leonhard Euler which, when completed, will consist of 81 quarto volumes. He spent most of his adult life in Saint Petersburg, Russia, and in Berlin, then the capital of Prussia. \
                David Hilbert was a German mathematician and one of the most influential mathematicians of the 19th and early 20th centuries. Hilbert discovered and developed a broad range of fundamental ideas including invariant theory, the calculus of variations, commutative algebra, algebraic number theory, the foundations of geometry, spectral theory of operators and its application to integral equations, mathematical physics, and the foundations of mathematics (particularly proof theory). Hilbert adopted and defended Georg Cantor's set theory and transfinite numbers. In 1900, he presented a collection of problems that set a course for mathematical research of the 20th century. Hilbert and his students contributed to establishing rigor and developed important tools used in modern mathematical physics. Hilbert was one of the founders of proof theory and mathematical logic.",
        doc_id=0, k=3))
