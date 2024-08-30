import json
import spacy
import random
import numpy as np
from typing import *
nlp = spacy.load('en_core_web_sm')
random.seed(2)

def split_sentences(doc: str) -> List[str]:
    """Split the document into sentences.

    Args:
        doc (str): The document text

    Returns:
        List[str]: The list of sentences
    """
    return [text.text.strip() for segment in doc.split("\n") for text in nlp(segment).sents]


def join_title_text(doc: Dict[str, str]) -> str:
    """Join the title and text of a document into a single string.

    Args:
        doc (Dict[str, str]): The document with keys "title" and "text".

    Returns:
        str: The title and text joined into a single string.
    """
    return (doc["title"] + "\n" + doc["text"]).strip() if "title" in doc and doc['title'] else doc["text"].strip()


def load_embeddings(doc_id: str, dataset: str, encoder_name: str) -> np.ndarray:
    """Load the embeddings for the given document.

    Args:
        doc_id (str): The document ID
        dataset (str): The dataset name
        encoder_name (str): The encoder name

    Returns:
        np.ndarray: The embeddings for the document
    """
    m = json.load(open(f"embeddings/{dataset}/{encoder_name}/docids.json", "r"))
    return np.load(f"embeddings/{dataset}/{encoder_name}/{m[doc_id]}.npy")