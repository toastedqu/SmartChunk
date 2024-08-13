from typing import List, Dict
from utils import join_title_text
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

def get_encoder(encoder_name: str):
    """Get the encoder for Langchain.

    This function is used elsewhere so should not be included in CustomRetriever.

    Args:
        model (str): The encoder model to use

    Returns:
        encoder: The encoder object
    """
    if encoder_name == "openai":
        return OpenAIEmbeddings(model='text-embedding-3-large')
    elif encoder_name == "huggingface":
        return HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-cos-v1", model_kwargs = {'device': 'cuda'})
    # elif encoder_name == "YOUR_CUSTOM_ENCODER_ON_LANGCHAIN":
    #     return YourCustomLangchainEmbeddings()
    else:
        raise ValueError("Invalid encoder name")

class CustomRetriever:
    def __init__(self, encoder_name):
        self.model = get_encoder(encoder_name)

    # so this is very stupid.
    # langchain's `embed_query()` function does not support batch encoding in parallel.
    # but BEIR requires batch_size in the args, so we are just gonna leave it here.
    # this is very slow. need batch encoding in the future.
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        return np.array([self.model.embed_query(query) for query in queries]).astype(float)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        docs = [join_title_text(doc) for doc in corpus]
        return np.array(self.model.embed_documents(docs)).astype(float)