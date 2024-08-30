import numpy as np
from typing import List, Dict
from utils import join_title_text
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class CustomEncoder:
    def __init__(self, encoder_name, device: str = "cpu"):
        self.name = encoder_name
        self.device = device
        self.get_encoder(encoder_name)

    def get_encoder(self, encoder_name: str):
        """Get the encoder for Langchain.

        This function is used elsewhere so should not be included in CustomRetriever.

        Args:
            model (str): The encoder model to use

        Returns:
            encoder: The encoder object
        """
        if encoder_name == "openai":
            self.model = OpenAIEmbeddings(model='text-embedding-3-small')
        elif encoder_name == "huggingface":
            self.model = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-cos-v1", model_kwargs = {'device': self.device})
        # elif encoder_name == "YOUR_CUSTOM_ENCODER_ON_LANGCHAIN":
        #     self.model = YourCustomLangchainEmbeddings()
        else:
            raise ValueError("Invalid encoder name")

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        return np.array(self.model.embed_documents(queries)).astype(float)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 16, **kwargs) -> np.ndarray:
        docs = [join_title_text(doc) for doc in corpus]
        return np.array(self.model.embed_documents(docs)).astype(float)