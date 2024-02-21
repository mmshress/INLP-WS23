from typing import List

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings


class EmbedWrapper:
    def __init__(self, model_name: str, source: str) -> None:
        if source == "huggingface":
            self.model = HuggingFaceEmbeddings(model_name=model_name)
        elif source == "openai":
            self.model = OpenAIEmbeddings(model=model_name)
        else:
            raise ValueError("Unsupported source. Choose 'huggingface' or 'openai'.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_text(text)
