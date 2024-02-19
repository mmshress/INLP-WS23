from typing import List
import langchain_core.embeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import OpenAIEmbedding

class EmbedWrapper(langchain_core.embeddings.Embeddings):
    def __init__(self, model_name: str, source: str):
        if source == 'huggingface':
            self.model = HuggingFaceEmbeddings(model_name=model_name)
        elif source == 'openai':
            self.model = OpenAIEmbedding(model=model_name)
        else:
            raise ValueError("Unsupported source. Choose 'huggingface' or 'openai'.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_text(text)