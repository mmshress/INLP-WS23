from typing import Sequence

from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
)


class ArticlesIndexer:
    def __init__(
        self,
        embedding_model: HuggingFaceEmbedding | None,
        index_name: str,
        verbose: bool = False,
        host: str = "localhost",
        port: int = 9200,
    ) -> None:
        self.INDEX_NAME = index_name
        self.INDEX_BODY = {"settings": {"index": {"number_of_shards": 2}}}
        self.verbose = verbose
        self.client = OpensearchVectorClient(
            f"http://{host}:{port}",
            self.INDEX_NAME,
            dim=1024,
            embedding_field="embedding",
            text_field="chunk",
            search_pipeline="hybrid-search-pipeline",
        )

        self.vector_store = OpensearchVectorStore(self.client)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.service_context = ServiceContext.from_defaults(
            embed_model=embedding_model, llm=None
        )

    def index_documents(self, documents: Sequence[Document]) -> VectorStoreIndex:
        return VectorStoreIndex(
            nodes=documents,
            storage_context=self.storage_context,
            service_context=self.service_context,
        )
