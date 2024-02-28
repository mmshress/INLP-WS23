import os
from typing import Any, List, Optional

from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llmlegalassistant.utils import get_evaluation_dataset_dir


class IndexerFactory:
    @staticmethod
    def create_index(
        documents: Optional[List[BaseNode]],
        index_name: str,
        database: str,
        embed_model: HuggingFaceEmbedding | None,
    ) -> Any:
        """
        Sets up the chunk size and overlap size

        Parameters:
        -----------
        splitter : str
            The type of splitter to be instanciated
        chunk_size : int
            The size of the chunk
        chunk_overlap : int
            The size of the overlap
        """
        match database:
            case "Chromadb":
                import chromadb
                from llama_index.core import StorageContext, VectorStoreIndex
                from llama_index.vector_stores.chroma import ChromaVectorStore

                database_dir = get_evaluation_dataset_dir("databases")
                db = chromadb.PersistentClient(
                    path=os.path.join(str(database_dir), "evaluation")
                )

                is_index = False
                if index_name in db.list_collections():
                    is_index = True

                chroma_collection = db.get_or_create_collection(index_name)
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )

                if is_index:
                    return VectorStoreIndex.from_vector_store(
                        vector_store=vector_store, storage_context=storage_context
                    )

                return VectorStoreIndex(
                    nodes=documents,
                    storage_context=storage_context,
                    embed_model=embed_model,
                )
            case _:
                return None
