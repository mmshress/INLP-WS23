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
        verbose: bool = False
    ) -> Any:
        """
        Creates an object of the index that is selected, after indexing the documents/nodes

        Parameters:
        -----------
        documents : Optional[List[BaseNode]]
            The documents that are needed to be indexed
        index_name : str
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

                if verbose:
                    print(f"[Chromadb] Loading Database from {database_dir} directory...")
                db = chromadb.PersistentClient(path=os.path.join(str(database_dir), "evaluation"))
                if verbose:
                    print(f"[Chromadb] Database from {database_dir} loaded!")

                is_index = False
                if index_name in db.list_collections():
                    if verbose:
                        print(f"[Chromadb] Index {index_name} Already exists!")
                    is_index = True

                if verbose:
                    print(f"[Chromadb] Loading Index {index_name}...")
                chroma_collection = db.get_or_create_collection(index_name)
                if verbose:
                    print(f"[Chromadb] Index {index_name} loaded!")

                if verbose:
                    print(f"[Chromadb] Creating Vector Store of index {index_name}...")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                if verbose:
                    print(f"[Chromadb] Vector Store of index {index_name} created!")

                if verbose:
                    print(f"[Chromadb] Creating Storage Context of index {index_name}...")
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )
                if verbose:
                    print(f"[Chromadb] Storage Context of index {index_name} created!")

                if verbose:
                    print(f"[Chromadb] Creating Vector Store Index of index {index_name}...")
                if is_index:
                    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
                else:
                    index = VectorStoreIndex(nodes=documents, storage_context=storage_context, embed_model=embed_model)
                if verbose:
                    print(f"[Chromadb] Vector Store Index of index {index_name} created!")

                return index
            case _:
                return None
