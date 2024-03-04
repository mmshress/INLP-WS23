import os
from typing import Any, List, Optional

from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llmlegalassistant.utils import get_articles_dir, get_evaluation_dataset_dir
from llmlegalassistant.utils.utils import get_dataset_dir


class IndexerFactory:
    @staticmethod
    def create_index(
        # nodes: Optional[List[BaseNode]],
        index_name: str,
        database: str,
        embed_model: HuggingFaceEmbedding | None,
        evaluate: bool = False,
        verbose: bool = False,
    ) -> Any:
        """
        Creates an object of the index that is selected, after indexing the nodes/nodes

        Parameters:
        -----------
        nodes : Optional[List[BaseNode]]
            The nodes that are needed to be indexed
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

                if not evaluate:
                    database_dir = os.path.join(get_articles_dir(), "txts")
                else:
                    database_dir = get_evaluation_dataset_dir("nodes")
                    if database_dir is None:
                        print("[LLMLegalAssistant] Evaluation directory doesn't exists")

                # if verbose:
                #     print(
                #         f"[Chromadb] Loading Database from {database_dir} directory..."
                #     )

                if not evaluate:
                    db = chromadb.PersistentClient(
                        path=os.path.join(str(database_dir), "evaluation")
                    )
                else:
                    dataset_dir = get_dataset_dir()
                    db = chromadb.PersistentClient(
                        path=os.path.join(dataset_dir, "datasetv1")
                    )

                # if verbose:
                #     print(f"[Chromadb] Database from {database_dir} loaded!")

                is_index = False
                if index_name in db.list_collections():
                    if verbose:
                        print(
                            f"[Chromadb] Index {index_name} Already exists in {db.list_collections}!"
                        )
                    is_index = True

                chroma_collection = db.get_or_create_collection(index_name)
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                )
<<<<<<< Updated upstream
                if is_index:
                    index = VectorStoreIndex.from_vector_store(
                        vector_store=vector_store, storage_context=storage_context
                    )
                else:
                    index = VectorStoreIndex(
                        nodes=nodes,
                        storage_context=storage_context,
                        embed_model=embed_model,
                    )
=======

                # if is_index:
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context,
                    embed_model=embed_model,
                )
                print(is_index, "index_index")
                # else:
                #     index = VectorStoreIndex(
                #         nodes=nodes,
                #         storage_context=storage_context,
                #         embed_model=embed_model,
                #     )
>>>>>>> Stashed changes

                if verbose:
                    print(
                        f"[Chromadb] Vector Store Index of index {index_name} Loaded/Created!"
                    )

                return index
            case _:
                return None
