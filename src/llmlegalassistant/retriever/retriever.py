from typing import Any, Optional

from llama_index.core.retrievers import BaseRetriever

# from typing import List


# from llama_index.core.schema import BaseNode


class RetrieverFactory:
    @staticmethod
    def generate_retriver(
        retriever_method: str,
        index: Any,
        docstore: Any,
        # nodes: Any,
        llm: Any = None,
        top_k: int = 1,
        verbose: bool = False,
    ) -> Optional[BaseRetriever]:
        match retriever_method:
            case "VectorIndexRetriever":
                return index.as_retriever(similarity_top_k=top_k)
            case "BM25Retriever":
                from llama_index.retrievers.bm25 import BM25Retriever

                return BM25Retriever.from_defaults(
                    docstore=docstore, similarity_top_k=top_k
                )
            case "QueryFusionRetriever":
                from llama_index.core.retrievers import QueryFusionRetriever
                from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
                from llama_index.retrievers.bm25 import BM25Retriever

                vector_retriever = index.as_retriever(similarity_top_k=top_k)
                bm25_retriever = BM25Retriever.from_defaults(
                    docstore=docstore, similarity_top_k=top_k
                )

                return QueryFusionRetriever(
                    [vector_retriever, bm25_retriever],
                    similarity_top_k=top_k,
                    num_queries=1,
                    mode=FUSION_MODES.RECIPROCAL_RANK,
                    use_async=True,
                    verbose=verbose,
                    llm=llm,
                )
            case _:
                return None
