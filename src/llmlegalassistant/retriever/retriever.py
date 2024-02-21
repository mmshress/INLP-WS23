from typing import Any

from llama_index.core.base_retriever import BaseRetriever


class RetrieverFactory:
    @staticmethod
    def generate_retriver(
        retriever_method: str,
        index: Any,
        top_k: int = 10,
        num_queries: int = 1,
        verbose: bool = False,
    ) -> BaseRetriever | None:
        match retriever_method:
            case "VectorIndexRetriever":
                from llama_index.retrievers import VectorIndexRetriever

                return VectorIndexRetriever(index=index, similarity_top_k=top_k)
            case "BM25Retriever":
                from llama_index.retrievers import BM25Retriever

                return BM25Retriever(docstore=index.docstore, similarity_top_k=top_k)
            case "QueryFusionRetriever":
                from llama_index.retrievers import BM25Retriever, \
                    QueryFusionRetriever

                vector_retriever = index.as_retriever(similarity_top_k=top_k)
                bm25_retriever = BM25Retriever.from_defaults(
                    docstore=index.docstore, similarity_top_k=top_k
                )

                return QueryFusionRetriever(
                    [vector_retriever, bm25_retriever],
                    similarity_top_k=top_k,
                    num_queries=num_queries,
                    mode="reciprocal_rerank",
                    use_async=True,
                    # query_gen_prompt="...",  # we could override the query generation prompt here
                    verbose=verbose,
                )
            case _:
                return None
