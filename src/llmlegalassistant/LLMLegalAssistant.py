import os
from typing import Any

from llama_index.core import SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llmlegalassistant.data import ArticlesIndexer
from llmlegalassistant.retriever import RetrieverFactory
from llmlegalassistant.splitter import SplitterFactory
from llmlegalassistant.utils import get_articles_dir, load_configurations


class LLMLegalAssistant:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def evaluate(self, configurations: list[str] | None = None) -> None:
        configs = load_configurations(configurations)

        for config in configs:
            text_splitter = None
            embed_model = None

            splitter = config["splitter"]["type"]
            model_name = config["embed"]
            index_name = config["store"]["index"]
            retriever = config["retriever"]["type"]
            # will need later when we add llm models
            # llm_model = config["model"]["name"]
            if model_name is not None:
                embed_model = HuggingFaceEmbedding(model_name=model_name)

            if splitter is not None:
                chunk_size = config["splitter"]["chunk_size"]
                overlap_size = config["splitter"]["overlap_size"]
                text_splitter = SplitterFactory.generate_splitter(
                    splitter=splitter,
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                )

            if retriever is not None:
                top_k = config["retriever"]["top_k"]
                num_queries = config["retriever"]["num_queries"]
                retriever = RetrieverFactory.generate_retriver(
                    retriever_method=retriever,
                    index=index_name,
                    top_k=top_k,
                    num_queries=num_queries,
                    verbose=True,
                )

            self._create_document_index(text_splitter, embed_model, index_name)
            # self._generate_query_engine(
            #     text_splitter, embed_model, index_name
            # )

    def _create_document_index(
        self, splitter: Any, embed_model: HuggingFaceEmbedding | None, index_name: str
    ) -> Any:
        """
        What I think
        ------------
        documents, splitter, embed_model, index_name

        What is
        -------
        self,
        embedding_model: Embeddings,
        chunking_strategy: str,
        verbose: bool = False,
        host: str = "localhost",
        port: int = 9200,

        def index_documents(self) -> VectorStoreIndex:
            files_dir = get_dataset_dir() + "/articles/txts/"
            documents = SimpleDirectoryReader(files_dir).load_data()
            return VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
                service_context=self.service_context,
            )
        """
        # 1.  index: splitter, text_files, embed_model
        # 1.1 splitter: text_files
        # 1.2 embed_model and index: text_files
        document_dir = os.path.join(get_articles_dir(), "txts")
        documents = SimpleDirectoryReader(input_dir=document_dir).load_data()
        if self.verbose:
            print(f"[LLMLegalAssistant] Number of Documents Loaded: {len(documents)}")

        documents_nodes = splitter.get_nodes_from_documents(documents)
        if self.verbose:
            print(
                f"[LLMLegalAssistant] Nodes created from documents: {len(documents_nodes)}"
            )

        article_indexer = ArticlesIndexer(self.verbose)
        article_indexer.index_documents(documents_nodes)

        return documents_nodes

    def _generate_query_engine(
        self,
        retriever: BaseRetriever,
        response_synthesiser: Any,
        node_postprocessor: Any = None,
    ) -> Any:
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesiser,
            node_postprocessors=node_postprocessor,
        )

        return query_engine
