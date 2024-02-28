import os
from typing import Any

from llama_index.core import SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llmlegalassistant.data import ArticlesIndexer, IndexerFactory
from llmlegalassistant.retriever import RetrieverFactory
from llmlegalassistant.splitter import SplitterFactory
from llmlegalassistant.utils import (
    get_articles_dir,
    get_evaluation_dataset_dir,
    load_configurations,
)


class LLMLegalAssistant:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def evaluate(self) -> None:
        configs = load_configurations()
        if self.verbose:
            print("[LLMLegalAssistant] Configurations loaded")

        for config in configs:
            splitter = config["splitter"]["method"]
            model_name = config["embed"]
            retriever_type = config["retriever"]["method"]
            response_synthesizer = config["retriever"]["response_synthesizer"]

            embed_model = HuggingFaceEmbedding(model_name=model_name)

            database_name = config["store"]["type"]
            index_name = config["store"]["index"]
            chunk_size = config["splitter"]["chunk_size"]
            overlap_size = config["splitter"]["overlap_size"]
            text_splitter = SplitterFactory.generate_splitter(
                splitter=splitter,
                embed_model=embed_model,
                chunk_size=chunk_size,
                overlap_size=overlap_size,
            )
            index = self._create_document_index(
                splitter=text_splitter,
                embed_model=embed_model,
                index_name=index_name,
                database_name=database_name,
                evaluate=True,
            )

            top_k = config["retriever"]["top_k"]
            retriever = RetrieverFactory.generate_retriver(
                retriever_method=retriever_type,
                index=index,
                top_k=top_k,
                verbose=self.verbose,
            )

            return self._generate_query_engine(
                retriever=retriever, response_synthesizer=response_synthesizer
            )

    def _create_document_index(
        self,
        splitter: Any,
        embed_model: HuggingFaceEmbedding | None,
        index_name: str,
        database_name: str,
        evaluate: bool = False,
    ) -> Any:
        if not evaluate:
            document_dir = os.path.join(get_articles_dir(), "txts")
        else:
            document_dir = get_evaluation_dataset_dir("documents")

        documents = SimpleDirectoryReader(input_dir=document_dir).load_data()
        if self.verbose:
            print(f"[LLMLegalAssistant] Number of Documents Loaded: {len(documents)}")

        documents_nodes = splitter.get_nodes_from_documents(documents)
        if self.verbose:
            print(
                f"[LLMLegalAssistant] Nodes created from documents: {len(documents_nodes)}"
            )

        IndexerFactory.create_index(
            documents=documents_nodes,
            index_name=index_name,
            database=database_name,
            embed_model=embed_model,
        )

        article_indexer = ArticlesIndexer(
            embedding_model=embed_model, index_name=index_name, verbose=self.verbose
        )

        index = article_indexer.index_documents(documents_nodes)
        if self.verbose:
            print("[LLMLegalAssistant] Documents are indexed!")

        return index

    def _generate_query_engine(
        self,
        retriever: Any,
        response_synthesizer: Any,
        node_postprocessor: Any = None,
    ) -> Any:
        if node_postprocessor is not None:
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=node_postprocessor,
            )
        else:
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
            )

        return query_engine
