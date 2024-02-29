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
            chunk_size = config["splitter"]["chunk_size"]
            overlap_size = config["splitter"]["overlap_size"]
            index_name = config["store"]["index"]
            database_name = config["store"]["type"]
            retriever_type = config["retriever"]["method"]
            top_k = config["retriever"]["top_k"]
            response_synthesizer = config["retriever"]["response_synthesizer"]

            embed_model = HuggingFaceEmbedding(model_name=model_name)
 
            if self.verbose:
                print(f"[LLMLegalAssistant] Generating Splitter...")
            text_splitter = SplitterFactory.generate_splitter(
                splitter=splitter,
                embed_model=embed_model,
                chunk_size=chunk_size,
                overlap_size=overlap_size,
            )
            if self.verbose:
                print(f"[LLMLegalAssistant] Splitter Generated!")

            if self.verbose:
                print(f"[LLMLegalAssistant] Creating Document Index...")
            index = self._create_document_index(
                splitter=text_splitter,
                embed_model=embed_model,
                index_name=index_name,
                database_name=database_name,
                evaluate=True,
            )
            if self.verbose:
                print(f"[LLMLegalAssistant] Document Index Created!")

            if self.verbose:
                print(f"[LLMLegalAssistant] Generating Retriever...")
            retriever = RetrieverFactory.generate_retriver(
                retriever_method=retriever_type,
                index=index,
                top_k=top_k,
                verbose=self.verbose,
            )
            if self.verbose:
                print(f"[LLMLegalAssistant] Retriever Generated!")

            if self.verbose:
                print(f"[LLMLegalAssistant] Generating Query Engine...")
            query_engine = self._generate_query_engine(
                retriever=retriever, 
                # response_synthesizer=response_synthesizer
            )
            if self.verbose:
                print(f"[LLMLegalAssistant] Query Engine Generated!")

            return query_engine

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
            if document_dir is None:
                print("[LLMLegalAssistant] Evaluation directory doesn't exists")

        documents = SimpleDirectoryReader(input_dir=document_dir).load_data()
        if self.verbose:
            print(f"[LLMLegalAssistant] Number of Documents Loaded: {len(documents)}")

        documents_nodes = splitter.get_nodes_from_documents(documents)
        if self.verbose:
            print(
                f"[LLMLegalAssistant] Nodes created from documents: {len(documents_nodes)}"
            )

        return IndexerFactory.create_index(documents=documents_nodes, index_name=index_name, database=database_name, embed_model=embed_model)
        
    def _generate_query_engine(self, retriever: Any, llm: Any = None, node_postprocessor: Any = None) -> Any:
    # def _generate_query_engine(self, retriever: Any, response_synthesizer: Any, node_postprocessor: Any = None) -> Any:
        if node_postprocessor is not None:
            return RetrieverQueryEngine(retriever=retriever, llm=llm, node_postprocessors=node_postprocessor)
            # return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer, node_postprocessors=node_postprocessor)
        else:
            return RetrieverQueryEngine(retriever=retriever, llm=llm)
            # return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
