import os
from typing import Any

from llama_index.core import SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

from llmlegalassistant.data import ArticlesIndexer, IndexerFactory
from llmlegalassistant.retriever import RetrieverFactory
from llmlegalassistant.splitter import SplitterFactory
from llmlegalassistant.utils import (
    get_articles_dir,
    get_evaluation_dataset_dir,
    load_configurations,
    get_models_dir,
)


class LLMLegalAssistant:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def evaluate(self) -> Any:
        configs = load_configurations()
        if self.verbose:
            print("[LLMLegalAssistant] Configurations loaded")

        llm = None
        llm_name = None
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
            language_model = config["model"]["name"]

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
                print(f"[LLMLegalAssistant] Loading a Language Model...")
            if language_model != llm_name:
                llm = self._initialize_llm(language_model)
            if self.verbose:
                print(f"[LLMLegalAssistant] Language Model Loaded!")

            if self.verbose:
                print(f"[LLMLegalAssistant] Generating Query Engine...")
            query_engine = self._generate_query_engine(
                index=index,
                index_name=index_name,
                retriever=retriever, 
                llm=llm,
                # response_synthesizer=response_synthesizer
            )
            if self.verbose:
                print(f"[LLMLegalAssistant] Query Engine Generated!")

            answer = query_engine.query("What are these documents?")
            if self.verbose:
                print(f"[LLMLegalAssistant] The answer is {answer}")

            return answer


    def _initialize_llm(self, language_model: str = None) -> Any:
        if language_model is None:
            return None

        model_path = os.path.join(get_models_dir(), language_model)

        return LlamaCPP(
            # optionally, you can set the path to a pre-downloaded model instead of model_url
            model_path=model_path,
            temperature=0.1,
            max_new_tokens=512,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=3900,
            # kwargs to pass to __call__()
            generate_kwargs={},
            # kwargs to pass to __init__()
            # set to at least 1 to use GPU
            model_kwargs={"n_gpu_layers": 64},
            # transform inputs into Llama2 format
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=self.verbose,
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
        
    def _generate_query_engine(self, retriever: Any, index: Any = None, index_name: Any = None, llm: Any = None, node_postprocessor: Any = None) -> Any:
    # def _generate_query_engine(self, retriever: Any, response_synthesizer: Any, node_postprocessor: Any = None) -> Any:
        # if index_name == "VectorIndexRetriever":
        return index.as_query_engine(llm=llm)
        # return RetrieverQueryEngine.from_args(llm=llm, retriever=retriever)

        # if node_postprocessor is not None:
        #     return RetrieverQueryEngine.from_args(retriever=retriever, llm=llm, node_postprocessors=node_postprocessor)
        #     return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer, node_postprocessors=node_postprocessor)
        # else:
        #     return RetrieverQueryEngine(retriever=retriever, llm=llm)
        #     return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
