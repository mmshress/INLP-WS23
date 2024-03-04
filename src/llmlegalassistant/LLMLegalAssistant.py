import os
from typing import Any

import transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms import openai
from llama_index.llms.openai import OpenAI
from torch import bfloat16
from transformers import AutoTokenizer

from llmlegalassistant.data import IndexerFactory
from llmlegalassistant.retriever import RetrieverFactory
from llmlegalassistant.splitter import SplitterFactory

from llmlegalassistant.utils import (
    get_articles_dir,
    get_evaluation_dataset_dir,
    get_project_dir,
    load_configurations,
)

# from openai import OpenAI


class LLMLegalAssistant:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def answer(
        self, prompt: str, is_openai: bool = False, api_key_file: str | None = None
    ) -> str:
        if api_key_file is None:
            api_key = os.environ["OPENAI_API_KEY"]
        else:
            print(api_key_file, os.path.join(get_project_dir(), api_key_file))
            with open(os.path.join(get_project_dir(), api_key_file), "r") as keyfile:
                api_key = keyfile.read()
                print(api_key)

        if is_openai:
            language_model = "gpt-3.5-turbo"
        else:
            language_model = "meta-llama/Llama-2-7b-chat-hf"

        llm = self._initialize_llm(language_model=language_model, api_key=api_key)
        Settings.llm = llm
        Settings.context_window = 4096

        print("embeding")

        embed_model = HuggingFaceEmbedding(model_name="infgrad/stella-base-en-v2")

        from llama_index.core.storage.docstore import SimpleDocumentStore

        docstore = SimpleDocumentStore.from_persist_path(
            "/home/ubuntu/projects/llmlegalassistant/datasets/datasets/docstore/stella_alldocs_semantic.json"
        )
        print(docstore)

        text_splitter = SplitterFactory.generate_splitter(
            splitter="SemanticTextNodeParser",
            embed_model=embed_model,
            model_name="infgrad/stella-base-en-v2",
            chunk_size=510,
            overlap_size=20,
        )

        print("start index")
        index_name = "Splitter_sentence_stella_emb"
        # nodes, index = self._create_document_index(
        index = self._create_document_index(
            splitter=text_splitter,
            embed_model=embed_model,
            index_name=index_name,
            database_name="Chromadb",
            evaluate=True,
        )

        retriever = RetrieverFactory.generate_retriver(
            retriever_method="QueryFusionRetriever",
            index=index,
            docstore=docstore,
            # nodes=nodes,
            top_k=1,
        )

        query_engine = self._generate_query_engine(
            retriever=retriever,
            llm=llm,
        )
        # print("".join[prompt])
        response = query_engine.query(prompt)
        return response

    def evaluate(self) -> Any:
        configs = load_configurations()
        if self.verbose:
            print("[LLMLegalAssistant] Configurations loaded")

        llm = None
        llm_name = None
        for config in configs:
            splitter = config["splitter"]["method"]
            if splitter != "SemanticSplitterNodeParser":
                continue
            model_name = config["embed"]
            chunk_size = config["splitter"]["chunk_size"]
            overlap_size = config["splitter"]["overlap_size"]
            index_name = config["store"]["index"]
            database_name = config["store"]["type"]
            retriever_type = config["retriever"]["method"]
            top_k = config["retriever"]["top_k"]
            language_model = config["model"]["name"]

            if splitter == "SemanticSplitterNodeParser":
                embed_model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cuda:0"},
                    encode_kwargs={"device": "cuda:0", "batch_size": 32},
                )

            else:
                embed_model = HuggingFaceEmbedding(model_name=model_name)

            if self.verbose:
                print("[LLMLegalAssistant] Generating Splitter...")

            text_splitter = SplitterFactory.generate_splitter(
                splitter=splitter,
                embed_model=embed_model,
                model_name=model_name,
                chunk_size=chunk_size,
                overlap_size=overlap_size,
            )

            if self.verbose:
                print("[LLMLegalAssistant] Splitter Generated!")

            if self.verbose:
                print("[LLMLegalAssistant] Creating Document Index...")

            nodes, index = self._create_document_index(
                splitter=text_splitter,
                embed_model=embed_model,
                index_name=index_name,
                database_name=database_name,
                evaluate=True,
            )

            if self.verbose:
                print("[LLMLegalAssistant] Document Index Created!")

            if self.verbose:
                print("[LLMLegalAssistant] Generating Retriever...")

            retriever = RetrieverFactory.generate_retriver(
                retriever_method=retriever_type,
                index=index,
                nodes=nodes,
                top_k=int(top_k),
                verbose=self.verbose,
            )

            if self.verbose:
                print("[LLMLegalAssistant] Retriever Generated!")

            if self.verbose:
                print("[LLMLegalAssistant] Loading a Language Model...")

            if language_model != llm_name:
                llm = self._initialize_llm(language_model)

            if self.verbose:
                print("[LLMLegalAssistant] Language Model Loaded!")

            if self.verbose:
                print("[LLMLegalAssistant] Generating Query Engine...")

            query_engine = self._generate_query_engine(
                retriever=retriever,
                llm=llm,
            )

            if self.verbose:
                print("[LLMLegalAssistant] Query Engine Generated!")

            answer = query_engine.query(
                "What is the main objective of Article I in the Cooperation Agreement?"
            )

            if self.verbose:
                print("[LLMLegalAssistant] The answer is")
                for source_node in answer.source_nodes:
                    print(source_node.text)

            return answer

    def _create_document_index(
        self,
        splitter: Any,
        embed_model: Any,
        index_name: str,
        database_name: str,
        evaluate: bool = False,
    ) -> Any:
        # if not evaluate:
        #     document_dir = os.path.join(get_articles_dir(), "txts")
        # else:
        document_dir = get_evaluation_dataset_dir("documents")
        if document_dir is None:
            print("[LLMLegalAssistant] Evaluation directory doesn't exists")

        # documents = SimpleDirectoryReader(input_dir=document_dir).load_data()
        #
        # if self.verbose:
        #     print("[LLMLegalAssistant] Number of Documents Loaded: {len(documents)}")
        #
        # documents_nodes = splitter.get_nodes_from_documents(documents)

        if self.verbose:
            print(
                "[LLMLegalAssistant] Nodes created from documents: {len(documents_nodes)}"
            )

        # return documents_nodes, IndexerFactory.create_index(
        #     nodes=documents_nodes,
        #     index_name=index_name,
        #     database=database_name,
        #     embed_model=embed_model,
        # )
        return IndexerFactory.create_index(
            # nodes=documents_nodes,
            index_name=index_name,
            database=database_name,
            embed_model=embed_model,
        )

    def _generate_query_engine(self, retriever: Any, llm: Any = None) -> Any:
        return RetrieverQueryEngine.from_args(llm=llm, retriever=retriever)

    def _initialize_llm(
        self, api_key: str, language_model: str = "meta-llama/Llama-2-7b-chat-hf"
    ) -> Any:
        if language_model == "gpt-3.5-turbo":
            return OpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0.01)

        hf_auth = api_key

        tokenizer = AutoTokenizer.from_pretrained(
            language_model, use_auth_token=hf_auth
        )

        bitsAndBites_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        )

        model_config = transformers.AutoConfig.from_pretrained(
            language_model, use_auth_token=hf_auth
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            language_model,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bitsAndBites_config,
            device_map="auto",
            token=hf_auth,
        )

        generate_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            task="text-generation",
            temperature=0.01,
            max_new_tokens=512,
            repetition_penalty=1.1,
        )

        return HuggingFacePipeline(pipeline=generate_text)
