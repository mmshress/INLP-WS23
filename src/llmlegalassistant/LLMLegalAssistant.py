import os
import subprocess
import time
from typing import Any

import transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from llama_index.core import SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from torch import bfloat16, cuda
from transformers import AutoTokenizer

from llmlegalassistant.data import ArticlesIndexer, IndexerFactory
from llmlegalassistant.retriever import RetrieverFactory
from llmlegalassistant.splitter import SplitterFactory
from llmlegalassistant.utils import get_api_key, get_articles_dir, get_evaluation_dataset_dir, get_models_dir, load_configurations


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
            if splitter != "SemanticSplitterNodeParser":
                continue
            model_name = config["embed"]
            chunk_size = config["splitter"]["chunk_size"]
            overlap_size = config["splitter"]["overlap_size"]
            index_name = config["store"]["index"]
            database_name = config["store"]["type"]
            retriever_type = config["retriever"]["method"]
            top_k = config["retriever"]["top_k"]
            response_synthesizer = config["retriever"]["response_synthesizer"]
            language_model = config["model"]["name"]

            if splitter == "SemanticSplitterNodeParser":
                embed_model = HuggingFaceEmbeddings(
                    model_name=model_name, 
                            model_kwargs={'device': 'cuda:0'}, 
                                    encode_kwargs={'device': 'cuda:0', 'batch_size': 32}
                                                    )

            else:
                embed_model = HuggingFaceEmbedding(model_name=model_name)
 
            if self.verbose:
                print(f"[LLMLegalAssistant] Generating Splitter...")

            text_splitter = SplitterFactory.generate_splitter(
                splitter=splitter,
                embed_model=embed_model,
                model_name=model_name,
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
                top_k=int(top_k),
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

            answer = query_engine.query("What is the main objective of Article I in the Cooperation Agreement?")

            if self.verbose:
                print("[LLMLegalAssistant] The answer is")
                # answer.print_response_stream()
                for source_node in answer.source_nodes:
                    print(source_node.text)

                # print(f"[LLMLegalAssistant] The nodes used for the above answer are {answer.source_nodes}")

            return answer

    def _create_document_index(
        self,
        splitter: Any,
        embed_model: Any,
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

    # def _generate_query_engine(self, retriever: Any, index: Any = None, index_name: Any = None, llm: Any = None, node_postprocessor: Any = None) -> Any:
    def _generate_query_engine(self, retriever: Any, index: Any = None, index_name: Any = None, llm: Any = None, response_mode: str = "tree_summarize", node_postprocessor: Any = None) -> Any:
        # if index_name == "VectorIndexRetriever":
        # return index.as_query_engine(llm=llm)
        # response_synthesizer = get_response_synthesizer(response_mode=response_mode, llm=llm, streaming=True, verbose=self.verbose)

        return RetrieverQueryEngine.from_args(llm=llm, retriever=retriever)
        # return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

        # if node_postprocessor is not None:
        #     return RetrieverQueryEngine.from_args(retriever=retriever, llm=llm, node_postprocessors=node_postprocessor)
            # return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer, node_postprocessors=node_postprocessor)
        # else:
        #     return RetrieverQueryEngine(retriever=retriever, llm=llm)
        #     return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

    def _initialize_llm(self, language_model: str = None) -> Any:
        # if language_model is None:
        #     return None

        # model_path = os.path.join(get_models_dir(), language_model)
        hf_auth = "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"
        model_id = 'meta-llama/Llama-2-7b-chat-hf'

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)

        bitsAndBites_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bitsAndBites_config,
            device_map='cuda:0',
            token=hf_auth,
            load_in_8bit_fp32_cpu_offload=True
        )
        generate_text = transformers.pipeline(
            model=model, 
            tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            # we pass model parameters here too
            temperature=0.01,
            max_new_tokens=512,  # max number of tokens to generate in the output
            repetition_penalty=1.1,  # without this output begins repeating
        )
        return HuggingFacePipeline(pipeline=generate_text)

