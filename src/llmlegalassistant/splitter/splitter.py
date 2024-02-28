from typing import Any

# from llama_index.core.node_parser import NodeParser, TextSplitter
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core.base.embeddings.base import BaseEmbedding


class SplitterFactory:
    @staticmethod
    def generate_splitter(
        splitter: str, embed_model: Any, chunk_size: int = 512, overlap_size: int = 50
    ) -> Any:
        """
        Sets up the chunk size and overlap size

        Parameters:
        -----------
        splitter : str
            The type of splitter to be instanciated
        chunk_size : int
            The size of the chunk
        chunk_overlap : int
            The size of the overlap
        """
        match splitter:
            case "RecursiveCharacterTextSplitter":
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                from llama_index.core.node_parser import LangchainNodeParser

                return LangchainNodeParser(
                    RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, chunk_overlap=overlap_size
                    )
                )
            case "SentenceSplitter":
                from llama_index.core.node_parser import SentenceSplitter

                return SentenceSplitter(
                    chunk_size=chunk_size, chunk_overlap=overlap_size
                )
            case "TokenTextSplitter":
                from llama_index.core.node_parser import TokenTextSplitter

                return TokenTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=overlap_size
                )
            case "SemanticSplitterNodeParser":
                from llama_index.core.node_parser import SemanticSplitterNodeParser

                return SemanticSplitterNodeParser(embed_model=embed_model)
            case _:
                return None
