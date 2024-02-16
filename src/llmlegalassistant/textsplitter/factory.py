from typing import Any

from llmlegalassistant.textsplitter.Splitter import Splitter


class SplitterFactory:
    """
    A factory class to create a splitter
    """

    @staticmethod
    def create_splitter(splitter: str, *args: Any, **kwargs: Any) -> Splitter:
        """
        Sets up the chunk size and overlap size

        Parameters:
        -----------
        chunk_size : int
            The size of the chunk
        chunk_overlap : int
            The size of the overlap
        """
        match splitter:
            case "recursive_character_text_splitter_v1":
                from llmlegalassistant.textsplitter import \
                    RecursiveTextSplitter

                return RecursiveTextSplitter(*args, **kwargs)
