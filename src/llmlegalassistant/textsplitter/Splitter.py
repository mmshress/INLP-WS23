from abc import ABC, abstractmethod


class Splitter(ABC):
    def __init__(
        self, chunk_size: int = 512, overlap_size: int = 50, verbose: bool = False
    ):
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    @abstractmethod
    def split(
        self, documents: list, noofdocuments: int = 50
    ) -> list[dict[str, str]] | None:
        raise NotImplementedError
