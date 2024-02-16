from langchain.text_splitter import RecursiveCharacterTextSplitter

from llmlegalassistant.textsplitter import Splitter


class RecursiveTextSplitter(Splitter):
    def __init__(
        self,
        seperators: list[str],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        verbose: bool = False,
    ) -> None:
        super().__init__(verbose)
        """
        Sets up the chunk size and overlap size

        Parameters:
        -----------
        chunk_size : int
            The size of the chunk
        chunk_overlap : int
            The size of the overlap
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.seperators = seperators
        # use these seperators to split the text
        # the seperators: [" ", "\n", "\t", "\n\n"]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            seperators=self.seperators,
        )

    def split(self, documents: list, noofdocuments: int = 50) -> list | None:
        """
        Creates chunks of the documents and returns number of documents mentioned

        Parameters:
        -----------
        documents : list
            The documents that are being chunked
        n : int
            The number of documents being returned

        Returns:
        --------
        list | None
            A list of chunks created from the list of documents
        """
        self.documents = self.text_splitter.create_documents(documents)
        self.noofdocuments = noofdocuments

        return self.documents[: self.noofdocuments]
