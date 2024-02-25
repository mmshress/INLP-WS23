from langchain_core.embeddings import Embeddings
from llama_index import ServiceContext, SimpleDirectoryReader, \
    StorageContext, VectorStoreIndex
from llama_index.vector_stores import OpensearchVectorClient, \
    OpensearchVectorStore

from llmlegalassistant.utils import Utils

# def metadata_add(x: str) -> str:
#     if type(x) is str:
#         return ", ".join(x)


class ArticlesIndexer:
    def __init__(
        self,
        embedding_model: Embeddings,
        chunking_strategy: str,
        verbose: bool = False,
        host: str = "localhost",
        port: int = 9200,
    ) -> None:
        embedding_model_index = embedding_model.model_name.split("/")[1].lower()
        self.INDEX_NAME = f"{0}-{1}"
        self.INDEX_BODY = {"settings": {"index": {"number_of_shards": 2}}}
        self.verbose = verbose
        self.client = OpensearchVectorClient(
            f"http://{host}:{port}",
            f"{embedding_model_index}-{chunking_strategy}",
            dim=1024,
            embedding_field="embedding",
            text_field="chunk",
            search_pipeline="hybrid-search-pipeline",
        )

        self.vector_store = OpensearchVectorStore(self.client)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.service_context = ServiceContext.from_defaults(
            embed_model=embedding_model, llm=None
        )
        self.utils = Utils()
        # self.metadata_dataframe = self.create_metadata_dataframe()

    # def get_metadata(self, filename):
    #     # This merged dataframe might contain duplicate rows, for metadata extraction we only use the first row found
    #     celex_number = filename.split("/")[-1].split(".")[0]
    #     metadata_row = self.metadata_dataframe[self.metadata_dataframe['CELEX number'] == celex_number].iloc[0]
    #     return {"celex_number": celex_number,
    #             'title': metadata_row["Title"],
    #             'published_date': metadata_row["Date of publication"],
    #             'start_date': metadata_row["Date of effect"],
    #             'end_date': metadata_row["Date of end of validity"]
    #             }
    #
    # def create_metadata_dataframe(self) -> pandas.DataFrame:
    #     csv_dir = self.utils.get_dataset_dir() + "/rawfiles/"
    #     dataframes = []
    #     for filename in os.listdir(csv_dir):
    #         if filename.endswith(".csv"):
    #             file_path = os.path.join(csv_dir, filename)
    #             dataframes.append(pandas.read_csv(file_path))
    #     merged_df = pandas.concat(dataframes)
    #     merged_df.replace('', pandas.NA, inplace=True)
    #     merged_df = merged_df.groupby('CELEX number').agg({'Title': 'first',
    #                                                        'Date of publication': 'first',
    #                                                        'Date of effect': 'first',
    #                                                        'Date of end of validity': 'first'}).reset_index()
    #     return merged_df

    def index_documents(self) -> VectorStoreIndex:
        files_dir = self.utils.get_dataset_dir() + "/articles/txts/"
        documents = SimpleDirectoryReader(files_dir).load_data()
        return VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=self.storage_context,
            service_context=self.service_context,
        )

    def upload_documents(self) -> None:
        pass
