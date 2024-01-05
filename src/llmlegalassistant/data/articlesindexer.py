# import os

# from elasticsearch import Elasticsearch

# from llmlegalassistant.utils import Utils


# class ArticlesIndexer:
#     def __init__(self, verbose=False,  host='localhost', port=9200):
#         self.verbose = verbose

#         # self.es = Elasticsearch([{'host': host, 'port': port}])
#         self.elasticsearch = Elasticsearch("http://localhost:9200")
#         self.utils = Utils()

#     def index(self, index_name):
#         files_dir = self.utils.get_file_dir("txt")

#         self._create_index()
#         for filename in os.listdir(files_dir):
#             if filename.endswith(".txt"):
#                 file_path = os.path.join(files_dir, filename)

#                 with open(file_path, "r", encoding="utf-8") as file:
#                     article = file.read()
#                     celex_number = filename.split(".")[0]
#                     result = self._upload_document(article, celex_number)

#     def _create_index(self) -> None:
#         schema = {
#             "mappings": {
#                 "properties": {
#                     "legal_text": {"type": "text"},
#                     "celex_number": {"type": "keyword"}
#                 }
#             }
#         }

#         # headers = self.es.transport.headers
#         self.elasticsearch.indices.create(index=self.index_name, body=schema)

#     def upload_document(self, document, id) -> any | None:
#         body = {
#             "celex": id,
#             "document": document
#         }

#         result = self.elasticsearch.index(index=self.index_name, body=body)
#         return result

#     def upload_documents(self, index) -> None:


#     # Example usage
#     def upload_documents(index_name):

#         es_populator = ElasticsearchPopulator(index_name='text')
#         utils = Utils()

#         es_populator.create_index()

#         text_files_directory = utils.get_file_dir("txt")

#         # Iterate over each text file in the directory  and extract celexnumber
#         for filename in os.listdir(text_files_directory):
#             if filename.endswith(".txt"):
#                 file_path = os.path.join(text_files_directory, filename)

#                 with open(file_path, 'r', encoding='utf-8') as file:
#                     legal_text = file.read()

#                     celex_number = os.path.splitext(filename)[0]

#                     # Push the document to Elasticsearch
#                     es_populator.push_document(legal_text=legal_text, celex_number=celex_number)
#                     print("Index created successfully")
# noqa: 401
