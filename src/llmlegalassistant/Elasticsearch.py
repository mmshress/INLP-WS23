import os
from elasticsearch import Elasticsearch
from llmlegalassistant.utils import Utils


class ElasticsearchPopulator:
    def __init__(self, index_name='your_index_name',  host='localhost', port=9200):
        self.index_name = index_name
        
        #self.es = Elasticsearch([{'host': host, 'port': port}])
        self.es = Elasticsearch("http://localhost:9200")

    def create_index(self):
        body = {
            "mappings": {
                "properties": {
                    "legal_text": {"type": "text"},
                    "celex_number": {"type": "keyword"}
                }
            }
        }
        

        #headers = self.es.transport.headers
        self.es.indices.create(index=self.index_name, body=body)

    def push_document(self, legal_text, celex_number):
        document = {
            "legal_text": legal_text,
            "celex_number": celex_number
        }

        result = self.es.index(index=self.index_name, body=document)
        return result

# Example usage
if __name__ == "__main__":
    
    es_populator = ElasticsearchPopulator(index_name='text')
    utils = Utils()
   
    es_populator.create_index()

    text_files_directory = utils.get_file_dir("txt")

    # Iterate over each text file in the directory  and extract celexnumber
    for filename in os.listdir(text_files_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(text_files_directory, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                legal_text = file.read()

            celex_number = os.path.splitext(filename)[0]

            # Push the document to Elasticsearch
            es_populator.push_document(legal_text=legal_text, celex_number=celex_number)
            print("Index created successfully")

