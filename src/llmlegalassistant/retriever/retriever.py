from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers import BM25Retriever,QueryFusionRetriever

class Retriever:
    def __init__(self,index,top_k,retriever_method) -> None:
        self.index = index
        self.top_k = top_k
        self.retriever = retriever_method
    
    def generate_retriver(self):
        if self.retriever == "VectorIndexRetriever":
            retriever = VectorIndexRetriever(index= self.index, similarity_top_k= self.top_k)
            
        elif self.retriever == "BM25Retriever":
            retriever = BM25Retriever(docstore = self.index.docstore, similarity_top_k= self.top_k)
        
        elif self.retriever == "QueryFusionRetriever":
            vector_retriever = self.index.as_retriever(similarity_top_k=2)
            bm25_retriever = BM25Retriever.from_defaults(docstore=self.index.docstore, similarity_top_k=2)
            retriever = QueryFusionRetriever([vector_retriever, bm25_retriever],similarity_top_k=2,num_queries=1,
                                             mode="reciprocal_rerank",use_async=True,verbose=True,
                                             # query_gen_prompt="...",  # we could override the query generation prompt here
                                             )
        return retriever
