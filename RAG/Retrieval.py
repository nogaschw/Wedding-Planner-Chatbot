import re

class Retrieval:
    def __init__(self, vectorstore, k=30):
        self.vector_store = vectorstore
        self.k = k

    def get_retriever(self, topic=None):
        search_kwargs = {"k": self.k / 2,  "fetch_k": self.k, "lambda_mult": 0.65}
        
        if topic:
            search_kwargs["filter"] = {"topic": topic}
        
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )