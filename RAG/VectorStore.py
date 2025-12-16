import faiss
import numpy as np
import pandas as pd
from Config import Config
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore

    
class VectorStore:
    def __init__(self):
        self.config = Config()
        self.model = SentenceTransformer(self.config.embeded_model_name)
        self.create_docs()

        self.embedding_function = lambda texts: self.model.encode(
            texts,
            batch_size=16,
            device="cpu"
        ).tolist()

    def create_docs(self):
        df = pd.read_csv(self.config.dataset)

        self.docs = []

        for _, row in df.iterrows():
            self.docs.append(
                Document(
                    page_content=f"{row['summary_text']}\n\nConversation:\n{row['original_msg']}",
                    metadata={
                        "topic": row["source_topic"],
                        "names": row["all_names"],
                        "locations": row["locations"],
                        "time": row["timing"],
                    }
                )
            )

    def load_vector_store(self):
        return FAISS.load_local(self.config.vector_store_path, embeddings=self.embedding_function, allow_dangerous_deserialization=True)

    def create_vector(self):
        embedding_dim = len(self.embedding_function(["hello world"])[0])
        index = faiss.IndexFlatL2(embedding_dim)

        # Create empty FAISS store
        self.vector_store = FAISS(
            embedding_function=self.embedding_function,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        metadatas = [doc.metadata for doc in self.docs]
        texts = [doc.page_content for doc in self.docs]

        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        self.vector_store.save_local(self.config.vector_store_path)