# src/retrieval.py

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_relevant_chunks(query, persist_dir="./chroma_db", k=3):
    """
    Finds top-k semantically similar chunks
    using local embeddings.
    """

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=model
    )

    docs = vectordb.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
