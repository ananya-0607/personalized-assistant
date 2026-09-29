# src/embeddings.py

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(text_chunks, persist_dir="./chroma_db"):
    """
    Converts text chunks into vectors locally
    and stores them in Chroma DB.
    """

    vectordb = Chroma.from_texts(
        texts=text_chunks,
        embedding=model,
        persist_directory=persist_dir
    )

    vectordb.persist()
    print("✅ Local embeddings created and stored.")
