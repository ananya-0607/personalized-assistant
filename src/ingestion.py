# src/ingestion.py

from langchain_text_splitters import CharacterTextSplitter
from embeddings import create_embeddings

FILE_PATH = "data/Web Tech Notes.txt"

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text):
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    text = load_text(FILE_PATH)
    chunks = split_text(text)
    create_embeddings(chunks)
