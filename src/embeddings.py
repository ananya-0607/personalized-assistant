import os
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def create_embeddings(chunks,persist_dir="./chroma_db"):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("GEMINI_API_KEY"))
    vectordb=Chroma.from_texts(texts=chunks,embedding=embeddings,persist_directory=persist_dir)
    vectordb.persist()
    return vectordb