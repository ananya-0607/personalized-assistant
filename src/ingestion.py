from langchain_text_splitters import CharacterTextSplitter

def load_text(file_path):
    with open(file_path , "r" , encoding='utf-8') as f:
        return f.read()

def split_text(text,chunk_size=100,overlap=10):
    splitter=CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    chunks=splitter.split_text(text)
    return chunks