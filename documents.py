import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loading documents
def load_documents(path: str) -> list[Document]:
    docs_loader = PyPDFDirectoryLoader(path)
    return docs_loader.load()

# Embedding documents
def get_embedding_function() -> OllamaEmbeddings:
    embeddings = OllamaEmbeddings(model="exaone3.5:7.8b")
    return embeddings

# Split documents
def split_documents(docs: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(docs)

# Add to chroma
def add_to_chroma(chroma_path: str, chunks: list[Document]):
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Existing items: {len(existing_ids)}")

    # Only add new docs
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks) > 0:
        print(f"Adding {len(new_chunks)} new items")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

        db.add_documents(documents=new_chunks, ids=new_chunk_ids)
    else:
        print("No new items to add")

# generate chunk ID
def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if last_page_id == current_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks

def clear_chroma(chroma_path: str):
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        print(f"Deleted {chroma_path}")

if __name__ == "__main__":
    # clear_chroma("chroma")
    docs = load_documents("docs")
    print(docs)
    chunks = split_documents(docs)
    add_to_chroma("chroma", chunks)
    print("Done!")
