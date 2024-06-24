import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from get_embedding import get_embedding

CHROMA_PATH = 'chroma'
DATA_PATH = 'data'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true', help='Reset the database')
    args = parser.parse_args()
    if args.reset:
        print('Resetting database')
        clear_database()
    
    documents = load_docs()
    chunks = split_docs(documents)
    add_to_chroma(chunks)
    
def load_docs():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    last_page = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get('source')
        page = chunk.metadata.get('page')
        current_page = f'{source}:{page}'
        
        if current_page == last_page:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
            
        chunk_id = f'{current_page}:{current_chunk_index}'
        last_page = current_page
        chunk.metadata['id'] = chunk_id
        
    return chunks

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding()
    )
    
    chunks_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items['ids'])
    
    print(f'Adding {len(chunks_ids)} chunks to the database')
    
    new_chunks = []
    for chunk in chunks_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("👉 No new documents to add")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
            
if __name__ == '__main__':
    main()
