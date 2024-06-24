from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding():
    embedding = OllamaEmbeddings(model='nomic-embed-text')
    return embedding