from langchain_ollama import OllamaEmbeddings

from src.config import OLLAMA_BASE_URL


def get_embedding_function():
    embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model="bge-m3")
    return embeddings
