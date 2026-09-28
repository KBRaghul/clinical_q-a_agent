from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_DIR = ".chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_vectorstore = None


def get_vectorstore() -> Chroma:
    """Lazily load and cache the persisted ChromaDB vectorstore"""
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return _vectorstore


def retrieve(query: str, k: int = 4):
    """Return the top-k most similar chunks for a query"""
    return get_vectorstore().similarity_search(query, k=k)
