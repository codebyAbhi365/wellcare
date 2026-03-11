# agent/rag_pipeline.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Setup local LLM and embeddings
Settings.llm = Ollama(model="llama3", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"  # Lightweight, fast, good quality
)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_knowledge_index():
    """Load and index the medical knowledge base"""
    collection = chroma_client.get_or_create_collection("nutriscan_knowledge")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load knowledge base documents
    docs = SimpleDirectoryReader("knowledge/").load_data()
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        show_progress=True
    )
    return index

def get_user_history_index(user_id: str):
    """Separate ChromaDB collection per user for their history"""
    collection = chroma_client.get_or_create_collection(f"user_{user_id}_history")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        [],  # Start empty, we add dynamically
        storage_context=storage_context
    )
    return index, collection