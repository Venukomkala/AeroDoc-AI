import os
import sys

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion import load_all_raw_documents
from src.chunking import process_documents_into_chunks
from src.embeddings import EmbeddingEngine
from src.vector_store import FAISSVectorStore

def run_test():
    print("=== AeroDoc-AI: End-to-End Vector Indexing Test ===")

    # 1. Load PDFs
    raw_folder = os.path.join("data", "raw")
    documents = load_all_raw_documents(raw_folder)

    # 2. Chunk text
    chunks = process_documents_into_chunks(documents, chunk_size=300, chunk_overlap=50)

    # 3. Embed chunks
    embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    embeddings_matrix, embedded_chunks = embedder.generate_chunk_embeddings(chunks)

    # 4. Build & save FAISS Index
    vector_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
    vector_store.add_embeddings(embeddings_matrix, embedded_chunks)
    vector_store.save(output_dir="data/processed")

    print("\n[Success] FAISS index built and stored inside data/processed/")

if __name__ == "__main__":
    run_test()