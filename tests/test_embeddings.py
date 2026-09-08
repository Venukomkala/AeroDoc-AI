import os
import sys

# Ensure project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion import load_all_raw_documents
from src.chunking import process_documents_into_chunks
from src.embeddings import EmbeddingEngine

def run_test():
    print("=== AeroDoc-AI: Testing Embeddings Pipeline ===")
    
    # 1. Load PDFs
    raw_folder = os.path.join("data", "raw")
    documents = load_all_raw_documents(raw_folder)
    
    # 2. Chunk documents
    chunks = process_documents_into_chunks(documents, chunk_size=300, chunk_overlap=50)
    print(f"\n[Success] Prepared {len(chunks)} text chunks.")

    # 3. Generate Embeddings
    embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    embeddings_matrix, embedded_chunks = embedder.generate_chunk_embeddings(chunks)

    print("\n--- EMBEDDING MATRIX DETAILS ---")
    print(f"Matrix Shape : {embeddings_matrix.shape}")
    print(f"Vector Dim   : {embedder.embedding_dim}")
    print(f"Sample Vector: {embeddings_matrix[0][:5]}... (first 5 dimensions)")
    print("---------------------------------")

if __name__ == "__main__":
    run_test()