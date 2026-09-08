import os
import json
import faiss
import numpy as np

class FAISSVectorStore:
    def __init__(self, embedding_dim=384):
        """
        Initializes an L2 inner-product FAISS index for normalized vector search.
        Default dimension 384 corresponds to all-MiniLM-L6-v2 output.
        """
        self.embedding_dim = embedding_dim
        # IndexFlatIP calculates inner product (equivalent to Cosine Similarity when vectors are normalized)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []

    def add_embeddings(self, embeddings_matrix, chunks):
        """
        Adds vector embeddings and corresponding chunk metadata into the FAISS index.
        """
        if embeddings_matrix.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected vector dimension {self.embedding_dim}, got {embeddings_matrix.shape[1]}")

        # Ensure correct float32 datatype for FAISS
        vectors = np.array(embeddings_matrix).astype("float32")
        self.index.add(vectors)

        # Store metadata (stripping embedding array to keep JSON lightweight)
        for chunk in chunks:
            meta = {
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"]
            }
            self.metadata.append(meta)

        print(f"[VectorStore] Added {vectors.shape[0]} vectors. Total index count: {self.index.ntotal}")

    def save(self, output_dir="data/processed"):
        """
        Persists the FAISS index and metadata JSON to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        index_path = os.path.join(output_dir, "faiss_index.bin")
        meta_path = os.path.join(output_dir, "metadata.json")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4)

        print(f"[VectorStore] Index saved to: {index_path}")
        print(f"[VectorStore] Metadata saved to: {meta_path}")

    def load(self, input_dir="data/processed"):
        """
        Loads an existing FAISS index and metadata JSON from disk.
        """
        index_path = os.path.join(input_dir, "faiss_index.bin")
        meta_path = os.path.join(input_dir, "metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing FAISS artifacts in {input_dir}")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"[VectorStore] Loaded {self.index.ntotal} vectors from {index_path}")