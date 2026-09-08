import os
import sys
from src.embeddings import EmbeddingEngine
from src.vector_store import FAISSVectorStore

class AeroRetriever:
    def __init__(self, processed_data_dir="data/processed", model_name="all-MiniLM-L6-v2"):
        """
        Initializes the retriever by loading the FAISS index and the embedding engine.
        """
        self.embedder = EmbeddingEngine(model_name=model_name)
        self.vector_store = FAISSVectorStore(embedding_dim=self.embedder.embedding_dim)
        
        print(f"[Retriever] Loading FAISS index and metadata from {processed_data_dir}...")
        self.vector_store.load(input_dir=processed_data_dir)

    def retrieve(self, query, top_k=3):
        print(f"[Retriever] Searching for top {top_k} results for query: '{query}'")
    
        query_vector = self.embedder.generate_query_embedding(query)
        scores, indices = self.vector_store.index.search(query_vector, top_k)

        results = []
        for rank in range(min(top_k, len(self.vector_store.metadata))):
            idx = indices[0][rank]
            score = float(scores[0][rank])

        # FAISS returns -1 for empty search slots
            if idx == -1 or idx >= len(self.vector_store.metadata):
                continue

            chunk_meta = self.vector_store.metadata[idx]
            results.append({
                "rank": rank + 1,
                "score": round(score, 4),
                "filename": chunk_meta["filename"],
                "page_number": chunk_meta["page_number"],
                "text": chunk_meta["text"]
            })

        return results

    def format_context(self, retrieved_chunks):
        """
        Formats retrieved chunks into a clean prompt context block with citations.
        """
        context_blocks = []
        for chunk in retrieved_chunks:
            citation = f"[Source: {chunk['filename']} | Page {chunk['page_number']}]"
            context_blocks.append(f"{citation}\n{chunk['text']}")

        return "\n\n---\n\n".join(context_blocks)