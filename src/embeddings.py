import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the SentenceTransformer embedding model.
        Default model: 'all-MiniLM-L6-v2' (fast, lightweight 384-dim embeddings).
        """
        print(f"[Embeddings] Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embeddings] Model loaded successfully. Dimension size: {self.embedding_dim}")

    def generate_chunk_embeddings(self, chunks):
        """
        Generates vector embeddings for a list of chunk dictionaries.
        Returns a tuple of (embeddings_matrix, enriched_chunks).
        """
        if not chunks:
            return np.array([]), []

        texts = [chunk["text"] for chunk in chunks]
        print(f"[Embeddings] Generating embeddings for {len(texts)} chunks...")
        
        # Generate normalized embeddings suitable for cosine/inner product search
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings_matrix = np.array(embeddings).astype("float32")

        # Attach embedding array to each chunk dictionary for convenience
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings_matrix[i]

        return embeddings_matrix, chunks

    def generate_query_embedding(self, query):
        """
        Generates a normalized embedding vector for a single user search query.
        """
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        return np.array(query_embedding).astype("float32")