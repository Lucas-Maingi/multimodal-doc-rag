from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    """
    Wrapper around HuggingFace SentenceTransformer.
    Converts text chunks and search queries into dense 384-dimensional vectors.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading Embedding Model: '{model_name}' (runs locally on cpu)...")
        self.model_name = model_name
        # Loads the neural network into RAM
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        print("Embedding model loaded successfully.")

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string (such as a user search query).
        Returns a Python list of 384 floating-point numbers.
        """
        cleaned = " ".join(text.split())
        vector = self.model.encode(cleaned, convert_to_numpy=True)
        return vector.tolist()

    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embeds multiple text chunks efficiently.
        Returns a list of 384-dimensional vectors.
        """
        cleaned_texts = [" ".join(t.split()) for t in texts]
        vectors = self.model.encode(cleaned_texts, convert_to_numpy=True, batch_size=32)
        return vectors.tolist()

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculates cosine similarity between two 384-dimensional vectors:
    Formula: (A . B) / (||A|| * ||B||)
    Returns a score between -1.0 and 1.0 (higher = more similar meaning).
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot_product / (norm_a * norm_b + 1e-8))

if __name__ == "__main__":
     # Smoke Test: Prove semantic understanding without keyword matching
    embedder = TextEmbedder()
    sentence_1 = "The tenant must pay a liquidated damages fee of $10,000 for early termination."
    sentence_2 = "Breaking the lease agreement prematurely incurs a ten thousand dollar financial penalty."
    sentence_3 = "The atmospheric weather in Nairobi is sunny and warm today."
    print("\n--- Generating Embeddings ---")
    vec_1 = embedder.embed_text(sentence_1)
    vec_2 = embedder.embed_text(sentence_2)
    vec_3 = embedder.embed_text(sentence_3)
    print(f"Vector 1 length: {len(vec_1)} dimensions")
    print(f"First 5 numbers of Vector 1: {[round(x, 4) for x in vec_1[:5]]}")
    sim_1_and_2 = cosine_similarity(vec_1, vec_2)
    sim_1_and_3 = cosine_similarity(vec_1, vec_3)
    print("\n--- Semantic Similarity Scores (Cosine Similarity) ---")
    print(f"Sentence 1 vs Sentence 2 (Lease Penalty vs Lease Penalty): {sim_1_and_2:.4f} (High Match!)")
    print(f"Sentence 1 vs Sentence 3 (Lease Penalty vs Nairobi Weather): {sim_1_and_3:.4f} (No Match!)")