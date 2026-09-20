import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from embedder import TextEmbedder

class DocumentVectorStore:
    """
    Manages persistent vector indexing and semantic retrieval using ChromaDB.
    """
    def __init__(self, persist_dir: str = "chromadb", collection_name: str = "contracts_corpus"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        print(f"Initializing ChromaDB persistent storage at: '{persist_dir}'...")
        # Initialize persistent client (saves data to disk)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Get or create the vector collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use Cosine Distance for semantic similarity
        )
        print(f"ChromaDB ready. Collection '{collection_name}' currently has {self.collection.count()} chunks.")

    def add_document_chunks(self, chunks: List[Dict], embedder: TextEmbedder) -> int:
        """
        Takes a list of chunk dicts from chunker.py, embeds them, and inserts into ChromaDB.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [c["metadata"]["chunk_id"] for c in chunks]

        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = embedder.embed_chunks(texts)

        print(f"Inserting {len(texts)} chunks into ChromaDB...")
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print("Insertion complete.")
        return len(texts)

    def search_similar(self, query: str, embedder: TextEmbedder, top_k: int = 3) -> List[Dict]:
        """
        Embeds a user query and finds the top-K most semantically relevant chunks.
        """
        query_vector = embedder.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc_text, meta, dist in zip(docs, metas, distances):
                # Cosine distance in ChromaDB: similarity = 1 - distance
                similarity_score = max(0.0, 1.0 - dist)
                formatted_results.append({
                    "text": doc_text,
                    "metadata": meta,
                    "similarity_score": round(similarity_score, 4)
                })

        return formatted_results

    def get_indexed_documents(self) -> List[str]:
        """Returns a list of unique document names currently in the database."""
        all_records = self.collection.get(include=["metadatas"])
        if not all_records or not all_records["metadatas"]:
            return []
        unique_docs = list(set(m["doc_name"] for m in all_records["metadatas"]))
        return unique_docs

if __name__ == "__main__":
    from chunker import chunk_text

    # Smoke Test: End-to-end Chunking -> Embedding -> ChromaDB -> Semantic Search
    embedder_instance = TextEmbedder()
    store = DocumentVectorStore(persist_dir="chroma_db_test")

    sample_doc = (
        "CLAUSE 4.1: The tenant agrees to pay monthly rent of $3,500 due on the first day of each month. "
        "CLAUSE 4.2: Late payments exceeding 5 calendar days incur a penalty fee of $250. "
        "CLAUSE 8.1: No pets of any kind (including dogs, cats, reptiles) are permitted on premises without written consent. "
        "CLAUSE 12.3: Security deposit of $7,000 shall be returned within 14 days of lease termination."
    )

    chunks = chunk_text(sample_doc, doc_name="apartment_lease.pdf", page_number=2, chunk_size=500, chunk_overlap=50)
    store.add_document_chunks(chunks, embedder_instance)

    # Test Search 1: Ask about animals
    query_1 = "Can I bring my golden retriever dog?"
    print(f"\nUser Query 1: '{query_1}'")
    hits_1 = store.search_similar(query_1, embedder_instance, top_k=3)
    for h in hits_1:
        print(f"  [Found Chunk ID]: {h['metadata']['chunk_id']} (Page {h['metadata']['page']})")
        print(f"  [Similarity Score]: {h['similarity_score']}")
        print(f"  [Text Excerpt]: \"{h['text']}\"")
    # Test Search 2: Ask about payment overdue
    query_2 = "What is the fine if I pay rent late?"
    print(f"\nUser Query 2: '{query_2}'")
    hits_2 = store.search_similar(query_2, embedder_instance, top_k=3)
    for h in hits_2:
        print(f"  [Found Chunk ID]: {h['metadata']['chunk_id']} (Page {h['metadata']['page']})")
        print(f"  [Similarity Score]: {h['similarity_score']}")
        print(f"  [Text Excerpt]: \"{h['text']}\"")