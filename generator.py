import os
from typing import List, Dict
from dotenv import load_dotenv
from google import genai

load_dotenv()

class GroundedGenerator:
    """
    Generates answers strictly grounded in retrieved document context.
    Uses Google Gemini API with anti-hallucination system prompting.
    """
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.8-flash"
        print(f"GroundedGenerator initialized with model: {self.model_name}")

    def generate_answer(self, query:str, retrieved_chunks: List[Dict]) -> Dict:
        """
        Takes a user query and retrieved context chunks, then generates
        a grounded answer with source citations.
        """
        if not retrieved_chunks:
            return{
                "answer": "No relevant context was found in the indexed documents.",
                "citations": [],
                "chunks_used": 0
            }

        # Build the context block from retrieved chunks
        context_block = ""
        for i, chunk in enumerate(retrieved_chunks):
            doc = chunk["metadata"].get("doc_name", "Unknown")
            page = chunk["metadata"].get("page", "?")
            score = chunk.get("similarity_score", 0)
            context_block += f"\n[Source {i+1} | Document: {doc} | Page: {page} | Relevance: {score}]\n"
            context_block += chunk["text"] + "\n"

        # The anti-hallucination system prompt
        system_prompt = (
            "You are an expert document analyst. Your task is to answer the user's question "
            "using ONLY the provided document excerpts below. Follow these rules strictly:\n\n"
            "1. Base your answer EXCLUSIVELY on the provided context. Do not use external knowledge.\n"
            "2. If the answer cannot be found in the context, respond: 'This information is not present in the provided documents.'\n"
            "3. Always cite the source document name and page number in your answer.\n"
            "4. Be concise and precise. Quote exact figures, dates, and dollar amounts from the context.\n"
            "5. If multiple sources contain relevant information, synthesize them and cite all sources.\n"
        )

        user_message = f"DOCUMENT CONTEXT:\n{context_block}\n\nUSER QUESTION: {query}"

        # Call Gemini API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_message,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.1, # Low temperature = factual, not creative
                "max_output_tokens": 500,
            }
        )

        answer_text = response.text.strip()

        # Extract citation metadata from the chunks used
        citations = []
        for chunk in retrieved_chunks:
            citations.append({
                "doc_name": chunk["metadata"].get("doc_name", "Unknown"),
                "page": chunk["metadata"].get("page", "?"),
                "similarity_score": chunk.get("similarity_score", 0)
            })

        return{
            "answer": answer_text,
            "citations": citations,
            "chunks_used": len(retrieved_chunks)
        }

if __name__ == "__main__":
    # Smoke Test: Simulate retrieved chunks and generate an answer
    generator = GroundedGenerator()
    fake_retrieved_chunks = [
        {
            "text": "CLAUSE 4.2: Late payments exceeding five (5) calendar days shall incur a penalty fee of $250 USD per occurrence.",
            "metadata": {"doc_name": "commercial_lease.pdf", "page": 2, "chunk_id": "lease_p2_c1"},
            "similarity_score": 0.82
        },
        {
            "text": "CLAUSE 4.1: Monthly rent shall be $3,500 USD payable on the first business day of each calendar month via wire transfer.",
            "metadata": {"doc_name": "commercial_lease.pdf", "page": 2, "chunk_id": "lease_p2_c0"},
            "similarity_score": 0.76
        }
    ]
    test_query = "What happens if I pay my rent late?"
    print(f"\nUser Query: '{test_query}'")
    print("Generating grounded answer from retrieved context...\n")
    result = generator.generate_answer(test_query, fake_retrieved_chunks)
    print(f"Answer: {result['answer']}")
    print(f"\nCitations: {result['citations']}")
    print(f"Chunks Used: {result['chunks_used']}")