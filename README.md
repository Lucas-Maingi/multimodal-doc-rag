# Multi-Modal Document Intelligence & RAG Engine

A production-ready Retrieval-Augmented Generation (RAG) system designed to process unstructured documents (digital PDFs, scanned contracts, images), extract text via OCR, generate dense semantic embeddings using Sentence-Transformers, and index them in a persistent ChromaDB vector store. Features grounded LLM synthesis with anti-hallucination verification and page-level source citations served via FastAPI.

## Architecture

```
[ INGESTION PIPELINE ]
  User Upload (PDF / Scanned Image)
            │
            ▼
    parser.py / ocr.py  ──► Detects digital text vs scanned pixels (Tesseract OCR)
            │
            ▼
    chunker.py          ──► Splits into 500-char semantic chunks (50-char overlap)
            │                Preserves metadata: {doc_id, page_number, chunk_id}
            ▼
    embedder.py         ──► Generates dense 384-d vectors (all-MiniLM-L6-v2)
            │
            ▼
    vector_store.py     ──► Indexes vectors + metadata in persistent ChromaDB

════════════════════════════════════════════════════════════════════════════════════

[ RETRIEVAL & GROUNDED SYNTHESIS ]
  User Question: "What is the penalty for early termination?"
            │
            ▼
    embedder.py         ──► Embeds user query into 384-d vector space
            │
            ▼
    vector_store.py     ──► Cosine Similarity / ANN search in ChromaDB (Top-K)
            │
            ▼
    generator.py        ──► Strict Grounding: "Answer ONLY from context. Cite pages."
            │
            ▼
    server.py (FastAPI) ──► JSON: { "answer": "...", "citations": [Page 4], "confidence": 0.94 }
```

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| Ingestion & OCR | `pypdf`, `pytesseract`, `Pillow` |
| Text Chunking | Semantic Sliding Window with Metadata Tracking |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (Local, 384-dim) |
| Vector Database | `ChromaDB` (Persistent local storage) |
| Serving & API | `FastAPI`, `Pydantic v2`, `Uvicorn` |
| Containerization | `Docker` |

## Key Engineering Features

- **Multi-Modal Document Parsing:** Seamlessly handles digital text and scanned image PDFs with automatic OCR fallback.
- **Context-Preserving Chunking:** 50-character overlap prevents information clipping across clause boundaries.
- **Local Dense Embeddings:** Runs 100% locally on CPU with zero external API fees or data privacy leaks.
- **Anti-Hallucination Grounding:** System prompts enforce strict context boundaries and reject queries outside the document corpus.
- **Auditable Source Citations:** Every returned answer includes exact page numbers and snippet references.

## Getting Started

```bash
git clone https://github.com/Lucas-Maingi/multimodal-doc-rag.git
cd multimodal-doc-rag
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate
pip install -r requirements.txt
uvicorn server:app --reload
```

## License

MIT