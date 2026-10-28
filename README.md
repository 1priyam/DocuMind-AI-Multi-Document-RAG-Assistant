# DocuMind AI — Multi-Document RAG Assistant

DocuMind AI is an advanced, production-grade Multi-Document Retrieval-Augmented Generation (RAG) assistant. It allows uploading and indexing multiple PDFs, text files, and markdown documents simultaneously, performing grounded synthesis with exact citations formatted as `(document.pdf, p. 3)`.

---

## 🌟 Key Features

1. **Multi-Document Ingestion & Management**
   - Upload multiple documents at once (PDF, TXT, MD).
   - Ingests chunks with `doc`, `doc_id`, `page`, and `text` metadata.
   - Dynamic document removal (`DELETE /documents/<id>`) with automated vector index rebuilding.
   - Per-document summary generation with Gemini right upon upload.

2. **Persistent Vector Store (FAISS on Disk)**
   - FAISS index and document metadata are automatically persisted to disk (`storage/` directory).
   - Vector store and document state survive server restarts seamlessly.

3. **Intelligent Document Filtering**
   - Checkboxes allow querying across all indexed documents or filtering queries to a specific subset of documents.
   - Candidate chunk over-sampling ensures top relevance within selected filters.

4. **Duplicate Detection via Cryptographic Hashing**
   - Computes SHA-256 hashes of uploaded files to automatically skip redundant uploads.

5. **Precise Citations & Sources Used**
   - Streaming responses cite exact sources formatted as `(filename.pdf, p. 3)`.
   - The UI displays an interactive "Sources used" accordion detailing document name, page, similarity score, and excerpt.

---

## 🚀 API Endpoints

- `GET /` — Serves the DocuMind AI interface.
- `POST /upload` — Accepts multiple files (`pdf` field), performs duplicate checks, chunks, embeds, creates per-document summaries, and updates the persistent index.
- `GET /documents` — Returns the list of all indexed documents with metadata, chunk counts, and summaries.
- `DELETE /documents/<id>` — Removes a document and dynamically rebuilds the FAISS index.
- `POST /chat` — Streams an answer via Server-Sent Events (SSE) with citations, document filtering support (`doc_ids`), and similarity-ranked sources.
- `POST /reset` — Clears conversation history for the current session.

---

## 🛠️ Quick Start

```bash
# 1. Activate your virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure Gemini API key
cp .env.example .env
# Add your GEMINI_API_KEY inside .env

# 4. Start the application
python app.py
# Open http://127.0.0.1:5000
```

---

## 🧪 Running Automated Tests

```bash
python test_rag.py
python test_app.py
```
