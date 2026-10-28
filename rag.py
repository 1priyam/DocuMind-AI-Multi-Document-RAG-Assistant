import hashlib
import io
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Generator, List, Tuple

import faiss
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader

EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"]


@dataclass
class Chunk:
    doc: str
    page: int
    text: str
    doc_id: str = ""


class PDFRAG:
    def __init__(
        self,
        embedder,
        chunk_words: int = 250,
        overlap: int = 50,
        storage_dir: str | None = None,
    ):
        self._embedder = embedder
        self.chunk_words = chunk_words
        self.overlap = overlap
        self.chunks: List[Chunk] = []
        self.index: faiss.IndexFlatIP | None = None
        # Format: {doc_id: {"id": str, "name": str, "chunk_count": int, "hash": str, "summary": str}}
        self.documents: dict[str, dict] = {}
        self.storage_dir = storage_dir
        self.last_sources: List[Tuple[Chunk, float]] = []

        if self.storage_dir:
            self.load_from_disk()

    @property
    def embedder(self):
        if not hasattr(self._embedder, "encode") and callable(self._embedder):
            self._embedder = self._embedder()
        return self._embedder

    def generate_summary(
        self, text_sample: str, api_key: str | None, model: str = DEFAULT_LLM
    ) -> str:
        """Generate a concise 2-3 sentence overview of the uploaded document with automatic retry and model fallback."""
        if not api_key:
            return "Summary unavailable (Gemini API key not provided)."
        
        models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
        client = genai.Client(api_key=api_key)
        prompt = (
            "Provide a clear, informative 2-3 sentence summary of the key subjects and findings "
            "in this document excerpt:\n\n"
            f"{text_sample[:4000]}"
        )

        last_err = None
        for current_model in models_to_try:
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model=current_model,
                        contents=prompt,
                    )
                    return (response.text or "").strip()
                except Exception as e:
                    last_err = e
                    err_str = str(e)
                    if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str:
                        time.sleep(1.2 * (attempt + 1))
                        continue
                    else:
                        break  # Try next model

        return f"Summary unavailable: {last_err}"

    def extract_text_chunks(
        self, file_bytes: bytes, filename: str, doc_id: str
    ) -> Tuple[List[Chunk], str]:
        """Extract text chunks and return combined sample text for summary generation."""
        lower_name = filename.lower()
        chunks: List[Chunk] = []
        sample_text_parts = []

        if lower_name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file_bytes))
            for page_idx, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                sample_text_parts.append(text)
                words = text.split()
                step = max(1, self.chunk_words - self.overlap)
                for i in range(0, len(words), step):
                    chunk_slice = words[i : i + self.chunk_words]
                    chunk_text = " ".join(chunk_slice).strip()
                    if chunk_text:
                        chunks.append(
                            Chunk(
                                doc=filename,
                                page=page_idx + 1,
                                text=chunk_text,
                                doc_id=doc_id,
                            )
                        )
        elif lower_name.endswith((".txt", ".md")):
            text = file_bytes.decode("utf-8", errors="replace").strip()
            sample_text_parts.append(text)
            words = text.split()
            step = max(1, self.chunk_words - self.overlap)
            for i in range(0, len(words), step):
                chunk_slice = words[i : i + self.chunk_words]
                chunk_text = " ".join(chunk_slice).strip()
                if chunk_text:
                    # Text/Markdown files are treated as Page 1 or single-document
                    chunks.append(
                        Chunk(
                            doc=filename,
                            page=1,
                            text=chunk_text,
                            doc_id=doc_id,
                        )
                    )
        else:
            # Fallback text attempt
            try:
                text = file_bytes.decode("utf-8", errors="replace").strip()
                sample_text_parts.append(text)
                words = text.split()
                step = max(1, self.chunk_words - self.overlap)
                for i in range(0, len(words), step):
                    chunk_slice = words[i : i + self.chunk_words]
                    chunk_text = " ".join(chunk_slice).strip()
                    if chunk_text:
                        chunks.append(
                            Chunk(
                                doc=filename,
                                page=1,
                                text=chunk_text,
                                doc_id=doc_id,
                            )
                        )
            except Exception:
                pass

        if not chunks:
            raise ValueError(
                f"No extractable text found in '{filename}'. Scanned or image-only documents require OCR."
            )

        combined_sample = " ".join(sample_text_parts)
        return chunks, combined_sample

    def add_document(
        self,
        file_bytes: bytes,
        filename: str,
        api_key: str | None = None,
        model: str = DEFAULT_LLM,
    ) -> dict:
        """Add a document to the persistent FAISS index with duplicate check and summary generation."""
        # 1. Duplicate check via SHA-256 hash
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        for doc_id, doc_info in self.documents.items():
            if doc_info.get("hash") == file_hash:
                return {
                    "status": "duplicate",
                    "doc_id": doc_id,
                    "name": doc_info["name"],
                    "chunk_count": doc_info["chunk_count"],
                    "summary": doc_info.get("summary", ""),
                    "message": f"'{filename}' was already indexed as '{doc_info['name']}'.",
                }

        doc_id = uuid.uuid4().hex[:10]

        # 2. Extract chunks
        new_chunks, sample_text = self.extract_text_chunks(file_bytes, filename, doc_id)

        # 3. Generate embeddings
        embeddings = self.embedder.encode(
            [c.text for c in new_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

        # Add chunks to existing FAISS index
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.chunks.extend(new_chunks)

        # 4. Generate per-document summary
        summary = self.generate_summary(sample_text, api_key=api_key, model=model)

        # 5. Store document record
        self.documents[doc_id] = {
            "id": doc_id,
            "name": filename,
            "chunk_count": len(new_chunks),
            "hash": file_hash,
            "summary": summary,
        }

        # 6. Persist to disk
        self.save_to_disk()

        return {
            "status": "indexed",
            "doc_id": doc_id,
            "name": filename,
            "chunk_count": len(new_chunks),
            "summary": summary,
            "message": f"Successfully indexed '{filename}' ({len(new_chunks)} chunks).",
        }

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and rebuild the FAISS index from remaining chunks."""
        if doc_id not in self.documents:
            return False

        del self.documents[doc_id]
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]

        if not self.chunks:
            self.index = None
        else:
            # Rebuild index from remaining chunks
            embeddings = self.embedder.encode(
                [c.text for c in self.chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

        self.save_to_disk()
        return True

    def get_documents_list(self) -> List[dict]:
        return list(self.documents.values())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: List[str] | None = None,
    ) -> List[Tuple[Chunk, float]]:
        if not self.index or not self.chunks:
            return []

        q_emb = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        filter_active = bool(doc_ids and len(doc_ids) > 0)
        # When filtering by document, query extra chunks to find top matches within selected documents
        search_k = min(len(self.chunks), max(top_k * 4, 25) if filter_active else top_k)
        scores, indices = self.index.search(np.ascontiguousarray(q_emb, dtype=np.float32), search_k)

        results: List[Tuple[Chunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                if filter_active and chunk.doc_id not in doc_ids:
                    continue
                results.append((chunk, float(score)))
                if len(results) >= top_k:
                    break
        return results

    def stream_answer(
        self,
        question: str,
        history: list[dict],
        api_key: str,
        model: str,
        top_k: int = 5,
        doc_ids: List[str] | None = None,
    ) -> Generator[str, None, None]:
        self.last_sources = self.retrieve(question, top_k=top_k, doc_ids=doc_ids)

        # Context formatted specifically as [Doc: <doc_name> | Page <page_number>]
        context_items = [
            f"[Doc: {c.doc} | Page {c.page}]:\n{c.text}" for c, _ in self.last_sources
        ]
        context_str = "\n\n---\n\n".join(context_items) if context_items else "No relevant document chunks found."

        system_instruction = (
            "You are DocuMind AI, an advanced Multi-Document RAG Assistant.\n"
            "Answer the user's question accurately using ONLY the provided document excerpts.\n"
            "If the answer cannot be found in the provided excerpts, politely state that the documents do not contain that information.\n"
            "Format your citations strictly like (document_name, p. page_number), for example (report.pdf, p. 3).\n"
            "When facts come from different documents, clearly attribute each fact to its source document and page.\n\n"
            f"Document Excerpts:\n{context_str}"
        )

        client = genai.Client(api_key=api_key)

        contents = []
        for h in history:
            role = "user" if h.get("role") == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=h.get("content", ""))]
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=question)]
            )
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        )

        models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
        last_error = None
        stream_started = False

        for current_model in models_to_try:
            for attempt in range(2):
                try:
                    response_stream = client.models.generate_content_stream(
                        model=current_model,
                        contents=contents,
                        config=config,
                    )
                    for chunk in response_stream:
                        if chunk.text:
                            stream_started = True
                            yield chunk.text
                    return
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    # If stream already began emitting tokens, do not try fallback mid-stream
                    if stream_started:
                        raise e
                    # If transient high-demand error, retry with backoff or switch model
                    if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    else:
                        break  # Non-transient error, try next model

        if last_error:
            raise last_error

    def save_to_disk(self) -> None:
        """Persist FAISS index and documents/chunks metadata to storage_dir."""
        if not self.storage_dir:
            return
        os.makedirs(self.storage_dir, exist_ok=True)
        index_path = os.path.join(self.storage_dir, "index.faiss")
        meta_path = os.path.join(self.storage_dir, "metadata.json")

        if self.index is not None and len(self.chunks) > 0:
            faiss.write_index(self.index, index_path)
        elif os.path.exists(index_path):
            os.remove(index_path)

        meta_data = {
            "documents": self.documents,
            "chunks": [
                {
                    "doc": c.doc,
                    "page": c.page,
                    "text": c.text,
                    "doc_id": c.doc_id,
                }
                for c in self.chunks
            ],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

    def load_from_disk(self) -> None:
        """Load FAISS index and documents/chunks metadata from storage_dir."""
        if not self.storage_dir or not os.path.exists(self.storage_dir):
            return
        index_path = os.path.join(self.storage_dir, "index.faiss")
        meta_path = os.path.join(self.storage_dir, "metadata.json")

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                self.documents = meta_data.get("documents", {})
                raw_chunks = meta_data.get("chunks", [])
                self.chunks = [
                    Chunk(
                        doc=c.get("doc", ""),
                        page=c.get("page", 1),
                        text=c.get("text", ""),
                        doc_id=c.get("doc_id", ""),
                    )
                    for c in raw_chunks
                ]
                if os.path.exists(index_path) and self.chunks:
                    self.index = faiss.read_index(index_path)
            except Exception as e:
                print(f"[PDFRAG] Warning: Could not restore vector store from disk: {e}")
