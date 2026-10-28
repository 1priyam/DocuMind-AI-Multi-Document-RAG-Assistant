import json
import os
import uuid

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    session,
    stream_with_context,
)

from rag import PDFRAG, EMBED_MODEL, DEFAULT_LLM

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit for multiple files
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Persistent vector store directory
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")

# Lazy-load sentence transformer embedding model to ensure instant startup
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        print("[DocuMind] Loading sentence transformer embedding model...")
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# Initialize persistent multi-document RAG engine
rag = PDFRAG(get_embedder, storage_dir=STORAGE_DIR)
print(f"[DocuMind] RAG engine initialized with {len(rag.documents)} document(s) and {len(rag.chunks)} chunk(s).")

# In-memory per-browser-session chat history: {sid: {"history": list}}
STORE: dict[str, dict] = {}


def sid() -> str:
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def get_session_history() -> list:
    s_id = sid()
    if s_id not in STORE:
        STORE[s_id] = {"history": []}
    return STORE[s_id]["history"]


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/")
def index():
    return render_template(
        "index.html",
        default_model=DEFAULT_LLM,
        has_env_key=bool(os.getenv("GEMINI_API_KEY")),
        documents=rag.get_documents_list(),
        total_chunks=len(rag.chunks),
    )


@app.get("/documents")
def list_documents():
    """List all indexed documents and summary metadata."""
    return jsonify(
        documents=rag.get_documents_list(),
        total_chunks=len(rag.chunks),
    )


@app.delete("/documents/<doc_id>")
def delete_document(doc_id: str):
    """Remove a document from the persistent index and rebuild."""
    removed = rag.remove_document(doc_id)
    if not removed:
        return jsonify(error="Document not found"), 404
    return jsonify(
        ok=True,
        documents=rag.get_documents_list(),
        total_chunks=len(rag.chunks),
        message="Document successfully removed and index rebuilt.",
    )


@app.post("/upload")
def upload():
    """Accept multiple PDF/text files and index them into the persistent RAG vector store."""
    files = []
    # Collect files from any possible field key (pdf, files, pdf[], files[], etc.)
    for key in request.files.keys():
        for f in request.files.getlist(key):
            if f and f.filename and f.filename.strip():
                files.append(f)

    if not files:
        return jsonify(error="Please select one or more documents to upload."), 400

    api_key = request.form.get("api_key") or os.getenv("GEMINI_API_KEY")
    model = request.form.get("model") or DEFAULT_LLM

    results = []
    indexed_count = 0
    duplicate_count = 0
    error_count = 0

    for file in files:
        filename = file.filename
        if not filename:
            continue

        lower = filename.lower()
        if not (lower.endswith(".pdf") or lower.endswith(".txt") or lower.endswith(".md")):
            results.append({
                "status": "error",
                "name": filename,
                "message": "Unsupported file format. Please upload PDF, TXT, or MD files.",
            })
            error_count += 1
            continue

        try:
            file_bytes = file.read()
            res = rag.add_document(
                file_bytes=file_bytes,
                filename=filename,
                api_key=api_key,
                model=model,
            )
            results.append(res)
            if res.get("status") == "indexed":
                indexed_count += 1
            elif res.get("status") == "duplicate":
                duplicate_count += 1
            elif res.get("status") == "error":
                error_count += 1
        except ValueError as e:
            results.append({
                "status": "error",
                "name": filename,
                "message": str(e),
            })
            error_count += 1
        except Exception as e:
            err_msg = str(e) if str(e) else repr(e)
            results.append({
                "status": "error",
                "name": filename,
                "message": f"Processing failed: {err_msg}",
            })
            error_count += 1

    return jsonify(
        results=results,
        indexed_count=indexed_count,
        duplicate_count=duplicate_count,
        error_count=error_count,
        documents=rag.get_documents_list(),
        total_chunks=len(rag.chunks),
    )


@app.post("/chat")
def chat():
    if not rag.chunks:
        return jsonify(error="Please upload and process at least one document first."), 400

    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    api_key = data.get("api_key") or os.getenv("GEMINI_API_KEY")
    model = data.get("model") or DEFAULT_LLM
    top_k = int(data.get("top_k") or 5)
    doc_ids = data.get("doc_ids")  # Optional document filter list

    if not question:
        return jsonify(error="Question is empty."), 400
    if not api_key:
        return jsonify(error="Please provide your Gemini API key (or set it in .env)."), 400

    history = get_session_history()

    @stream_with_context
    def generate():
        parts = []
        try:
            for token in rag.stream_answer(
                question=question,
                history=history,
                api_key=api_key,
                model=model,
                top_k=top_k,
                doc_ids=doc_ids,
            ):
                parts.append(token)
                yield sse({"type": "token", "text": token})

            sources = [
                {
                    "doc": c.doc,
                    "doc_id": c.doc_id,
                    "page": c.page,
                    "score": round(s, 2),
                    "text": c.text[:400],
                }
                for c, s in rag.last_sources
            ]
            yield sse({"type": "sources", "sources": sources})

            # Append to session conversation history
            history.extend([
                {"role": "user", "content": question},
                {"role": "model", "content": "".join(parts)},
            ])
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/reset")
def reset():
    """Reset session chat conversation history."""
    STORE[sid()] = {"history": []}
    return jsonify(ok=True, message="Chat conversation history cleared.")


@app.post("/clear_all_documents")
def clear_all_documents():
    """Clear all documents and reset the persistent vector store."""
    for doc in list(rag.documents.values()):
        rag.remove_document(doc["id"])
    STORE[sid()] = {"history": []}
    return jsonify(ok=True, documents=[], total_chunks=0)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
