import io
import os
import shutil
import time
import app as flask_app_module
from rag import PDFRAG
from test_rag import DummyEmbedder

def test_flask_routes():
    # Use isolated test storage directory
    test_storage = os.path.join(os.path.dirname(__file__), "test_app_storage")
    if os.path.exists(test_storage):
        shutil.rmtree(test_storage)
        
    orig_rag = flask_app_module.rag
    flask_app_module.rag = PDFRAG(DummyEmbedder(), storage_dir=test_storage)
    app = flask_app_module.app
    app.config["TESTING"] = True
    client = app.test_client()

    try:
        # 1. GET /
        res = client.get("/")
        assert res.status_code == 200
        assert b"DocuMind AI" in res.data
        print("GET /: OK (DocuMind AI title found)")

        # 2. GET /documents
        res = client.get("/documents")
        assert res.status_code == 200
        data = res.get_json()
        assert "documents" in data
        assert "total_chunks" in data
        print("GET /documents: OK (returned valid document list)")

        # 3. POST /upload with multiple unique files
        ts = str(time.time())
        f1 = (io.BytesIO(f"Unique test content A {ts} for RAG indexing.".encode()), f"test_a_{ts}.txt")
        f2 = (io.BytesIO(f"Unique test content B {ts} for cloud database.".encode()), f"test_b_{ts}.txt")
        res = client.post(
            "/upload",
            data={"pdf": [f1, f2]},
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["indexed_count"] == 2
        uploaded_ids = [r["doc_id"] for r in data["results"] if r.get("status") == "indexed"]
        print("POST /upload (multiple files): OK (indexed 2 docs successfully)")

        # 4. Duplicate upload check
        f1_dup = (io.BytesIO(f"Unique test content A {ts} for RAG indexing.".encode()), "test_a_dup.txt")
        res = client.post(
            "/upload",
            data={"pdf": [f1_dup]},
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["duplicate_count"] == 1
        print("POST /upload duplicate check: OK (detected duplicate file by hash)")

        # 5. DELETE /documents/<doc_id>
        if uploaded_ids:
            doc_to_delete = uploaded_ids[0]
            res = client.delete(f"/documents/{doc_to_delete}")
            assert res.status_code == 200
            data = res.get_json()
            assert data["ok"] is True
            assert all(d["id"] != doc_to_delete for d in data["documents"])
            print(f"DELETE /documents/{doc_to_delete}: OK (document removed & index rebuilt)")

        # 6. POST /reset
        res = client.post("/reset")
        assert res.status_code == 200
        print("POST /reset: OK")

        print("\nALL FLASK ROUTE TESTS PASSED SUCCESSFULLY!")
    finally:
        flask_app_module.rag = orig_rag
        if os.path.exists(test_storage):
            shutil.rmtree(test_storage)

if __name__ == "__main__":
    test_flask_routes()
