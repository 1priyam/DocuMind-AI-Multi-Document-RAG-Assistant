import os
import shutil
from rag import Chunk, PDFRAG

class DummyEmbedder:
    def encode(self, texts, **kwargs):
        import numpy as np
        # Return simple deterministic dummy embeddings of dim 8
        np.random.seed(42)
        arr = np.random.randn(len(texts), 8).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms

def test_rag_functionality():
    test_dir = os.path.join(os.path.dirname(__file__), "test_storage")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    embedder = DummyEmbedder()
    rag = PDFRAG(embedder, storage_dir=test_dir)
    
    # Test 1: Add first document (TXT)
    content1 = b"DocuMind AI is a multi-document RAG assistant. It supports several documents and citations."
    res1 = rag.add_document(content1, "guide.txt")
    print("Test 1 (Add guide.txt):", res1["status"], "- Chunks:", res1["chunk_count"])
    assert res1["status"] == "indexed"
    assert len(rag.documents) == 1
    doc1_id = res1["doc_id"]
    
    # Test 2: Duplicate check
    res2 = rag.add_document(content1, "guide_copy.txt")
    print("Test 2 (Duplicate check):", res2["status"], "- Message:", res2.get("message"))
    assert res2["status"] == "duplicate"
    assert len(rag.documents) == 1
    
    # Test 3: Add second document
    content3 = b"Financial report quarterly revenue grew by 25 percent in Q3 2026. Profits increased."
    res3 = rag.add_document(content3, "finance.txt")
    print("Test 3 (Add finance.txt):", res3["status"], "- Chunks:", res3["chunk_count"])
    assert res3["status"] == "indexed"
    assert len(rag.documents) == 2
    doc2_id = res3["doc_id"]
    
    # Test 4: Retrieval across all documents
    all_results = rag.retrieve("revenue", top_k=5)
    print("Test 4 (Retrieve all):", [(c.doc, c.page, round(s, 2)) for c, s in all_results])
    assert len(all_results) > 0
    
    # Test 5: Document filtering
    filtered_results = rag.retrieve("revenue", top_k=5, doc_ids=[doc1_id])
    print("Test 5 (Filter doc1 only):", [(c.doc, c.page) for c, _ in filtered_results])
    for c, _ in filtered_results:
        assert c.doc_id == doc1_id
        
    # Test 6: Persistence test (reload into fresh PDFRAG instance)
    rag2 = PDFRAG(embedder, storage_dir=test_dir)
    print("Test 6 (Persistence check): Loaded docs:", len(rag2.documents), "chunks:", len(rag2.chunks))
    assert len(rag2.documents) == 2
    assert len(rag2.chunks) == len(rag.chunks)
    
    # Test 7: Remove document
    rem_ok = rag2.remove_document(doc1_id)
    print("Test 7 (Remove doc1):", rem_ok, "Remaining docs:", len(rag2.documents))
    assert rem_ok is True
    assert len(rag2.documents) == 1
    assert doc1_id not in rag2.documents
    assert all(c.doc_id != doc1_id for c in rag2.chunks)
    
    # Cleanup test dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    print("\nALL RAG TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_rag_functionality()
