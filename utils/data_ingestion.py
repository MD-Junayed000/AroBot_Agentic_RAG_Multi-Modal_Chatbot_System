import os, fitz
from pathlib import Path
from core.vector_store import PineconeStore

def load_pdfs(folder):
    docs=[]
    for p in Path(folder).glob("*.pdf"):
        doc = fitz.open(str(p))
        for i in range(len(doc)):
            text = doc[i].get_text("text")
            if text and text.strip():
                docs.append({"text": text.strip(), "meta": {"id": f"{p.stem}_p{i}", "source": str(p), "page": i}})
    return docs

if __name__=="__main__":
    store = PineconeStore(dimension=384)
    folder = "knowledge/pdfs"
    docs = load_pdfs(folder)
    print("Loaded pages:", len(docs))
    store.upsert_texts([d["text"] for d in docs], [d["meta"] for d in docs])
    print("Indexed to Pinecone.")
