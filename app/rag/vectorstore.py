from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Using a high-performance multilingual model 
# It effectively understands both Arabic and English semantic meanings.
EMB = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def _paths(index_dir: str | Path):
    """
    Helper function to manage and create the directory structure 
    for the FAISS index files.
    """
    d = Path(index_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d, (d / "index.faiss"), (d / "index.pkl")

def build_vectorstore(docs: List[Document], index_dir: str | Path) -> FAISS:
    """
    Creates a new FAISS vector store from documents and saves it locally.
    """
    d, _, _ = _paths(index_dir)
    vs = FAISS.from_documents(docs, EMB)
    vs.save_local(str(d))
    print(f"✅ Vector store built and saved to: {d}")
    return vs

def load_vectorstore(index_dir: str | Path) -> FAISS:
    """
    Loads an existing FAISS vector store from the local directory.
    """
    d, faiss_path, pkl_path = _paths(index_dir)
    if not faiss_path.exists() or not pkl_path.exists():
        raise FileNotFoundError(f"Index not found in {d}. Please upload and process a file first.")
    
    # Load the index using the same embedding model defined above
    vs = FAISS.load_local(str(d), EMB, allow_dangerous_deserialization=True)
    return vs