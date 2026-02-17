from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil

from app.settings import DOCS_DIR, INDEX_DIR
from app.rag.ingest import ingest
from app.rag.chains import build_rag_chain
from app.rag.memory import add_turn, build_history_text, clear_all

app = FastAPI(title="Final_RAG System (Local Backend)")

# Global variable to hold the RAG chain instance
rag_chain = None

@app.on_event("startup")
def startup_load():
    """
    Initializes the RAG chain on server startup if an existing index is found.
    """
    global rag_chain
    # Checking for vector store files (Chroma/FAISS indices)
    if (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists():
        rag_chain = build_rag_chain(str(INDEX_DIR))

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """
    Handles PDF/Docx uploads, triggers the ingestion pipeline, and updates the RAG chain.
    """
    dst = DOCS_DIR / file.filename
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process the document: Chunking, Embedding, and Storing
    res = ingest(str(dst), str(INDEX_DIR))

    global rag_chain
    # Rebuild the chain with the new index
    rag_chain = build_rag_chain(str(INDEX_DIR), top_k=10)

    # ✅ Best Practice: A new upload starts a new context. 
    # Clear all previous chat sessions to avoid context mixing.
    clear_all()

    return {"status": "success", "message": "File processed successfully", **res}

@app.post("/chat")
def chat(payload: dict):
    """
    Processes chat queries using retrieval-augmented generation and maintains session memory.
    """
    global rag_chain
    if rag_chain is None:
        return {"error": "System not ready. Please upload a document first to create an index."}

    session_id = (payload.get("session_id") or "default").strip()
    q = (payload.get("question") or "").strip()
    
    if not q:
        return {"error": "Question field is required."}

    # ✅ Contextualization: Inject chat history into the question.
    # This helps the LLM understand follow-up questions like "Explain more" or "What about it?".
    history = build_history_text(session_id)
    augmented_question = q if not history else f"Context History:\n{history}\n\nCurrent User Question: {q}"

    # Invoke the RAG chain
    result = rag_chain.invoke({"question": augmented_question})

    # Error handling if the chain returns an error object
    if isinstance(result, dict) and result.get("error"):
        return result

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # Update session memory with the new turn
    add_turn(session_id, q, answer)

    return {
        "answer": answer, 
        "sources": sources, 
        "session_id": session_id
    }