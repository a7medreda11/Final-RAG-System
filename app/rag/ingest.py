import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
# Importing the embedding model and builder from the local vectorstore module
from app.rag.vectorstore import EMB, build_vectorstore 
from .loaders import load_document_text

def ingest(file_path: str, index_dir: str):
    """
    Reads a document, splits it into chunks, and updates/creates the FAISS vector index.
    """
    print(f"\n--- ⏳ STARTING INGESTION FOR: {file_path} ---")
    
    # 1. Attempt to extract text from the document
    try:
        text, source_name = load_document_text(file_path)
        print(f"✅ TEXT EXTRACTED! Length: {len(text)} characters.")
        
        # Validation: If text is empty or too short, it might be a scanned image or empty file
        if len(text.strip()) < 50:
            print("⚠️ WARNING: The file seems empty or is a scanned image (No text found).")
            return {"status": "failed", "error": "File is empty or scanned image."}
            
    except Exception as e:
        print(f"❌ ERROR reading file: {e}")
        return {"status": "error", "message": str(e)}

    # 2. Text Splitting (Chunking)
    # chunk_size=1000 provides a good balance between context and retrieval speed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    print(f"✂️ Created {len(chunks)} chunks.")

    # Convert chunks into LangChain Document objects with metadata
    docs = [Document(page_content=c, metadata={"source": source_name, "chunk_id": i}) for i, c in enumerate(chunks)]

    # 3. Save/Update Vector Database
    # Check if an index already exists to merge data, otherwise create a new one
    
    
    if os.path.exists(index_dir) and (Path(index_dir) / "index.faiss").exists():
        try:
            print("🔄 Updating existing index...")
            vs = FAISS.load_local(index_dir, EMB, allow_dangerous_deserialization=True)
            vs.add_documents(docs)
            vs.save_local(index_dir)
        except Exception as e:
            print(f"⚠️ Update failed, rebuilding index: {e}")
            build_vectorstore(docs, index_dir)
    else:
        print("🆕 Creating NEW index...")
        build_vectorstore(docs, index_dir)

    print(f"💾 SAVED to {index_dir}")
    print("--- ✅ DONE ---\n")
    return {"chunks": len(docs), "status": "success"}