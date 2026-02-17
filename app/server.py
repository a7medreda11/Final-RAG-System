import os
import sys
from typing import List
from pydantic import BaseModel, Field  # Library to define data schemas for the UI

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from fastapi import FastAPI
from langserve import add_routes
from app.rag.chains import build_rag_chain

# --- (Input Schema) Define the input structure for the Playground ---
class InputChat(BaseModel):
    question: str = Field(..., description="Enter your question regarding the document here")

# -------------------------------------------------------

app = FastAPI(
    title="RAG Smart Assistant API",
    version="1.0",
    description="API server for the Retrieval-Augmented Generation (RAG) system",
)

db_path = os.path.join(root_dir, "chroma_db")

# Database connection check
if not os.path.exists(db_path):
    print(f"⚠️ Warning: Database not found at {db_path}")
    rag_chain = None
else:
    print(f"✅ Successfully linked to Database: {db_path}")
    # Initialize the chain
    rag_chain = build_rag_chain(index_dir=db_path)

if rag_chain:
    # --- Registering Routes with Type Support ---
    # with_types maps our Pydantic model to the Chain for the LangServe Playground
    add_routes(
        app,
        rag_chain.with_types(input_type=InputChat), 
        path="/rag",
    )

if __name__ == "__main__":
    import uvicorn
    # Start the server using Uvicorn
    print("🚀 Starting LangServe server on http://localhost:8000")
    uvicorn.run(app, host="localhost", port=8000)