from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # Final_RAG/
STORAGE_DIR = BASE_DIR / "storage"

DOCS_DIR = STORAGE_DIR / "docs"
INDEX_DIR = STORAGE_DIR / "index"
RUNS_DIR = STORAGE_DIR / "runs"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Chunking defaults 
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Retrieval defaults
TOP_K = 4
