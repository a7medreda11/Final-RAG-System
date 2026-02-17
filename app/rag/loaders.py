from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
from docx import Document as DocxDocument

def load_pdf_pymupdf(path: Path) -> str:
    doc = fitz.open(str(path))
    return "\n".join([page.get_text("text") for page in doc]).strip()

def load_pdf_pdfplumber(path: Path) -> str:
    texts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts).strip()

def load_docx(path: Path) -> str:
    d = DocxDocument(str(path))
    return "\n".join(p.text for p in d.paragraphs).strip()

def load_document_text(file_path: str) -> tuple[str, str]:
    p = Path(file_path)
    suf = p.suffix.lower()

    if suf == ".pdf":
        text = load_pdf_pymupdf(p)
        if not text:
            text = load_pdf_pdfplumber(p)
        return text, p.name

    if suf == ".docx":
        return load_docx(p), p.name

    # fallback txt
    return p.read_text(encoding="utf-8", errors="ignore"), p.name
