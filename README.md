# 🤖 Smart-RAG: Context-Aware Document Intelligence System

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://www.langchain.com/)
[![AI-Model](https://img.shields.io/badge/LLM-Gemini--1.5--Flash-orange.svg)](https://deepmind.google/technologies/gemini/)
[![Database](https://img.shields.io/badge/VectorDB-ChromaDB-red.svg)](https://www.trychroma.com/)

**Smart-RAG** is a high-performance Retrieval-Augmented Generation system designed to transform static documents into interactive knowledge bases. By leveraging **Google Gemini's** advanced reasoning and **ChromaDB's** vector search, this system provides factual, context-grounded answers while strictly mitigating AI hallucinations.

---

## 🎯 Key Features

* **Semantic Intelligence:** Uses **HuggingFace Transformers** to understand the deep meaning of queries, not just keywords.
* **Fact-Grounded Responses:** Employs RAG architecture to ensure every answer is backed by your uploaded PDF/Text data.
* **Dual-Speed Logic:** Built with **Gemini 1.5 Flash** for ultra-fast chat responses and **Gemini 1.5 Pro** for high-precision evaluation.
* **Automated Evaluation (LLM-as-a-Judge):** Includes a built-in testing pipeline that scores answer accuracy and provides logical justifications.
* **Conversational Memory:** Maintains a sliding-window context to handle complex, multi-turn dialogues seamlessly.

---

## 🛠️ Technical Architecture

| Layer | Technology |
| :--- | :--- |
| **Orchestration** | **LangChain** (Chains & Memory Management) |
| **Large Language Model** | **Google Gemini API** (Flash & Pro versions) |
| **Embedding Engine** | **HuggingFace** (`all-MiniLM-L6-v2`) |
| **Vector Storage** | **ChromaDB** (Local Persistent Storage) |
| **Backend / UI** | **FastAPI** + **Gradio** Interactive Interface |

---

## 🚀 System Workflow

1.  **Ingestion:** Documents are parsed and split into semantic chunks with optimized overlap.
2.  **Embedding:** Text chunks are transformed into high-dimensional vectors.
3.  **Retrieval:** At query time, the system retrieves the Top-K most relevant context pieces from the vector store.
4.  **Generation:** The LLM synthesizes a final response based *exclusively* on the retrieved context.
5.  **Evaluation:** The system runs a comparison between the AI's output and human-verified "Ground Truth" to ensure quality.

---

## 📁 Project Structure

```text
Final_RAG/
├── app/
│   ├── rag/           # Core RAG Logic (Chains, Memory, Ingestion)
│   ├── eval/          # Automated Evaluation Pipeline & Dataset
│   └── main.py        # FastAPI Server & Gradio UI
├── chroma_db/         # Local Vector Store (Excluded from Git tracking)
└── requirements.txt   # Production Dependencies