🤖 Smart RAG Chatbot: Context-Aware AI Assistant
This project is a sophisticated Retrieval-Augmented Generation (RAG) system. It enables users to upload private documents (PDFs, Text) and interact with them through a conversational AI interface. By anchoring the Google Gemini LLM to a local vector database, the system provides factual, source-based answers, effectively eliminating AI hallucinations.

🌟 Key Features
Dynamic Document Context: Answers questions based strictly on uploaded data.

Hybrid Model Strategy: Utilizes Gemini 1.5/2.5 Flash for fast user interactions and Gemini 1.5 Pro for rigorous evaluation.

Persistent Vector Memory: Uses ChromaDB to store document embeddings for lightning-fast retrieval.

Semantic Search: Powered by HuggingFace Transformers to understand the meaning behind questions, not just keywords.

Conversation Memory: Maintains a sliding window of chat history to handle follow-up questions naturally.

Automated Evaluation: Includes a dedicated evaluation pipeline (LLM-as-a-Judge) to score answer accuracy.

🏗️ System Architecture & Workflow
Ingestion Phase:

Documents are loaded and split into smaller, overlapping chunks (Chunking).

Each chunk is converted into a numerical vector (Embedding) using HuggingFace models.

Vectors are stored in ChromaDB.

Retrieval Phase:

When a user asks a question, the system converts the query into a vector.

It performs a Similarity Search in ChromaDB to find the most relevant document segments.

Generation Phase:

The retrieved segments (Context) and the user's question are injected into a specialized Prompt Template.

The LLM generates a concise answer based only on the provided context.

Evaluation Phase:

The system compares generated answers against a Ground Truth dataset.

A "Judge" LLM (Gemini Pro) assigns a score (0-10) and provides reasoning for the grade.