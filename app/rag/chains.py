from typing import Dict, Any, List
import os
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from .memory import get_llm

# Function to format retrieved document chunks
def _format_context(docs: List[Document]) -> str:
    text = []
    for i, d in enumerate(docs, start=1):
        content = (d.page_content or "").strip()
        text.append(f"--- Source {i} Start ---\n{content}\n--- Source {i} End ---")
    return "\n\n".join(text)

# Function to format conversation history for the LLM
def _format_history(history: List) -> str:
    if not history:
        return ""
    formatted_history = ""
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted_history += f"{role}: {content}\n"
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
            formatted_history += f"User: {msg[0]}\nAssistant: {msg[1]}\n"
    return formatted_history

def build_rag_chain(index_dir: str, top_k: int = 5):
    
    # 1. Load the Vectorstore (ChromaDB)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory=index_dir, 
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        print(f"✅ Vectorstore loaded successfully from {index_dir}")
    except Exception as e:
        print(f"❌ Error loading vectorstore: {e}")
        return RunnableLambda(lambda x: {"answer": f"Database load error: {e}", "sources": []})

    # 2. Load the LLM
    llm = get_llm()

    def run(payload: Dict[str, Any]) -> Dict[str, Any]:
        full_input = payload.get("question") or payload.get("input") or ""
        
        if isinstance(full_input, dict):
             full_input = full_input.get("question", "")

        chat_history = payload.get("chat_history", [])
        current_question = str(full_input).strip()
        
        print(f"\n🔍 Processing Question: '{current_question}'")

        # --- Step 1: Query Rephrasing ---
        search_query = current_question 
        
        if chat_history:
            history_text = _format_history(chat_history)
            if history_text:
                rephrase_prompt = f"""
                Given a conversation history and a new follow-up question, 
                rephrase the follow-up question to be a standalone question.
                Maintain the language of the follow-up question (e.g., if it's in Arabic, stay in Arabic).
                
                History: {history_text}
                Question: {current_question}
                Standalone Question:"""
                try:
                    refined_response = llm.invoke(rephrase_prompt)
                    search_query = refined_response.content.strip()
                    print(f"🔄 Rephrased Query: '{search_query}'")
                except:
                    pass

        # --- Step 2: Retrieval ---
        docs = retriever.invoke(search_query)
        if not docs:
            return {"answer": "I couldn't find relevant info / لم أجد معلومات ذات صلة", "sources": []}

        # --- Step 3: Final Answer Generation (Flexible Language) ---
        context_str = _format_context(docs)
        
        final_prompt = f"""
        You are an expert assistant. Use the following context to answer the question.
        
        Context:
        {context_str}
        
        User Question: {current_question}
        
        Instructions:
        1. Answer strictly based on the context provided.
        2. VERY IMPORTANT: Answer in the SAME LANGUAGE as the user's question (Arabic if Arabic, English if English).
        3. Be direct and professional.
        
        Final Answer:
        """

        print("🤖 Generating Final Answer...")
        try:
            response = llm.invoke(final_prompt)
            answer = response.content
        except Exception as e:
            return {"answer": f"Error: {e}", "sources": []}

        sources = []
        for d in docs:
            sources.append({
                "source": d.metadata.get("source", "document"),
                "page": d.metadata.get("page", "N/A"),
                "preview": d.page_content[:100] + "..."
            })

        return {"answer": answer, "sources": sources}

    return RunnableLambda(run)