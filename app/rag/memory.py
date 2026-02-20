import os
import google.generativeai as genai
from typing import Dict, List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI


_STORE: Dict[str, List[Tuple[str, str]]] = {}

def add_turn(session_id: str, role: str, content: str, max_turns: int = 12) -> None:
    """Adds a new interaction (User/Assistant) to the session history."""
    session_id = session_id or "default"
    role = role or "user"
    content = content or ""
    hist = _STORE.setdefault(session_id, [])
    hist.append((role, content))
    
    # Maintain a sliding window of the last 'max_turns'
    if len(hist) > max_turns:
        _STORE[session_id] = hist[-max_turns:]

def build_history_text(session_id: str, max_turns: int = 12) -> str:
    """Formats the session history into a single string for prompt injection."""
    session_id = session_id or "default"
    hist = _STORE.get(session_id, [])[-max_turns:]
    lines = []
    for role, content in hist:
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()

def clear_all() -> None:
    """Clears memory for all active sessions."""
    _STORE.clear()

def clear_session(session_id: str) -> None:
    """Clears history for a specific session."""
    session_id = session_id or "default"
    _STORE.pop(session_id, None)

# =========================
# Smart Google Gemini Auto-Selector
# =========================

_LLM = None

def get_llm():
    """
    Initializes and returns the best available Google Gemini model.
    It automatically detects if 'flash' or 'pro' versions are available in the API key quota.
    """
    global _LLM
    if _LLM is not None:
        return _LLM

    # 1. API Key Configuration
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Fallback key (ensure this is kept secure in production)
        api_key = "AIzaSyAuoIMxpGzfPxmzjQ0SnxfpZWuUpd-imcs"
    
    os.environ["GOOGLE_API_KEY"] = api_key # Required for LangChain compatibility

    # 2. Intelligent Model Discovery
    print("\n🔍 Auto-detecting best available Gemini model...")
    selected_model = "gemini-1.5-pro" # Default fallback
    
    try:
        genai.configure(api_key=api_key)
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                # Prioritize 'Flash' models for speed in RAG applications
                if "flash" in name:
                    selected_model = name
                    break
                elif "pro" in name:
                    selected_model = name
    except Exception as e:
        print(f"⚠️ Could not list models ({e}), defaulting to 'gemini-1.5-pro'")

    print(f"✅ Selected Model: {selected_model}")

    # 3. LLM Initialization
    _LLM = ChatGoogleGenerativeAI(
        model=selected_model,
        google_api_key=api_key,
        temperature=0.3, # Low temperature for factual RAG responses
        convert_system_message_to_human=True 
    )

    return _LLM