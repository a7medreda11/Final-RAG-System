import re
from typing import List, Tuple, Set
from langchain_core.documents import Document

# Words that do not carry significant meaning (Stop Words) to ignore during comparison
STOP_WORDS = {
    "the", "is", "at", "which", "on", "and", "a", "an", "of", "in", "to", "for", "with", "by", "that", "this", "it",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "but", "if", "or", "as",
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "أن", "ان", "التي", "الذي", "ما", "لا", "نعم", "هو", "هي"
}

def guardrail_check(question: str, docs: List[Document], min_docs: int = 1) -> Tuple[bool, str]:
    """
    Checks if any documents were actually retrieved. 
    If no docs are found, it blocks the process to prevent the LLM from guessing.
    """
    if not docs or len(docs) < min_docs:
        return False, "I'm sorry, I couldn't find any relevant information in the uploaded documents to answer that."
    return True, ""


def grounded_overlap_ok(answer: str, docs: List[Document], min_overlap: float = 0.15) -> bool:
    """
    Ensures that the important keywords in the AI's answer actually exist within the retrieved documents.
    This acts as a 'Truthfulness' check.
    """
    if not answer: return False

    # Combine all document text into one context string
    context = " ".join((d.page_content or "") for d in docs).lower()
    answer_lower = answer.lower()

    # Regular expression to extract words (Tokens) supporting both Latin and Arabic scripts
    token_re = r"[a-zA-Z\u0600-\u06FF]+"
    ctx_tokens = set(re.findall(token_re, context))
    ans_tokens = set(re.findall(token_re, answer_lower))

    # Filter only meaningful words (Ignore stop words and very short characters)
    meaningful_ans = {t for t in ans_tokens if t not in STOP_WORDS and len(t) > 2}

    # If the answer is extremely short (e.g., "Yes"), we pass it
    if not meaningful_ans:
        return True 

    # Calculate the overlap score between the answer and the context
    overlap_count = len(ctx_tokens & meaningful_ans)
    score = overlap_count / len(meaningful_ans)

    # Return True if the overlap meets the minimum threshold
    return score >= min_overlap


def is_gibberish(text: str) -> bool:
    """
    Detects if the input or output text is nonsensical or contains spammy characters.
    """
    if not text: return True
    t = text.strip()
    
    # Allow short valid responses like "Yes" or "Okay"
    if len(t) < 5: return False 
    
    # Check for character-to-symbol ratio (checks for excessive special characters)
    letters = re.findall(r"[a-zA-Z\u0600-\u06FF]", t)
    if len(t) > 0 and (len(letters) / len(t)) < 0.35: 
        return True
    
    # Detect repetition/spam (e.g., too many periods or symbols)
    if t.count(".") > 20: 
        return True
    
    return False