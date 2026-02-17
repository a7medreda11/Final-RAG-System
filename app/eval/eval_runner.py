import sys
import os
import json
import time

# --- Absolute Paths Setup ---
# Get the full path of the project root directory (Final_RAG)
current_dir = os.path.dirname(os.path.abspath(__file__)) # Currently in 'eval' folder
root_dir = os.path.dirname(os.path.dirname(current_dir)) # Go up to 'Final_RAG'
sys.path.append(root_dir)

# Import necessary components
try:
    from app.rag.chains import build_rag_chain
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def run_evaluation():
    # Define paths accurately
    db_path = os.path.join(root_dir, "chroma_db")
    dataset_path = os.path.join(current_dir, "dataset.jsonl")
    
    # Confirm database existence before starting
    if not os.path.exists(db_path):
        print(f"❌ Error: Database not found at: {db_path}")
        print("💡 Tip: Please upload and process a document via the Gradio interface first.")
        return

    print(f"✅ Database found at: {db_path}")
    
    # 1. Load the RAG chain (uses Gemini by default)
    print("🤖 Loading RAG chain...")
    rag_chain = build_rag_chain(index_dir=db_path)
    
    # 2. Setup the Judge LLM
    judge_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    print("🚀 Starting updated evaluation process...")

    if not os.path.exists(dataset_path):
        print(f"❌ Error: dataset.jsonl file not found at: {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                question = data['question']
                ground_truth = data['ground_truth']

                print(f"\n--- Test Case {i} ---")
                print(f"❓ Question: {question}")
                
                # Get response from the RAG system
                result = rag_chain.invoke({"question": question})
                generated_answer = result.get("answer", "")

                # Request evaluation from the Judge LLM
                prompt = f"""
                As an expert evaluator, compare the System's Answer against the Ground Truth (Ideal Answer).
                
                Question: {question}
                Ground Truth: {ground_truth}
                System Answer: {generated_answer}
                
                Task:
                1- Assign a score out of 10.
                2- Provide a concise justification for the score in English.
                
                Output Format: Score: X/10 | Reason: ...
                """
                
                eval_output = judge_llm.invoke(prompt).content
                print(f"⭐ Evaluation Result: {eval_output}")
                
                # Sleep to avoid Rate Limits (Quota)
                time.sleep(5) 

            except Exception as e:
                print(f"⚠️ Error processing question {i}: {e}")

if __name__ == "__main__":
    run_evaluation()