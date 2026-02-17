import sys
import os
import shutil
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import Core Libraries
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from app.rag.chains import build_rag_chain

# Directory Setup
DB_PATH = os.path.join(parent_dir, "chroma_db")
UPLOAD_FOLDER = os.path.join(parent_dir, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDfkB1u_Y5hX_8fMRQMEMiET6K-c39Djus"

current_chain = None
processed_docs = [] 

def process_file(file_obj):
    global current_chain, processed_docs
    if not file_obj:
        return "⚠️ Please select a PDF or DOCX file first."

    try:
        # Clear existing database for a fresh start
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)

        file_path = os.path.join(UPLOAD_FOLDER, os.path.basename(file_obj.name))
        shutil.copy(file_obj.name, file_path)

        # Support PDF and Word
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith('.docx') or file_path.lower().endswith('.doc'):
            loader = Docx2txtLoader(file_path)
        else:
            return "❌ Unsupported format! Please upload PDF or DOCX only."

        processed_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", ".", " ", ""]
        )
        splits = text_splitter.split_documents(processed_docs)

        # Local Embeddings for Vector Store
        embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_func,
            persist_directory=DB_PATH
        )
        
        current_chain = build_rag_chain(index_dir=DB_PATH)
        return f"✅ Success! Processed '{os.path.basename(file_obj.name)}'."
    
    except Exception as e:
        return f"❌ Error during processing: {str(e)}"

def summarize_file():
    global processed_docs
    if not processed_docs:
        return "⚠️ You must upload and process a file before summarization."
    
    try:
        # بدل ما نكتب الإسم يدوياً، هنستخدم الدالة اللي عملناها بتنقي الموديل الصح
        from app.rag.memory import get_llm
        llm = get_llm() 
        
        # دمج محتوى المستند (أول 15 قطعة)
        context_text = "\n".join([d.page_content for d in processed_docs[:15]])
        
        prompt = f"""Provide a comprehensive and professional summary of the following text. 
        Focus on the main points and key objectives. 
        Respond in the same language as the text provided:
        
        {context_text}
        
        Summary:"""
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"❌ Error during summarization: {str(e)}"

def chat_fn(message, history):
    global current_chain
    if current_chain is None:
        return "⚠️ Please upload and process a file first."
    
    try:
        result = current_chain.invoke({"question": message})
        return result.get("answer", "No answer found in the document.")
    except Exception as e:
        return f"❌ Response Error: {str(e)}"

# Professional UI Layout
with gr.Blocks(title="AI Tutor - Smart Contract Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Smart Document Assistant")
    gr.Markdown("Upload your documents (PDF/Word) to start chatting or generate instant summaries.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Document Management")
            file_input = gr.File(label="Upload File Here", file_types=[".pdf", ".docx"])
            upload_btn = gr.Button("Start Processing 🚀", variant="primary")
            upload_status = gr.Textbox(label="System Status", interactive=False)
            upload_btn.click(process_file, inputs=[file_input], outputs=[upload_status])

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("💬 Smart Chat"):
                    chatbot = gr.ChatInterface(fn=chat_fn)
                
                with gr.TabItem("📝 Document Summary"):
                    gr.Markdown("### Generate Comprehensive Summary")
                    sum_btn = gr.Button("Generate Summary ✨")
                    sum_output = gr.Textbox(label="Summary Result", lines=15, interactive=False)
                    sum_btn.click(summarize_file, outputs=[sum_output])

if __name__ == "__main__":
    demo.launch()