import streamlit as st
import pyttsx3,threading,requests
from dotenv import load_dotenv 
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from docx import Document
from firebase_config import auth, db
from groq import Groq
import pytesseract
from pdf2image import convert_from_path,convert_from_bytes
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
import os, cv2

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
WEB_SEARCH_URL = os.getenv("WEB_SEARCH_URL", "http://localhost:5000/web_search")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("Huggingface_API")

# LLMS
groq_client = Groq(api_key=GROQ_API_KEY)
def call_groq(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    try:
        chat_completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"--- Groq Error --- {e}"
    
# web search results to ollama
def call_ollama(prompt: str, model: str = "llama3") -> str:
    try:
        llm = OllamaLLM(model=model)  
        return llm.invoke(prompt)
    except Exception as e:
        return f"--- Ollama Error --- {e}"
    
def calculate_confidence(vectorstore, question: str, answer: str) -> float:
    try:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={"device": "cpu"})
        q_emb = embeddings.embed_query(question)
        a_emb = embeddings.embed_query(answer)
        sim = cosine_similarity([q_emb], [a_emb])[0][0]
        return round(float(sim), 2)
    except Exception:
        return 0.0

# FILE PROCESSING
# to extract text from files
def extract_text_from_file(file):
    if file.name.endswith(".pdf"):
        try:
            reader = PdfReader(file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

            # Fallback to OCR if PDF has no extractable text
            if not text.strip():
                file.seek(0)
                images = convert_from_bytes(file.read())
                text = "\n".join([pytesseract.image_to_string(img) for img in images])
            return text

        except Exception:
            # Hard fallback to OCR only
            file.seek(0)
            images = convert_from_bytes(file.read())
            return "\n".join([pytesseract.image_to_string(img) for img in images])

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    
    elif file.name.lower().endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(file)
        processed = preprocess_image(image)
        return pytesseract.image_to_string(processed)

    else:
        return ""

# OCR image processing
def preprocess_image(pil_image):
    img = np.array(pil_image)  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    denoised = cv2.medianBlur(thresh, 3)
    return Image.fromarray(denoised) 
    
# Combine text from all uploaded documents
def get_combined_text(uploaded_files):
    all_text = ""
    for file in uploaded_files:
        all_text += extract_text_from_file(file) + "\n"
    return all_text

# chunking of text
def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks
    
# vector storage
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"} 
    )
    vectorstore = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    ) 
    return vectorstore

# AGENTS 
def document_agent(vectorstore, question: str) -> tuple:
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(question)
        context = "\n".join([d.page_content for d in docs]) if docs else "No relevant context found."
        prompt = (
            f"You are a helpful assistant. Use only the provided context if relevant.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Provide a clear, factual, and concise answer."
        )
        answer = call_groq(prompt)
        confidence = calculate_confidence(vectorstore, question, answer)
        return answer, confidence
    except Exception as e:
        return f"--- Document Agent Error --- {e}", 0.0

def web_agent(question: str) -> tuple:   
    search_results = perform_web_search(question)
    if not search_results or search_results.startswith("--- Web Search Error ---"):
        return "No relevant web search results found.", 0.0
    prompt=f"""
    You are a helpful AI assistant. Use ONLY the following web search snippets to answer the user’s question.
    Web Search Results:
    {search_results}
    Question: {question}
    Instructions:
    - Summarize the key answer based on the snippets above.
    - Be factual and concise.
    - If sources/links are provided, include them at the end.
    - If the snippets do not contain relevant information, say "I couldn't find reliable web results for this."
    """

    answer = call_ollama(prompt)
    return answer,0.5
    
# retrieving chunks from vectorstore    
def retrieve_and_ask_groq(vectorstore, question: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)

    context_items = []
    for i, d in enumerate(docs, 1):
        content = d.page_content if hasattr(d, 'page_content') else str(d)
        context_items.append(f"{content}")

    context = "\n".join(context_items) if context_items else "No relevant context found."
    prompt = f"You are a helpful assistant. Use the provided context if relevant.\n\nContext:\n{context}\n\nQuestion: {question}\n\nProvide a clear, factual, and concise answer."
    
    return call_groq(prompt)

# to read the answer
def speak(text: str):
    def run_speech():
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.say(text)
        engine.runAndWait()
        
    threading.Thread(target=run_speech).start()
    
# Login page        
def login_ui():
    if "user" not in st.session_state:
        st.session_state.user = None

    st.sidebar.title("Login / Sign Up")
    action = st.sidebar.selectbox("Choose", ["Login", "Sign Up"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if action == "Sign Up":
        if st.sidebar.button("Create Account"):
            try:
                auth.create_user_with_email_and_password(email, password)
                st.success("Account created. Please log in.")
            except Exception as e:
                st.error(f"Signup error: {e}")

    elif action == "Login":
        if st.sidebar.button("Login"):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state.user = user
                st.success(f"Logged in as {email}")
            except Exception as e:
                st.warning("No user found. Please Sign up to continue")

    return st.session_state.user is not None

def store_chat_history(user_id, question, answer):
    user_ref = db.collection("chat_history").document(user_id)
    doc = user_ref.get()
    if doc.exists:
        history = doc.to_dict().get("history", [])
    else:
        history = []
    history.append({"question": question, "answer": answer})
    history = history[-5:]
    user_ref.set({"history": history})
    

def get_recent_chats(user_id):
    user_ref = db.collection("chat_history").document(user_id)
    doc = user_ref.get()
    if doc.exists:
        return doc.to_dict().get("history", [])
    return []


# Web search    
def perform_web_search(query: str) -> str:
    try:
        resp = requests.post(WEB_SEARCH_URL, json={"query": query}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("search_results", "No web results found.")
    except requests.exceptions.RequestException as e:
        return f"--- Web Search Error ---\nCould not perform web search: {e}"

# Query Input
def handle_userinput(question):
    vectorstore = st.session_state.get('vectorstore')
    if vectorstore is None:
        st.warning("Please process documents first.")
        return
    
    try:
        doc_answer, confidence = document_agent(vectorstore, question)   
    except Exception as e:
        st.error(f"Groq error: {e}")
        doc_answer, confidence = call_ollama(question)

    final_answer = doc_answer

    low_conf = confidence < 0.65  
    short_answer = len(doc_answer.strip()) < 120
    keywords = any(
        x in doc_answer.lower() 
        for x in ["i don't know", "not found", "sorry", "unable to", "unfortunately", "couldn't", "no relevant context", "not available"]
        )

    if low_conf or short_answer or keywords:
        st.info("Low confidence in document answer. Searching the web...")
        web_answer, web_conf = web_agent(question)
        if web_conf > confidence:
            final_answer, confidence = web_answer, web_conf

    user_id = st.session_state.user['localId']
    store_chat_history(user_id, question, final_answer)

    st.write(f"🤖 {final_answer}")
    st.caption(f"Confidence Score: {confidence}")
    
    
def main():
    st.set_page_config(page_title="ChatBot")
    
    authenticated = login_ui()
    if not authenticated:
        st.warning("Please log in to continue.")
        return
    
    user_id = st.session_state.user['localId']
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    
    st.header("Hello!!")
    question=st.text_input("Ask a question")
     
    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files=st.file_uploader("Upload your files here 📂", type=["pdf", "txt", "docx" ,".png", ".jpg", ".jpeg"], accept_multiple_files=True)
        if st.button("Process"):
            if not uploaded_files:
                st.warning("Please upload at least one file before processing.")
            else:
                with st.spinner("Processing"):
                    # to get pdf text
                    raw_text=get_combined_text(uploaded_files)
                    
                    # to make chunks of the pdf
                    text_chunks=get_text_chunks(raw_text)
                    
                    # vectorstore creation
                    vectorstore=get_vectorstore(text_chunks)
                    st.session_state.vectorstore = vectorstore 
                    
                st.success("Processing complete! Ask your question.")
                
        # display recent chats
        st.subheader("Recent Chat History")
        if st.session_state.user:
            user_id = st.session_state.user['localId']
            history = get_recent_chats(user_id)
            for i, chat in enumerate(reversed(history), 1):
                st.markdown(f"**{i}. You:** {chat['question']}")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🤖 {chat['answer']}")
                
    if question and st.session_state.vectorstore is not None:
        with st.spinner("Thinking..."):
            handle_userinput(question)


if __name__=="__main__":
    main()