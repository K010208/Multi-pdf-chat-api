"""
Streamlit Multi-PDF ChatApp using LangChain + Google Gemini embeddings + FAISS
- Upload multiple PDFs
- Build / persist FAISS index of embeddings (per project)
- Chat with the documents using Gemini chat model via LangChain

Before running:
1) pip install -r requirements.txt
2) Set GOOGLE_API_KEY environment var (or use .env)
   - Create API key in Google AI Studio / Vertex and export as GOOGLE_API_KEY
"""

import os
import tempfile
import pickle
from typing import List
from pathlib import Path

import streamlit as st
from PyPDF2 import PdfReader
from tqdm import tqdm
from dotenv import load_dotenv

# LangChain / Google Gemini imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# Load .env if present
load_dotenv()

# ---- CONFIG ----
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"  # LangChain wrapper model id
GEMINI_CHAT_MODEL = "models/gemini-2.5-pro"  # change if you want a different Gemini model
INDEX_DIR = Path("./faiss_indexes")
INDEX_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Multi-PDF ChatApp (Gemini + FAISS)", layout="wide")

# ---- UI ----
st.title("📚 Multi-PDF ChatApp — Gemini + FAISS + Streamlit")
st.markdown(
    """
Upload one or more PDFs. The app will extract text, chunk it, create embeddings using Google Gemini (via LangChain),
store vectors in FAISS, and let you ask questions in a conversational way.
"""
)

# API key check
if "GOOGLE_API_KEY" not in os.environ:
    st.warning("Set your GOOGLE_API_KEY environment variable (or add to .env).")
    if st.button("I want help setting up the API key"):
        st.info("Create an API key in Google AI Studio / Vertex AI. Then `export GOOGLE_API_KEY='YOUR_KEY'` or place it in a .env file.")
    st.stop()

uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

index_name = st.text_input("Index name (used to persist FAISS index)", value="mypdf_index")

build_index_btn = st.button("Build / Rebuild index from uploaded PDFs")

# Utility: extract text from a single PDF file
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(pdf_bytes)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

# Utility: convert uploaded files to LangChain Documents
def pdfs_to_documents(files) -> List[Document]:
    docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(f.read())
            tf.flush()
            tf_path = tf.name
        text = extract_text_from_pdf_bytes(open(tf_path, "rb"))
        # metadata includes source filename for traceability
        metadata = {"source": getattr(f, "name", "uploaded_pdf")}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

# Build / rebuild index
if build_index_btn:
    if not uploaded_files:
        st.error("Please upload at least one PDF.")
    else:
        with st.spinner("Parsing PDFs and building vector index..."):
            docs = pdfs_to_documents(uploaded_files)

            # Split long docs into chunks to keep embeddings effective
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            split_docs = []
            for d in docs:
                chunks = splitter.split_text(d.page_content)
                for i, chunk in enumerate(chunks):
                    md = dict(d.metadata)
                    md["chunk"] = i
                    split_docs.append(Document(page_content=chunk, metadata=md))

            # Create embeddings
            st.info("Creating embeddings with Gemini (Google generative AI embeddings)...")
            embeddings = GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)

            # Use FAISS via LangChain VectorStore wrapper
            vectorstore_path = INDEX_DIR / f"{index_name}.faiss"
            # Create the FAISS index from documents
            faiss_index = FAISS.from_documents(split_docs, embeddings)

            # Save the index + docstore (langchain's FAISS has save_local)
            # Persist to directory
            faiss_index.save_local(str(vectorstore_path))
            # Save metadata / small pickle for retrieval later
            with open(INDEX_DIR / f"{index_name}_meta.pkl", "wb") as pf:
                pickle.dump({"n_docs": len(split_docs)}, pf)

            st.success(f"Index built and saved as '{vectorstore_path.name}' with {len(split_docs)} chunks.")

# Load existing index
load_index_btn = st.button("Load existing index")
vectorstore = None
if load_index_btn or (not build_index_btn and INDEX_DIR.joinpath(f"{index_name}.faiss").exists()):
    vectorstore_path = INDEX_DIR / f"{index_name}.faiss"
    if not vectorstore_path.exists():
        st.error(f"No index file found at {vectorstore_path}. Build one first.")
    else:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
            vectorstore = FAISS.load_local(str(vectorstore_path), embeddings)
            st.success(f"Loaded index '{index_name}' — it contains vectors.")
        except Exception as e:
            st.exception(e)

# Chat UI (only enabled when vectorstore loaded)
if vectorstore:
    st.subheader("Chat with your PDFs")
    # optional: context window size or number of retrieved docs
    top_k = st.slider("Number of retrieved chunks (k)", 1, 10, 4)

    # setup chat model (Gemini chat)
    chat_llm = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, temperature=0.2)

    # build conversational retrieval chain
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_llm,
        retriever=retriever,
        return_source_documents=True,
    )

    # session state: store chat history
    if "history" not in st.session_state:
        st.session_state["history"] = []

    user_question = st.text_input("Ask a question about the uploaded PDFs:")

    if st.button("Send") and user_question:
        with st.spinner("Retrieving answer..."):
            # run the chain
            result = qa_chain({"question": user_question, "chat_history": st.session_state["history"]})
            answer = result["answer"]
            src_docs = result.get("source_documents", [])

            # update history: append (user, assistant) pair
            st.session_state["history"].append((user_question, answer))

            st.markdown("**Answer:**")
            st.write(answer)

            if src_docs:
                st.markdown("**Source chunks (top results):**")
                for i, d in enumerate(src_docs[:top_k]):
                    src = d.metadata.get("source", "unknown")
                    chunk_id = d.metadata.get("chunk", None)
                    st.write(f"- **{src}** (chunk: {chunk_id}) — {d.page_content[:400]}...")

    if st.button("Clear chat history"):
        st.session_state["history"] = []
        st.success("Chat history cleared.")

st.markdown("---")
st.info("Tip: For production, persist FAISS index to cloud storage or use a managed vector DB. Also secure your API key and rate-limit requests.")
