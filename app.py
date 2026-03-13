"""
Vessel RAG Chatbot — Streamlit UI
Lightning.ai + Ollama + LangChain + ChromaDB
Run: streamlit run app.py
"""

import os
import time
import streamlit as st
from pathlib import Path

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle RAG Chatbot",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CONFIG ────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "llama3.2"
EMBED_MODEL     = "nomic-embed-text"
DOCS_DIR        = "./documents"
CHROMA_DIR      = "./chroma_db"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 200
TOP_K           = 5


# ── LAZY IMPORTS (cached) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_chain():
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.chat_models import ChatOllama
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.prompts import PromptTemplate

    # ── Check docs exist ─────────────────────────────────────
    docs_path = Path(DOCS_DIR)
    if not docs_path.exists() or not list(docs_path.glob("**/*.pdf")):
        return None, "No PDF files found in ./documents/ — please upload your vessel PDF."

    # ── Load PDFs ─────────────────────────────────────────────
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=False
    )
    documents = loader.load()

    # ── Chunk ─────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # ── Embeddings + Vector Store ─────────────────────────────
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    db_path = Path(CHROMA_DIR)
    if db_path.exists() and any(db_path.iterdir()):
        vectorstore = Chroma(
            collection_name="vessel_docs",
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="vessel_docs",
            persist_directory=CHROMA_DIR
        )
        vectorstore.persist()

    # ── LLM ──────────────────────────────────────────────────
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_ctx=4096,
    )

    # ── Prompt ───────────────────────────────────────────────
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""You are a vehicle manual documentation assistant specializing in the 2023 BMW X1.
Answer ONLY from the context below. Do not use outside knowledge.
- Cite page numbers or sections when available
- For safety-critical info, always recommend verifying with the original manual
- If the answer is not in the context, say: "This information is not in the provided documents."
- Quote exact specification values when relevant
      

=== CONTEXT ===
{context}

=== CHAT HISTORY ===
{chat_history}

=== QUESTION ===
{question}

=== ANSWER ==="""
    )

    # ── Retriever + Memory ───────────────────────────────────
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20}
    )

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # ── Chain ────────────────────────────────────────────────
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False
    )

    return chain, None


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:

    st.title("🗄️ Vehicle RAG Chatbot")
    st.caption("Every file is its own knowledge base — mix and match any combination")

    st.markdown("### 📁 Upload PDF")
    uploaded_file = st.file_uploader(
        "Upload vessel PDF",
        type=["pdf"],
        help="Upload your vessel documentation PDF"
    )

    if uploaded_file:
        os.makedirs(DOCS_DIR, exist_ok=True)
        save_path = Path(DOCS_DIR) / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Saved: {uploaded_file.name}")
        if st.button("🔄 Rebuild Index"):
            import shutil
            if Path(CHROMA_DIR).exists():
                shutil.rmtree(CHROMA_DIR)
            st.cache_resource.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_sources = st.toggle("Show sources", value=True)
    show_chunks  = st.toggle("Show retrieved chunks", value=False)

    st.markdown("---")
    st.markdown("### 📊 Status")

    # Ollama status check
    try:
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        models = [m["name"] for m in r.json().get("models", [])]
        st.success("🟢 Ollama running")
        for m in models:
            st.caption(f"  • {m}")
    except Exception:
        st.error("🔴 Ollama not running")
        st.code("ollama serve &", language="bash")

    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    if st.button("💣 Reset vector store"):
        import shutil
        if Path(CHROMA_DIR).exists():
            shutil.rmtree(CHROMA_DIR)
        st.cache_resource.clear()
        st.success("Vector store cleared — reload to rebuild.")
        st.rerun()


# ── MAIN UI ───────────────────────────────────────────────────
st.title("🚢 Vehicle manual Documentation RAG Chatbot")
st.warning(
        "⚠️ **POC Notice — Please Read Before Testing**\n\n"
        "This application is a **Proof of Concept (POC)** designed exclusively for "
        "**This RAG is scoped to the following documents only:**\n"
        "- 📄 `2023-bmw-x1.pdf` — vehicle manual assistant for the 2023 BMW X1\n"
        "**Do not upload or test with out-of-scope files.** "
        "The retrieval model is tuned for vessel/terminal domain language. "
        "Uploading unrelated documents (e.g. vehicle manuals, HR policies, product catalogues) "
        "will cause the model to retrieve irrelevant content and produce hallucinated or misleading answers — "
        "even if the response sounds confident.\n\n"   
        "Out-of-scope queries will not return reliable results.\n\n"
        "Please stay within the defined scope for meaningful and accurate results during this POC phase."
    )
st.caption(f"Model: `{LLM_MODEL}` · Embeddings: `{EMBED_MODEL}` · Docs: `{DOCS_DIR}`")

# Load chain with spinner
with st.spinner("⚙️ Loading RAG pipeline (first load takes 2–3 min for embedding)..."):
    chain, error = load_chain()

if error:
    st.warning(f"⚠️ {error}")
    st.info("Upload a PDF using the sidebar, then the chatbot will initialize automatically.")
    st.stop()

st.success("✅ RAG pipeline ready — ask your vessel questions below!")
st.markdown("---")

# ── CHAT HISTORY ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.caption(src)
        if msg["role"] == "assistant" and show_chunks and msg.get("chunks"):
            with st.expander("🔍 Retrieved chunks"):
                for i, chunk in enumerate(msg["chunks"]):
                    st.text_area(f"Chunk {i+1}", chunk, height=100)


# ── CHAT INPUT ────────────────────────────────────────────────
if query := st.chat_input("Ask about your vessel documentation..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                start = time.time()
                result = chain({"question": query})
                elapsed = time.time() - start

                answer = result["answer"]
                source_docs = result.get("source_documents", [])

                # Format sources
                seen = set()
                sources = []
                for doc in source_docs:
                    src  = Path(doc.metadata.get("source", "Unknown")).name
                    page = doc.metadata.get("page", "")
                    ref  = f"{src}" + (f" — page {int(page)+1}" if page != "" else "")
                    if ref not in seen:
                        sources.append(ref)
                        seen.add(ref)

                # Format chunks
                chunks_text = [doc.page_content[:300] + "..." for doc in source_docs]

                # Display answer
                st.markdown(answer)
                st.caption(f"⏱️ {elapsed:.1f}s · {len(source_docs)} chunks retrieved")

                if show_sources and sources:
                    with st.expander("📎 Sources"):
                        for src in sources:
                            st.caption(f"• {src}")

                if show_chunks and chunks_text:
                    with st.expander("🔍 Retrieved chunks"):
                        for i, chunk in enumerate(chunks_text):
                            st.text_area(f"Chunk {i+1}", chunk, height=100)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "chunks": chunks_text
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure Ollama is running: `ollama serve &` in terminal")