"""
Multi-File RAG Chatbot — Separate Collections per File
Supports: PDF, TXT, MD, DOCX, XLSX, CSV, JPG, PNG
Platform: Lightning.ai + Ollama + LangChain + ChromaDB
Run: streamlit run app_multifile.py

Install deps:
    pip install langchain langchain-community chromadb pypdf \
        unstructured docx2txt openpyxl pytesseract Pillow streamlit
    Linux: sudo apt-get install tesseract-ocr
    Mac:   brew install tesseract
"""

import os
import time
import json
import shutil
import hashlib
import streamlit as st
from pathlib import Path
from typing import Optional

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Collection RAG",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CONFIG ────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "llama3.2"
EMBED_MODEL     = "nomic-embed-text"
DOCS_DIR        = "./documents"
CHROMA_DIR      = "./chroma_db"
META_FILE       = "./documents/.meta.json"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 200
TOP_K           = 5

SUPPORTED_TYPES = {
    "pdf":  {"icon": "📄", "label": "PDF",      "is_image": False},
    "txt":  {"icon": "📝", "label": "Text",     "is_image": False},
    "md":   {"icon": "📝", "label": "Markdown", "is_image": False},
    "docx": {"icon": "📘", "label": "Word",     "is_image": False},
    "xlsx": {"icon": "📊", "label": "Excel",    "is_image": False},
    "csv":  {"icon": "📊", "label": "CSV",      "is_image": False},
    "jpg":  {"icon": "🖼️", "label": "Image",    "is_image": True},
    "jpeg": {"icon": "🖼️", "label": "Image",    "is_image": True},
    "png":  {"icon": "🖼️", "label": "Image",    "is_image": True},
}


def get_ollama_models() -> tuple[list[str], list[str]]:
    import requests

    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
    response.raise_for_status()
    names = [model.get("name", "") for model in response.json().get("models", [])]
    bases = [name.split(":")[0] for name in names if name]
    return names, bases


def resolve_embedding_model() -> str:
    names, bases = get_ollama_models()
    if EMBED_MODEL in names or EMBED_MODEL in bases:
        return EMBED_MODEL

    embed_candidates = [name for name in names if "embed" in name.lower()]
    if embed_candidates:
        return embed_candidates[0]

    raise RuntimeError(
        f"No Ollama embedding model available. Pull one first: `ollama pull {EMBED_MODEL}`"
    )


# ── METADATA HELPERS ──────────────────────────────────────────
def load_meta() -> dict:
    if Path(META_FILE).exists():
        with open(META_FILE, "r") as f:
            return json.load(f)
    return {}

def save_meta(meta: dict):
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def safe_collection_name(filename: str) -> str:
    stem = Path(filename).stem
    safe = "".join(c if c.isalnum() else "_" for c in stem)
    return f"col_{safe[:40]}"


# ── IMAGE OCR ─────────────────────────────────────────────────
def extract_text_from_image(filepath: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
        img  = Image.open(filepath)
        text = pytesseract.image_to_string(img)
        return text.strip() if text.strip() else "[No readable text detected in image]"
    except ImportError:
        return "[Install pytesseract + Pillow for image OCR]"
    except Exception as e:
        return f"[OCR error: {e}]"


# ── FILE LOADER ───────────────────────────────────────────────
def load_file(filepath: Path) -> list:
    from langchain_core.documents import Document
    ext  = filepath.suffix.lower().lstrip(".")
    info = SUPPORTED_TYPES.get(ext, {})
    docs = []

    try:
        if info.get("is_image"):
            text = extract_text_from_image(filepath)
            docs = [Document(
                page_content=text,
                metadata={"source": str(filepath), "filename": filepath.name, "filetype": ext, "page": 0}
            )]

        elif ext == "pdf":
            from langchain_community.document_loaders import PyPDFLoader
            docs = PyPDFLoader(str(filepath)).load()

        elif ext == "txt":
            from langchain_community.document_loaders import TextLoader
            docs = TextLoader(str(filepath), encoding="utf-8").load()

        elif ext == "md":
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            docs = UnstructuredMarkdownLoader(str(filepath)).load()

        elif ext == "docx":
            from langchain_community.document_loaders import Docx2txtLoader
            docs = Docx2txtLoader(str(filepath)).load()

        elif ext == "csv":
            from langchain_community.document_loaders import CSVLoader
            docs = CSVLoader(str(filepath), encoding="utf-8").load()

        elif ext == "xlsx":
            from langchain_community.document_loaders import UnstructuredExcelLoader
            docs = UnstructuredExcelLoader(str(filepath), mode="elements").load()

        for doc in docs:
            doc.metadata.setdefault("filename", filepath.name)
            doc.metadata.setdefault("filetype", ext)

        return docs

    except Exception as e:
        st.warning(f"⚠️ Could not load `{filepath.name}`: {e}")
        return []


# ── BUILD COLLECTION ──────────────────────────────────────────
def build_collection(filepath: Path) -> tuple:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma

    docs = load_file(filepath)
    if not docs:
        return False, f"No content extracted from {filepath.name}"

    ext = filepath.suffix.lower().lstrip(".")
    if SUPPORTED_TYPES.get(ext, {}).get("is_image"):
        chunks = docs
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)

    try:
        embed_model = resolve_embedding_model()
    except Exception as e:
        return False, str(e)

    embeddings = OllamaEmbeddings(model=embed_model, base_url=OLLAMA_BASE_URL)
    col_name   = safe_collection_name(filepath.name)
    col_dir    = str(Path(CHROMA_DIR) / col_name)

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=col_name,
        persist_directory=col_dir
    )
    vs.persist()
    return True, f"Indexed {len(chunks)} chunks"


def delete_collection(filename: str):
    col_dir = Path(CHROMA_DIR) / safe_collection_name(filename)
    if col_dir.exists():
        shutil.rmtree(col_dir)
    meta = load_meta()
    if filename in meta:
        fpath = Path(meta[filename].get("path", ""))
        if fpath.exists():
            fpath.unlink()
        del meta[filename]
        save_meta(meta)


# ── TEMPLATE REGISTRY ────────────────────────────────────────
# Add new file entries here as you upload more documents.
# The key must exactly match the uploaded filename.
TEMPLATES = {
    "2023-bmw-x1.pdf": {
        "role":      "vehicle manual assistant for the 2023 BMW X1",
        "cite":      "Always cite page numbers and quote exact specs (torque values, tire pressure, fuel type, fluid capacities)",
        "safety":    "For ALL safety-critical topics (brakes, airbags, warning lights, child seats), append: 'Verify this in your physical BMW X1 manual before acting.'",
        "not_found": "This is not covered in the provided BMW X1 Owner's Manual.",
        "extras":    "If the question involves a warning light or dashboard symbol, describe what it looks like and what action is required.",
    },
    "IATA-Cargo-Interchange-Message-Procedure.pdf": {
        "role":      "cargo documentation assistant specializing in IATA interchange message procedures",
        "cite":      "Cite section numbers, message type codes, and field names precisely — these are compliance-critical",
        "safety":    "For compliance or regulatory procedures, append: 'Verify against the current official IATA publication before implementation.'",
        "not_found": "This is not covered in the provided IATA Cargo Interchange Message Procedure documentation.",
        "extras":    "When referencing message formats, reproduce field names and codes exactly as they appear in the document.",
    },
    # ── APM Terminals Maasvlakte II — Vessel & Container Operator Manual ──
    "operational-manual-vessel-and-container-operators-v-42.pdf": {
        "role":      "terminal operations assistant for APM Terminals Maasvlakte II (MVII), specializing in vessel and container operator procedures",
        "cite":      "Always cite the chapter, section number, and page number (e.g. Section 4.3.1, page 9). Quote exact deadlines, email addresses, and phone numbers verbatim",
        "safety":    "For ALL HSSE, IMO 1, IMO 7, confined space, or radiation procedures, append: Verify with the HSSE department at MVII before proceeding. Emergency: +31 (0)6 83076494.",
        "not_found": "This is not covered in the provided APM Terminals MVII Operational Manual v4.2.",
        "extras":    "Always distinguish vessel operator vs container operator responsibilities. For deadlines quote exact hours from Appendix II. For contacts provide exact email and phone from Appendix I. EDI codes (BAPLIE, COPRAR, MOVINS etc.) must be reproduced exactly. For Yard Opening Time always state the 5-day rule and 24hr closure before ETA.",
    },
}

# Fallback used for any file not listed above
DEFAULT_TEMPLATE = {
    "role":      "document assistant",
    "cite":      "Always cite page numbers or sections when available",
    "safety":    "For critical information, recommend verifying with the original source document",
    "not_found": "This information is not found in the selected documents.",
    "extras":    "Quote exact values from tables, specifications, or numbered procedures.",
}


def build_dynamic_prompt(selected_files: list):
    """
    Builds the best prompt for the current file selection:
    - 1 file  → file-specific tailored prompt
    - 2+ files → merged multi-doc prompt with per-file rules
    - unknown file → safe default prompt
    """
    from langchain_core.prompts import PromptTemplate

    # ── Single file ───────────────────────────────────────────
    if len(selected_files) == 1:
        fname  = selected_files[0]
        cfg    = TEMPLATES.get(fname, DEFAULT_TEMPLATE)
        template = f"""You are a {cfg['role']}.

Rules:
- Answer ONLY from the context below — never use outside knowledge
- {cfg['cite']}
- {cfg['safety']}
- {cfg['extras']}
- If the answer is not in the context, say exactly: "{cfg['not_found']}"

=== CONTEXT ===
{{context}}

=== CHAT HISTORY ===
{{chat_history}}

=== QUESTION ===
{{question}}

=== ANSWER ==="""

    # ── Multiple files → merged rules ────────────────────────
    else:
        scope = ", ".join(selected_files)
        cite_lines   = []
        safety_lines = []
        extra_lines  = []

        for fname in selected_files:
            cfg   = TEMPLATES.get(fname, DEFAULT_TEMPLATE)
            stem  = Path(fname).stem
            cite_lines.append(f"  • {stem}: {cfg['cite']}")
            safety_lines.append(f"  • {stem}: {cfg['safety']}")
            extra_lines.append(f"  • {stem}: {cfg['extras']}")

        template = f"""You are a multi-document assistant with access to: {scope}

Rules:
- Answer ONLY from the context below — never use outside knowledge
- ALWAYS clearly state which document each piece of information comes from
- If the same topic appears in multiple files, compare them and cite each source
- Citation rules per document:
{chr(10).join(cite_lines)}
- Safety / compliance rules per document:
{chr(10).join(safety_lines)}
- Additional rules per document:
{chr(10).join(extra_lines)}
- If the answer is not in the context, say: "This is not found in the selected documents."

=== CONTEXT ===
{{context}}

=== CHAT HISTORY ===
{{chat_history}}

=== QUESTION ===
{{question}}

=== ANSWER ==="""

    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )


# ── CHAIN BUILDER (LCEL — LangChain 0.3+ compatible) ──────────
def build_chain(selected_files: list):
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_community.vectorstores import Chroma
    from langchain_community.retrievers import MergerRetriever
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda

    try:
        embed_model = resolve_embedding_model()
    except Exception as e:
        st.error(str(e))
        return None

    embeddings = OllamaEmbeddings(model=embed_model, base_url=OLLAMA_BASE_URL)
    retrievers = []

    for fname in selected_files:
        col_name = safe_collection_name(fname)
        col_dir  = str(Path(CHROMA_DIR) / col_name)
        if not Path(col_dir).exists():
            continue
        try:
            vs = Chroma(
                collection_name=col_name,
                embedding_function=embeddings,
                persist_directory=col_dir
            )
            retrievers.append(vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": TOP_K, "fetch_k": 20}
            ))
        except Exception:
            continue

    if not retrievers:
        return None

    retriever = retrievers[0] if len(retrievers) == 1 else MergerRetriever(retrievers=retrievers)
    llm       = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2, num_ctx=4096)
    prompt    = build_dynamic_prompt(selected_files)

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {d.metadata.get('filename', Path(d.metadata.get('source','?')).name)}"
            f" | Page: {int(d.metadata.get('page', 0)) + 1}]\n{d.page_content}"
            for d in docs
        )

    # LCEL chain — retriever returns docs, we keep them for source display
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"]))),
            source_documents=RunnableLambda(lambda x: retriever.invoke(x["question"]))
        )
        | RunnablePassthrough.assign(
            answer=(prompt | llm | StrOutputParser())
        )
    )

    return chain


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🗄️ Multi-Collection RAG")
    st.caption(f"Model: `{LLM_MODEL}`")
    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────
    st.markdown("### 📁 Upload Files")
    st.caption("PDF · TXT · MD · DOCX · XLSX · CSV · JPG · PNG")

    uploaded = st.file_uploader(
        "files", type=list(SUPPORTED_TYPES.keys()),
        accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploaded:
        os.makedirs(DOCS_DIR, exist_ok=True)
        meta = load_meta()
        to_index = []

        for uf in uploaded:
            save_path = Path(DOCS_DIR) / uf.name
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            fhash = file_hash(str(save_path))
            if uf.name not in meta or meta[uf.name]["hash"] != fhash:
                to_index.append(save_path)
            meta[uf.name] = {
                "hash": fhash,
                "path": str(save_path),
                "collection": safe_collection_name(uf.name),
                "filetype": save_path.suffix.lstrip(".")
            }
        save_meta(meta)

        if to_index:
            bar = st.progress(0, text="Indexing...")
            for i, fp in enumerate(to_index):
                bar.progress((i + 1) / len(to_index), text=f"Indexing {fp.name}...")
                ok, msg = build_collection(fp)
                st.success(f"✅ {fp.name}") if ok else st.warning(f"⚠️ {msg}")
            bar.empty()
            st.session_state.pop("chain_cache", None)
            st.rerun()
        else:
            st.info("All files already indexed")

    st.markdown("---")

    # ── File library with checkboxes ──────────────────────────
    st.markdown("### 📚 File Library")
    meta = load_meta()

    if "selected_files" not in st.session_state:
        st.session_state.selected_files = list(meta.keys())

    col1, col2 = st.columns(2)
    if col1.button("✅ All", use_container_width=True):
        st.session_state.selected_files = list(meta.keys())
        st.session_state.pop("chain_cache", None)
        st.rerun()
    if col2.button("⬜ None", use_container_width=True):
        st.session_state.selected_files = []
        st.session_state.pop("chain_cache", None)
        st.rerun()

    selected = []
    for fname, info in meta.items():
        ext  = info.get("filetype", "")
        icon = SUPPORTED_TYPES.get(ext, {}).get("icon", "📄")
        c1, c2 = st.columns([5, 1])
        if c1.checkbox(f"{icon} {fname}", value=fname in st.session_state.selected_files, key=f"chk_{fname}"):
            selected.append(fname)
        if c2.button("🗑️", key=f"del_{fname}"):
            delete_collection(fname)
            if fname in st.session_state.get("selected_files", []):
                st.session_state.selected_files.remove(fname)
            st.session_state.pop("chain_cache", None)
            st.rerun()

    # Detect selection change → rebuild chain
    if sorted(selected) != sorted(st.session_state.selected_files):
        st.session_state.selected_files = selected
        st.session_state.pop("chain_cache", None)

    st.markdown("---")

    # ── Settings ──────────────────────────────────────────────
    st.markdown("### ⚙️ Settings")
    show_sources = st.toggle("Show sources", value=True)
    show_chunks  = st.toggle("Show chunks",  value=False)
    show_preview = st.toggle("Image preview", value=True)

    st.markdown("---")

    # ── Ollama status ─────────────────────────────────────────
    st.markdown("### 📊 Status")
    try:
        import requests
        r      = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        models = [m["name"] for m in r.json().get("models", [])]
        st.success("🟢 Ollama running")
        for m in models:
            st.caption(f"  • {m}")
    except Exception:
        st.error("🔴 Ollama not running")
        st.code("ollama serve &")

    st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.pop("chain_cache", None)
        st.rerun()
    if c2.button("💣 Reset all"):
        for d in [CHROMA_DIR, DOCS_DIR]:
            if Path(d).exists():
                shutil.rmtree(d)
        st.session_state.clear()
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════
st.title("🗄️ Multi-Collection RAG Chatbot")
st.caption("Every file is its own knowledge base — mix and match any combination")

st.warning(
    "⚠️ **POC Notice — Please Read Before Testing**\n\n"
    "This application is a **Proof of Concept (POC)** designed exclusively for "
    "**Vessel & Container Terminal Operations** at APM Terminals Maasvlakte II (MVII).\n\n"
    "**This RAG is scoped to the following documents only:**\n"
    "- 📄 `operational-manual-vessel-and-container-operators-v-42.pdf` — APM Terminals MVII Operational Manual\n"
    "- 📄 `IATA-Cargo-Interchange-Message-Procedure.pdf` — IATA Cargo EDI Procedures\n\n"
    "**Do not upload or test with out-of-scope files.** "
    "The retrieval model is tuned for vessel/terminal domain language. "
    "Uploading unrelated documents (e.g. vehicle manuals, HR policies, product catalogues) "
    "will cause the model to retrieve irrelevant content and produce hallucinated or misleading answers — "
    "even if the response sounds confident.\n\n"
    "✅ **In scope:** Vessel operational procedures, container handling, HSSE guidelines, "
    "EDI message formats, berth/port specifications, cargo cut-off rules.\n\n"
    "❌ **Out of scope:** Anything unrelated to vessel/terminal operations at MVII. "
    "Out-of-scope queries will not return reliable results.\n\n"
    "Please stay within the defined scope for meaningful and accurate results during this POC phase."
)

# ── Stats bar ─────────────────────────────────────────────────
if meta:
    type_counts = {}
    for fname in meta:
        ext   = meta[fname].get("filetype", "")
        label = f"{SUPPORTED_TYPES.get(ext,{}).get('icon','📄')} {SUPPORTED_TYPES.get(ext,{}).get('label', ext.upper())}"
        type_counts[label] = type_counts.get(label, 0) + 1

    cols = st.columns(len(type_counts) + 2)
    for i, (k, v) in enumerate(type_counts.items()):
        cols[i].metric(k, v)
    cols[-2].metric("📂 Total", len(meta))
    cols[-1].metric("✅ Active", len(st.session_state.get("selected_files", [])))
    st.markdown("---")

# ── Image preview ─────────────────────────────────────────────
if show_preview:
    imgs = [
        (f, info) for f, info in meta.items()
        if SUPPORTED_TYPES.get(info.get("filetype",""), {}).get("is_image")
        and f in st.session_state.get("selected_files", [])
    ]
    if imgs:
        with st.expander(f"🖼️ Selected images ({len(imgs)})", expanded=True):
            ic = st.columns(min(3, len(imgs)))
            for i, (fname, info) in enumerate(imgs):
                fp = Path(info.get("path", ""))
                if fp.exists():
                    ic[i % 3].image(str(fp), caption=fname, use_container_width=True)
                    ocr = extract_text_from_image(fp)
                    ic[i % 3].caption(f"OCR preview: {ocr[:120]}..." if len(ocr) > 120 else f"OCR: {ocr}")

# ── Guards ────────────────────────────────────────────────────
if not meta:
    st.info("👈 Upload files using the sidebar to get started.")
    st.stop()

if not st.session_state.get("selected_files"):
    st.warning("⚠️ No files selected — check files in the sidebar.")
    st.stop()

# ── Build/cache chain ─────────────────────────────────────────
if "chain_cache" not in st.session_state:
    with st.spinner(f"⚙️ Loading {len(st.session_state.selected_files)} collection(s)..."):
        chain = build_chain(st.session_state.selected_files)
    if not chain:
        st.error("Could not build retriever. Try re-uploading files.")
        st.stop()
    st.session_state["chain_cache"] = chain

chain = st.session_state["chain_cache"]

# ── Active scope display ──────────────────────────────────────
pills = "  ·  ".join([
    f"{SUPPORTED_TYPES.get(meta[f].get('filetype',''),{}).get('icon','📄')} **{f}**"
    for f in st.session_state.selected_files
])
st.markdown(f"**Querying:** {pills}")
st.markdown("---")

# ── Chat history ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.caption(f"• {src}")
        if msg["role"] == "assistant" and show_chunks and msg.get("chunks"):
            with st.expander("🔍 Chunks"):
                for i, chunk in enumerate(msg["chunks"]):
                    st.text_area(f"Chunk {i+1}", chunk, height=100, key=f"hc_{id(msg)}_{i}")

# ── Starter suggestions ───────────────────────────────────────
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    examples = [
        "Summarize each document",
        "What are the key topics across all files?",
        "Compare information between files",
        "What does the image contain?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, use_container_width=True):
            st.session_state["pending_query"] = ex
            st.rerun()

# ── Input ─────────────────────────────────────────────────────
query = st.chat_input("Ask across your selected files...")
if not query and "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching collections..."):
            try:
                start   = time.time()
                # LCEL chain invocation — chat_history managed manually in session
                chat_history = "\n".join([
                    f"{m['role'].upper()}: {m['content']}"
                    for m in st.session_state.messages[-10:]
                    if m["role"] in ("user", "assistant")
                ])
                result  = chain.invoke({
                    "question": query,
                    "chat_history": chat_history
                })
                elapsed = time.time() - start

                answer      = result.get("answer", "No answer returned.")
                source_docs = result.get("source_documents", [])

                seen, sources = set(), []
                for doc in source_docs:
                    fname = doc.metadata.get("filename", Path(doc.metadata.get("source","?")).name)
                    page  = doc.metadata.get("page", "")
                    ext   = Path(fname).suffix.lstrip(".")
                    icon  = SUPPORTED_TYPES.get(ext, {}).get("icon", "📄")
                    ref   = f"{icon} {fname}" + (f" — page {int(page)+1}" if page != "" else "")
                    if ref not in seen:
                        sources.append(ref)
                        seen.add(ref)

                chunks_text  = [d.page_content[:300] + "..." for d in source_docs]
                unique_files = list({s.split(" — ")[0] for s in sources})

                st.markdown(answer)
                st.caption(f"⏱️ {elapsed:.1f}s · {len(source_docs)} chunks · {len(unique_files)} file(s) referenced")

                if show_sources and sources:
                    with st.expander("📎 Sources"):
                        for src in sources:
                            st.caption(f"• {src}")

                if show_chunks and chunks_text:
                    with st.expander("🔍 Retrieved chunks"):
                        for i, c in enumerate(chunks_text):
                            st.text_area(f"Chunk {i+1}", c, height=100, key=f"nc_{i}_{int(time.time())}")

                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sources": sources, "chunks": chunks_text
                })

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure Ollama is running: `ollama serve &`")