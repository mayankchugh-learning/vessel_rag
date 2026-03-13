# vessel_rag

### Add uv to PATH using Windows Command Prompt alternatively it can be set permanently 

#### Set the User environment variable:
```bash 
cmd
setx Path "%PATH%;C:\Users\madfa\.local\bin"
```

#### Update the current session's PATH variable:
```bash 
cmd
set PATH=%PATH%;C:\Users\madfa\.local\bin
```

### Verify installation
```bash
uv --version
uv self version
```

### Initialize a new project
```bash
uv init 
uv python install 3.11
uv venv --python 3.11
```

### Activate virtual environment
```bash
.venv\Scripts\activate

python -m venv .venv

.\.venv\Scripts\activate

uv add -r requirements.txt
pip install -r requirements.txt
```

### Pull required Ollama models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### start app
```bash
streamlit run app_multifile.py --server.port 8080
```
### pip install langchain==0.2.16
### pip install langchain-community==0.2.16
### pip install langchain-huggingface
### pip install chromadb==0.5.5
### pip install pypdf==4.3.1
### pip install unstructured
### pip install sentence-transformers
### pip install transformers
### pip install accelerate
### pip install bitsandbytes
### pip install streamlit
### pip install torch --quiet

### pip install -U langchain-chroma
### pip install --upgrade posthog

### pip install pydantic --upgrade
### pip install langchain-community langchain --upgrade

### pip install -U langchain-ollama langchain-text-splitters langchain-core langchain-community --break-system-packages\

### pip install -U langchain-ollama