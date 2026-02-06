# 🧠 Local RAG Application

A fully local **Retrieval-Augmented Generation (RAG)** system that answers questions strictly from the documents you provide — with **zero external internet dependency**.

This project combines semantic search, vector storage, and local LLM generation to create a private, offline AI knowledge assistant.

---

# 🚀 Features

* ✅ 100% Offline Operation
* ✅ Document chunking with metadata tracking
* ✅ Semantic search using embeddings
* ✅ Qdrant vector database for storage
* ✅ Ollama for local LLM text generation
* ✅ Strict answer grounding (refuses if answer is not in documents)
* ✅ CLI + Web UI (Streamlit)

---

# 📋 Prerequisites

## 1️⃣ Install Qdrant

### Docker Method (Recommended)

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### OR Standalone Binary

Download from:
[https://github.com/qdrant/qdrant/releases](https://github.com/qdrant/qdrant/releases)

---

## 2️⃣ Install Ollama

Download from:
[https://ollama.ai/download](https://ollama.ai/download)

After installation, pull a model:

```bash
ollama pull llama2
```

You can use other models as well (mistral, phi3, etc.).

---

## 3️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

# 📁 Project Structure

```
LOCAL_ADV_Rag/
├── docs/                    # Store .txt documents
├── src/
│   ├── document_loader.py   # Load & chunk documents
│   ├── embedder.py          # Generate embeddings
│   ├── vector_store.py      # Qdrant integration
│   ├── retriever.py         # Retrieve relevant chunks
│   └── rag_pipeline.py      # Main RAG pipeline
│
├── streamlit_app.py         # Streamlit Web UI
├── STREAMLIT_GUIDE.md       # Streamlit documentation
├── main.py                  # CLI entry point
├── requirements.txt
└── README.md
```

---

# 🧠 Usage

## 1️⃣ Add Documents

Place your `.txt` files inside:

```
docs/
```

---

## 2️⃣ Index Documents

Run whenever documents are added or updated:

```bash
python main.py --index
```

---

## 3️⃣ Ask Questions

```bash
python main.py --query "Your question here"
```

Example:

```bash
python main.py --query "What is Machine Learning?"
```

---

## 4️⃣ Interactive Chat Mode

```bash
python main.py --interactive
```

---

# 🌐 Streamlit Web UI

This project also includes a **Streamlit-based web interface** for interactive querying.

---

## ▶️ Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

Open in browser:

```
http://localhost:8501
```

---

## ✨ Streamlit Features

* Upload and manage documents
* Ask questions interactively
* View retrieved context chunks
* Grounded AI answers
* Source document tracking
* Clean web interface

---

## 📄 Streamlit File Reference

```
streamlit_app.py      → Main Streamlit interface
STREAMLIT_GUIDE.md   → Detailed setup & usage guide
```

For full instructions, see:

**STREAMLIT_GUIDE.md**

---

## 🔄 Streamlit vs CLI

| Mode            | Command                          | Use Case       |
| --------------- | -------------------------------- | -------------- |
| CLI Query       | `python main.py --query`         | Quick queries  |
| Interactive CLI | `python main.py --interactive`   | Terminal chat  |
| Web UI          | `streamlit run streamlit_app.py` | User interface |

---

# ⚙️ How It Works

### 1. Document Loading

Reads all `.txt` files from `docs/`.

### 2. Chunking

Splits documents into **300–500 word chunks** with IDs.

### 3. Embedding

Converts chunks into vectors using Sentence-Transformers.

### 4. Storage

Stores embeddings in Qdrant with metadata.

### 5. Retrieval

Finds Top-K similar chunks via cosine similarity.

### 6. Generation

Ollama generates answers using retrieved context.

### 7. Validation

System refuses answers not grounded in documents.

---

# 🔧 Configuration

Edit settings in **main.py**:

| Parameter       | Description      | Default          |
| --------------- | ---------------- | ---------------- |
| EMBEDDING_MODEL | Embedding model  | all-MiniLM-L6-v2 |
| LLM_MODEL       | Ollama model     | llama2           |
| CHUNK_SIZE      | Words per chunk  | 300–500          |
| TOP_K           | Retrieved chunks | 3                |

---

# 🛠️ Troubleshooting

## Qdrant Connection Error

* Ensure Qdrant runs on port **6333**

Test:

```bash
curl http://localhost:6333/
```

---

## Ollama Connection Error

Check installation:

```bash
ollama list
```

---

## CUDA / GPU Errors

If GPU fails, run Ollama in CPU mode:

```bash
setx OLLAMA_NO_CUDA 1
```

Restart terminal afterward.

---

## No Documents Found

* Ensure files are in `docs/`
* Use `.txt` format only

---

# 📝 Notes

* First run downloads embedding model (~80MB)
* Ollama models may be several GB
* All processing is local
* No internet required after setup

---

# 📦 Releases & Packages

No releases published yet.

You can create versions via the **GitHub Releases** section when distributing updates.

---

# 👨‍💻 Author

Developed as a **Local Advanced RAG System** for private, offline document question answering.

---

⭐ If you like this project, consider starring the repository!

