# Local RAG Application

A fully local Retrieval-Augmented Generation (RAG) system that answers questions using only provided documents.

## Features
- 100% offline operation
- Document chunking with metadata tracking
- Semantic search using embeddings
- Qdrant vector database for storage
- Ollama for local LLM generation
- Strict answer grounding (refuses if answer not in docs)

## Prerequisites

1. **Install Qdrant** (Docker method - easiest):
   ```bash
   docker pull qdrant/qdrant
   docker run -p 6333:6333 qdrant/qdrant
   ```
   
   Or download Qdrant standalone binary from: https://github.com/qdrant/qdrant/releases

2. **Install Ollama**:
   - Download from: https://ollama.ai/download
   - After installation, pull a model:
   ```bash
   ollama pull llama2
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
LOCAL_ADV_Rag/
├── docs/                  # Place your .txt files here
├── src/
│   ├── document_loader.py # Load and chunk documents
│   ├── embedder.py        # Generate embeddings
│   ├── vector_store.py    # Qdrant integration
│   ├── retriever.py       # Retrieve relevant chunks
│   └── rag_pipeline.py    # Main RAG logic
├── main.py                # Entry point
├── requirements.txt
└── README.md
```

## Usage

1. **Add documents**: Place your .txt files in the `docs/` folder

2. **Index documents** (run once or when documents change):
   ```bash
   python main.py --index
   ```

3. **Ask questions**:
   ```bash
   python main.py --query "Your question here"
   ```

4. **Interactive mode**:
   ```bash
   python main.py --interactive
   ```

## Example

```bash
# Index your documents
python main.py --index

# Ask a question
python main.py --query "What is machine learning?"

# Interactive chat
python main.py --interactive
```

## How It Works

1. **Document Loading**: Reads all .txt files from `docs/` folder
2. **Chunking**: Splits documents into 300-500 word chunks with unique IDs
3. **Embedding**: Converts chunks to vectors using sentence-transformers
4. **Storage**: Stores embeddings in Qdrant with metadata
5. **Retrieval**: Finds top-3 most similar chunks using cosine similarity
6. **Generation**: Uses Ollama to generate grounded answers
7. **Validation**: Refuses to answer if context doesn't contain the answer

## Configuration

Edit the configuration in `main.py`:
- `EMBEDDING_MODEL`: Default is "all-MiniLM-L6-v2"
- `LLM_MODEL`: Default is "llama2"
- `CHUNK_SIZE`: Words per chunk (default 300-500)
- `TOP_K`: Number of chunks to retrieve (default 3)

## Troubleshooting

**Qdrant connection error**:
- Ensure Qdrant is running on port 6333
- Check: `curl http://localhost:6333/`

**Ollama connection error**:
- Ensure Ollama is installed and running
- Check: `ollama list`

**No documents found**:
- Place .txt files in the `docs/` folder
- Ensure files have .txt extension

## Notes

- First run downloads the embedding model (~80MB)
- Ollama models can be large (several GB)
- All data stays on your machine
- No internet required after setup
