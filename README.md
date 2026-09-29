# NeuroSphere

A neural retrieval and semantic knowledge search platform designed for indexing and searching local document corpora with dense vector embeddings.

---

## Repository Structure

```text
NeuroSphere/
│
├── data/
│   ├── index/
│   └── test_corpus/
│
├── neurosphere/
│   ├── __init__.py
│   ├── config.py
│   ├── database.py
│   ├── extractors.py
│   ├── chunker.py
│   ├── scanner.py
│   ├── embeddings.py
│   ├── indexer.py
│   └── search.py
│
├── app.py
├── build_index.py
├── run.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Core Components

- **`data/`**: Storage directories for raw documents (`test_corpus/`) and generated vector indices and SQLite metadata (`index/`).
- **`neurosphere/config.py`**: Global configuration for chunk sizes, model names, storage paths, and network ports.
- **`neurosphere/database.py`**: SQLite database layer managing documents, chunks, hashes, and relational mappings.
- **`neurosphere/extractors.py`**: Multi-format text extractors supporting `.txt`, `.md`, `.json`, `.csv`, `.pdf`, code files, and more.
- **`neurosphere/chunker.py`**: Text chunking engine with configurable chunk size, token overlap, and boundary-aware splitting.
- **`neurosphere/scanner.py`**: Recursive file scanner detecting new, modified, and removed files using SHA-256 hashes.
- **`neurosphere/embeddings.py`**: Vector embedding manager supporting SentenceTransformers (`all-MiniLM-L6-v2`) with deterministic fallback support.
- **`neurosphere/indexer.py`**: Vector index builder and cosine similarity search engine.
- **`neurosphere/search.py`**: Semantic search engine querying the index and retrieving top matches with document context.
- **`app.py`**: Modern web user interface and REST API server.
- **`build_index.py`**: CLI script to scan the corpus and build the vector index.
- **`run.py`**: Unified command-line interface for building, searching, inspecting, and serving.

---

## Quick Start

### 1. Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Populate Corpus

Place documents (`.txt`, `.md`, `.pdf`, etc.) in `data/test_corpus/`.

### 3. Build the Vector Index

Run the index builder:

```bash
python build_index.py
```

Or using the unified CLI:

```bash
python run.py build
```

To force a full rebuild:

```bash
python run.py build --rebuild
```

### 4. Search via Command Line

Perform instant semantic searches from your terminal:

```bash
python run.py search "dense vector embeddings"
```

### 5. Launch Web UI & REST API

Start the interactive web app:

```bash
python app.py
```

Then open your browser at **`http://127.0.0.1:8000`**.

API endpoints:
- `GET /api/search?q=<query>&top_k=5`
- `GET /api/stats`
- `POST /api/reindex`
