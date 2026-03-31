# ESS Governance Assistant

A Retrieval-Augmented Generation (RAG) chatbot for the University of Ottawa Engineering Students' Society (ESS). It answers questions about meeting minutes, policies, and bylaws using a local LLM and a semantic vector database.

---

## Overview

The assistant parses, vectorizes, and semantically searches governance documents — including Board of Directors (BoD) meeting minutes, officer meeting minutes, policies, and bylaws — and uses a locally hosted LLM to generate grounded answers.

---

## Architecture

```
Documents/ (.txt)
     │
     ▼
preprocessor.py      — Parses documents into structured chunks with metadata
     │
     ▼
vectorization.py     — Embeds chunks using BGE and stores them in ChromaDB
     │
     ▼
retrieval.py         — Detects query intent and retrieves relevant chunks
     │
     ▼
llm.py               — Builds context, generates HyDE, and calls the LLM
     │
     ▼
app.py               — Streamlit chat interface
```

---

## Project Structure

| File | Description |
|---|---|
| `app.py` | Streamlit web UI |
| `llm.py` | Context building, HyDE, LLM orchestration |
| `retrieval.py` | ChromaDB query logic, filters, date resolution |
| `preprocessor.py` | Document parsing and chunk extraction |
| `vectorization.py` | Embedding and upserting chunks into ChromaDB |
| `run_preprocessing.py` | Script to preprocess all documents in `Documents/` |
| `rag_test.py` | Pytest test suite |

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with `llama3.1:8b` pulled

### Install dependencies

```bash
pip install streamlit chromadb sentence-transformers spacy python-dateutil requests torch pytest
python -m spacy download en_core_web_sm
```

### Add documents

Place `.txt` governance documents in a `Documents/` folder in the project root. Filenames are used to detect document type (e.g. `bod-2026-02.txt`, `policy-travel.txt`, `bylaws-2025.txt`).

### Preprocess and vectorize

```bash
# Parse documents into structured JSON chunks
python run_preprocessing.py

# Embed and store chunks in ChromaDB
python vectorization.py
```

### Run the app

```bash
streamlit run app.py
```

---

## Document Types Supported

| Type | Subtype | Detection |
|---|---|---|
| Meeting Minutes | `bod` | Filename or content contains "bod" / "board" |
| Meeting Minutes | `officer` | Filename or content contains "officer" / "executive" |
| Policy | — | Filename or content contains "policy" |
| Bylaws | — | Filename or content contains "bylaws" / "by-laws" |

---

## Key Features

- **HyDE (Hypothetical Document Embeddings):** For ambiguous queries, the LLM generates a plausible answer first, which is then used to improve retrieval.
- **Role-aware chunking:** Attendance sections are parsed to associate content with specific officers.
- **Score thresholding:** Returns a fallback message when no sufficiently relevant chunk is found (cosine distance > 0.44).
- **Meeting index:** Resolves relative references like "last board meeting" to a specific ISO date.

---

## Running Tests

```bash
pytest rag_test.py -v
```

Tests cover the preprocessor, vectorization, retrieval filters, and LLM prompt construction.
