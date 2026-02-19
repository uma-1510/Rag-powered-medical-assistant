# 🏥 RAG-Powered Medical Assistant

A Retrieval-Augmented Generation (RAG) based medical question-answering system designed to provide grounded, citation-based responses using curated medical knowledge.

This project focuses on **reducing hallucinations**, **optimizing LLM usage**, and **improving answer reliability** by combining semantic retrieval with controlled language generation.

---

## Overview

Traditional LLM applications often rely on large context windows, which can lead to:

* High token costs
* Attention dilution
* Increased hallucinations
* Poor reasoning over long inputs

This project solves these problems by using **RAG as the primary intelligence layer**, allowing the LLM to focus only on synthesis rather than search.

---

## Architecture

The system follows a modular pipeline where each component has a clearly defined responsibility.

### 1️⃣ Data Ingestion

* Medical Q&A data sourced from the **MedQuAD dataset**
* Raw documents are parsed and normalized

### 2️⃣ Data Cleaning & Processing

* Removal of noise and irrelevant metadata
* Structured formatting for downstream retrieval

### 3️⃣ Chunking Strategy

* Documents split into semantically meaningful chunks
* Designed to preserve medical context rather than fixed token splits

### 4️⃣ Embedding Layer

* Sentence Transformers generate vector embeddings
* Embeddings stored in a **FAISS vector database**

### 5️⃣ Retrieval Layer (Core of the System)

* Semantic similarity search
* RAG Fusion query expansion
* Top-K relevant chunks retrieved

### 6️⃣ Reranking

* Cross-encoder reranker improves relevance precision
* Reduces retrieval noise before LLM invocation

### 7️⃣ Prompt Construction

* Retrieved context injected into structured prompts
* Instructions enforce grounded and citation-based answers

### 8️⃣ Generation Layer

* LLM synthesizes responses from retrieved context
* Outputs summarized medical guidance with references

---

## Tech Stack

| Component  | Technology            |
| ---------- | --------------------- |
| Backend    | Flask                 |
| Retrieval  | FAISS                 |
| Embeddings | Sentence Transformers |
| Reranking  | Cross Encoder         |
| LLM        | Gemini                |
| Language   | Python                |

---

## System Flow

User Query → Query Expansion → Vector Search → Top-K Retrieval → Reranking → Prompt Construction → LLM Generation → Final Answer with Citations

---

## Key Features

* Retrieval-grounded responses
* Reduced hallucinations
* Token-efficient LLM usage
* Source-aware answers
* Modular and extensible architecture

---

## Project Structure

```
project/
│
├── app.py                 # Flask application entry point
├── ingestion/             # Data loading and preprocessing
├── embeddings/            # Embedding generation logic
├── retrieval/             # FAISS search and query logic
├── reranker/              # Cross-encoder ranking
├── llm/                   # Prompting and generation
├── utils/                 # Helper utilities
└── data/                  # Processed datasets
```

---

## Getting Started

### 1. Clone Repository

```
git clone https://github.com/uma-1510/Rag-powered-medical-assistant.git
cd Rag-powered-medical-assistant
```

### 2. Create Virtual Environment

```
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Build Vector Database

Run the ingestion and embedding pipeline to create FAISS indexes.

### 5. Start Application

```
python app.py
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not a substitute for professional medical advice.

---

## ⭐ Acknowledgements

* MedQuAD Dataset
* FAISS by Meta
* Sentence Transformers
* Open-source AI comm
