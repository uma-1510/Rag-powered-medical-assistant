# Rag-powered-medical-assistant
A powerful, retrieval-augmented generation (RAG) medical assistant using advanced question fusion, FAISS-based dense retrieval, and Gemini LLM for high-quality answers. Built with Flask backend and a modern HTML+JS frontend for easy interaction.

Features
RAG Fusion Search: Reformulates your medical question into multiple perspectives using an LLM, boosting coverage by performing multi-query retrieval (“query fusion”).
Dense Retrieval + Cross-Encoder Reranking: Finds highly relevant medical Q&A passages from the MedQuAD dataset using FAISS and re-ranks them for context quality.
Gemini LLM Answer Generation: Produces accurate, concise answers using the Gemini API, grounded strictly in the retrieved evidence.
Sources Transparency: Every answer includes top source Q&As and URLs.
Easy-to-Use Web UI: Ask any medical question through a responsive web frontend (no CLI lines needed).
Flask REST API: For integration with other apps or clients

How Does It Work?
User asks a question in the web interface.

RAG-Fusion:
Generates several alternative forms of the question (paraphrases/different angles).
Retrieves relevant documents for each query.
Fuses them (Reciprocal Rank Fusion) into a robust, comprehensive candidate list.
Evidence reranking: Cross-encoder scores/reorders retrieved passages.
LLM Generation: Gemini LLM synthesizes a direct answer using only the top evidence, ensuring factual accuracy.
Results: Shows the answer and references the supporting sources.

Technologies
Python, Flask (REST API backend)
HTML/CSS/JS (frontend)
FAISS, Sentence Transformers (for dense retrieval)
Cross-Encoder (for reranking)
Gemini API (for question rewriting & answer generation)
MedQuAD dataset (curated medical Q&A knowledge base)
