RAG-Powered Medical Assistant

A Retrieval-Augmented Generation (RAG) based medical question-answering system that combines semantic search, query fusion, reranking, and LLM reasoning to produce accurate, evidence-grounded medical responses.

This project demonstrates how modern AI systems move beyond pure LLM prompting by integrating knowledge retrieval pipelines to reduce hallucinations and improve factual reliability.

Overview

Large Language Models are powerful but unreliable when answering domain-specific questions without grounding.

This system solves that problem using a RAG pipeline:

Retrieve medically relevant information from a curated dataset.

Rank and refine retrieved evidence.

Generate answers strictly based on retrieved context.

The assistant answers medical questions through a web interface while showing supporting sources for transparency.
