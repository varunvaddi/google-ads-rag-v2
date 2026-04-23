# Google Ads Policy RAG — v2

Production-grade RAG system for Google Ads policy compliance.

## What's new in v2
- Ollama (llama3.2) replaces Gemini — unlimited, free, local
- LangGraph orchestration replaces sequential pipeline
- Full eval suite: Recall@5, MRR, P95 latency, RAGAS metrics

## Stack
- Embeddings: BGE-large-en-v1.5
- Vector DB: FAISS + BM25 (hybrid)
- Reranker: BGE-reranker-large
- LLM: Ollama llama3.2 (local)
- Orchestration: LangGraph
- Evaluation: RAGAS

## Status
🚧 In progress — building step by step
