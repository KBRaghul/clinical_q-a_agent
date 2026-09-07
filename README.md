# Clinical Q&A Agent

Agentic medical Q&A system using LangGraph + LangChain, running a multi-step RAG
pipeline over PubMed abstracts, with relevance grading and a web-search fallback
via a PubMed MCP server. Served through a FastAPI streaming endpoint.

## Pipeline

`retrieve` → `grade_relevance` → (enough good context? → `generate` : `web_search_fallback` (PubMed MCP) → `generate`)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in GROQ_API_KEY, NCBI_EMAIL, etc.
```

## Layout

- `src/ingest.py` — pulls PubMed abstracts, chunks, embeds into ChromaDB
- `src/retriever.py` — Chroma similarity search
- `src/grader.py` — LLM relevance grading of retrieved chunks
- `src/agent.py` — LangGraph graph (nodes + conditional edges)
- `api/main.py` — FastAPI streaming endpoint
- `data/` — local corpus / vector store (gitignored)
