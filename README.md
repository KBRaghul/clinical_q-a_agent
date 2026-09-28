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

- `GROQ_API_KEY` — from console.groq.com, used for both the relevance grader and answer generation.
- `NCBI_EMAIL` — required by NCBI's usage policy for any Entrez request. `NCBI_API_KEY` is optional but raises the rate limit.

## Running it

```bash
# 1. Build the local corpus (only needed once, or when you want to refresh it)
python src/ingest.py

# 2. Try the agent directly from the command line
python src/agent.py

# 3. Or serve it over HTTP with streaming
uvicorn api.main:app --reload --app-dir .
curl -N -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are first-line treatments for heart failure with reduced ejection fraction?"}'
```

The PubMed MCP server (`mcp_server/pubmed_server.py`) doesn't need to be started
separately — the agent spawns it over stdio on demand, only when local retrieval
is graded insufficient.

## Layout

- `src/ingest.py` — pulls PubMed abstracts, chunks, embeds into ChromaDB
- `src/retriever.py` — Chroma similarity search
- `src/grader.py` — LLM relevance grading of retrieved chunks
- `src/agent.py` — LangGraph graph (retrieve → grade → generate/fallback)
- `src/mcp_client.py` — spawns the PubMed MCP server and calls its search tool
- `mcp_server/pubmed_server.py` — MCP server exposing live PubMed search as a tool
- `api/main.py` — FastAPI streaming endpoint (`POST /ask`)
- `data/` — local corpus / vector store (gitignored)
