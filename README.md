# 🏗 Multi-Agent GitHub Repository Analysis System

A production-grade, multi-agent backend system that analyzes GitHub repositories using Grok API (xAI), tree-sitter parsing, and RAG-based question answering.

## Architecture Overview

```
User → FastAPI → Orchestrator
                    │
                    ├─ Phase 1: Clone & Scan (GitPython)
                    ├─ Phase 2-3: Batch Agent Processing (Worker Pool + Grok API)
                    ├─ Phase 4: Combine Reports + Dependency Graph
                    ├─ Phase 5: Build RAG Vector Store (ChromaDB)
                    └─ Phase 6: Generate Mermaid Diagrams (Grok API)
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Git
- A Grok API key from [console.x.ai](https://console.x.ai)

### 2. Installation

```bash
# Clone this project
git clone <this-repo>
cd multi-agent-repo-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install the project
pip install -e .
```

### 3. Configuration

```bash
cp .env.example .env
# Edit .env and set your XAI_API_KEY
```

### 4. Run the Server

```bash
python -m app.main
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. API Usage

**Start an analysis:**
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"repository_url": "https://github.com/pallets/flask"}'
```

Response:
```json
{
  "analysis_id": "a1b2c3d4-...",
  "status": "pending",
  "message": "Analysis started. Use GET /api/v1/report/a1b2c3d4-... to check results."
}
```

**Check status:**
```bash
curl http://localhost:8000/api/v1/status/{analysis_id}
```

**Get report:**
```bash
curl http://localhost:8000/api/v1/report/{analysis_id}
```

**Ask questions (RAG):**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "a1b2c3d4-...",
    "question": "How does the routing system work?"
  }'
```

## How Batching Works

The system does **NOT** create one agent per file. Instead:

1. All files are split into **batches** of configurable size (default: 10).
2. Each batch is processed concurrently using `asyncio.gather`.
3. A global `asyncio.Semaphore` (in `GrokClient`) limits concurrent API calls (default: 5).
4. After one batch completes, the next begins.

**Example**: 100 files, batch_size=10, max_concurrent=5:
- 10 batches of 10 files each
- Within each batch, at most 5 files hit the API simultaneously
- Each batch completes before the next starts

## How Rate Limiting Works

Three layers of protection:

1. **Semaphore**: `asyncio.Semaphore(max_concurrent_llm_calls)` limits parallel API calls.
2. **Retry with backoff**: `tenacity` retries on 429/timeout/connection errors with exponential backoff (2s → 4s → 8s → ... up to 60s).
3. **Batch throttling**: Processing files in rounds naturally prevents burst traffic.
4. **Token tracking**: `TokenUsageTracker` logs all token consumption for monitoring.

## Project Structure

```
app/
├── api/              # FastAPI routes and dependencies
├── core/             # Logging, exceptions
├── agents/           # Grok client + LLM agents (file, architecture, mermaid, RAG)
├── parsers/          # Tree-sitter code parser
├── rag/              # ChromaDB vector store
├── graph/            # Dependency graph builder
├── schemas/          # Pydantic models (all structured)
├── services/         # Business logic (scanner, batch processor, orchestrator, etc.)
└── main.py           # FastAPI app entry point
```

## LLM Prompts Used

### File Analysis Agent
Analyzes individual files with: raw content + tree-sitter structure → structured JSON with functions, classes, imports, dependencies.

### Architecture Summary Agent
Takes all file analysis results → produces: overview, key components, design patterns, entry points, tech stack.

### Mermaid Generator Agent
Takes dependency graph JSON → produces: file flow diagram, function call flow diagram, entry point flow diagram (all in Mermaid syntax).

### RAG Answer Agent
Takes user question + vector search results + dependency context → produces a grounded answer referencing actual files/functions.

## Design Decisions

- **Pydantic everywhere**: All inputs/outputs are validated Pydantic models.
- **No raw strings from LLM**: All LLM outputs are parsed into structured models.
- **Stateless agents**: All state lives in the store/vector DB, agents are pure functions.
- **Caching**: Repos are cached on disk; re-analyzing pulls latest.
- **Scalability**: Replace file store with PostgreSQL, add Redis task queue for horizontal scaling.
- **Low temperature (0.2)**: Ensures deterministic, factual outputs from Grok.