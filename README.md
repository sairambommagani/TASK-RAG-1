# Multi-User RAG Pipeline — Space Exploration

A production-hardened RAG pipeline where multiple users independently query
the same shared document. Each user gets an isolated session; asking a
question generates 5 unique similar questions, retrieves in parallel, fuses
rankings, reranks, and returns one grounded answer.

## What this demonstrates

| Requirement | How it's implemented |
|---|---|
| Vanilla RAG | Embed -> store in Pinecone -> retrieve -> generate, grounded in real document content |
| Multi-query generation | 5 unique reworded questions (regex-parsed, dedup with retry) |
| Pinecone multi-tenancy | One shared namespace + metadata-filtered access control (see Improvement 1) |
| Reranking | Cross-encoder re-scores fused candidates before the final answer |
| Dynamic multi-user support | `handle_user_query(index, user_id, question)` works for any user, no hardcoding |

## Production-hardening improvements (this version)

**1. Shared namespace + access control, not vector duplication**
Documents are ingested ONCE into a `shared` namespace, not copied into every
user's own namespace. Access is controlled via Pinecone metadata filtering
(`is_public` / `allowed_users`) at query time — this scales to many users
without duplicating storage, and supports future private-document use cases
by passing `allowed_users=[...]` to `ingest_shared_document()`.

**2. Regex-based multi-query parsing**
`generate_query_variations()` strips numbering and markdown bullets
(`1.`, `1)`, `-`, `*`, `•`) via regex, and filters out non-question
conversational noise (any line without a `?`). Robust to small LLMs that
don't follow the requested format exactly.

**3. Semantic (cosine similarity) evaluation**
`evaluate.py` compares generated answers to ground-truth answers using
embedding cosine similarity, not brittle substring matching — a correct
answer phrased differently (e.g. "one and a half million km" vs
"1.5 million km") is still scored correctly. For deeper rigor
(Faithfulness, Answer Relevance, Context Precision), integrating Ragas or
TruLens is a natural next step, flagged here rather than implemented, to
avoid an extra heavy dependency in a demo script.

**4. Parallel retrieval + Reciprocal Rank Fusion (RRF)**
Retrieval across the 5 query variations runs concurrently
(`ThreadPoolExecutor`) instead of a sequential loop. Results are merged with
RRF (`score = sum(1 / (k + rank))` across all 5 ranked lists) instead of
naive concatenation — chunks retrieved by multiple query variations
correctly float to the top before reranking.

**5. FastAPI service layer**
`api.py` exposes `POST /query` with pydantic request validation, so this can
run as a real service instead of only from a notebook/script. Streaming
responses are noted as a follow-up (requires restructuring the LLM call into
a generator) rather than implemented here.

## Setup (Google Colab — for the notebook/script side)

1. Upload `corpus.txt`, `rag_pipeline.py`, `run_demo.py`, `evaluate.py` to Colab.
2. Add your Pinecone API key to Colab Secrets, named `PINECONE_API_KEY`.
3. Run, each in its own cell:

```python
# Cell 1
from google.colab import userdata
import os
os.environ['PINECONE_API_KEY'] = userdata.get('PINECONE_API_KEY')

# Cell 2
!pip install -q langchain-text-splitters sentence-transformers transformers torch pinecone

# Cell 3
import sys
sys.path.append('/content')

# Cell 4
from rag_pipeline import get_or_create_index, handle_user_query
```

## Usage

```python
index = get_or_create_index("rag-multiuser-demo")

result = handle_user_query(index, "user_1", "Which rocket is the most powerful?")
print(result["variations"])
print(result["answer"])
```

## Running the API locally (not in Colab)

```bash
pip install fastapi uvicorn
uvicorn api:app --reload
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_1", "question": "Which rocket is the most powerful?"}'
```

## Running the evaluation

```python
from evaluate import run_evaluation
run_evaluation(index)
```

## Design decisions

- **Embedding model:** `all-MiniLM-L6-v2` — small, fast, local, no API cost.
- **LLM:** `Qwen2.5-1.5B-Instruct` — small enough for Colab's free tier while
  still following instructions reliably.
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — standard, lightweight.
- **Chunking:** 300 characters, 50-character overlap.
- **top_k:** 5 per query variation (25 candidates across 5 queries before
  RRF fusion), reranked down to the best 5 before answering.

## Known limitations / next steps

- Corpus is a small fixed set of 15 passages.
- Evaluation uses cosine similarity, not a full Ragas/TruLens pipeline yet.
- Streaming responses not implemented in the API layer.
- No auth/rate-limiting on the API endpoint yet (would be needed before any real deployment).
