"""
api.py

IMPROVEMENT 5: Wraps handle_user_query() as a real HTTP service instead of
only being callable from a script/notebook — request validation via
pydantic, a POST /query endpoint, and a health check.

Run locally:
    pip install fastapi uvicorn
    uvicorn api:app --reload

Then POST to http://localhost:8000/query with JSON body:
    {"user_id": "user_1", "question": "Which rocket is the most powerful?"}

Note: this is NOT meant to run inside Colab (Colab isn't built for serving
long-running HTTP servers). Use this locally or deploy it (e.g. Render,
Railway, an EC2 box) once you're ready to move past the notebook stage.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pipeline import get_or_create_index, handle_user_query

app = FastAPI(title="Multi-User RAG API", version="1.0")

INDEX_NAME = "rag-multiuser-demo"
_index = None  # lazy-initialized on first request, not at import time


def get_index():
    global _index
    if _index is None:
        _index = get_or_create_index(INDEX_NAME)
    return _index


class QueryRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique ID for the requesting user")
    question: str = Field(..., min_length=3, description="The user's question")


class QueryResponse(BaseModel):
    question: str
    qa_pairs: list[dict]   # all 5 similar questions with their own answers, ranked
    best_answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        index = get_index()
        result = handle_user_query(index, request.user_id, request.question)
        return QueryResponse(
            question=result["question"],
            qa_pairs=result["qa_pairs"],
            best_answer=result["best_answer"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Streaming note (not implemented here) ---
# For token-by-token streaming responses, FastAPI supports StreamingResponse
# combined with a generator-based LLM call (model.generate(..., streamer=...)
# from transformers' TextIteratorStreamer). Left as a follow-up since it
# requires restructuring _run_llm() into a generator, which is a bigger change
# than the other 4 improvements — flagging it explicitly rather than
# quietly skipping it.
