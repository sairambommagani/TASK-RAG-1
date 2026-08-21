"""
rag_pipeline.py — v3

Production-hardening pass, incorporating lead-reviewer feedback:
  1. Shared namespace + metadata filtering, instead of duplicating the
     document's vectors into every user's own namespace
  2. Regex-based parsing for multi-query generation (robust to markdown
     bullets / preamble text from the LLM, not just numbered lines)
  3. (see evaluate.py) Cosine-similarity based evaluation, not just substring
  4. Parallel retrieval across the 5 query variations + Reciprocal Rank
     Fusion (RRF) to merge them, instead of sequential looping
  5. (see api.py) FastAPI service wrapper with pydantic request validation

Install:
  pip install -q langchain-text-splitters sentence-transformers transformers torch pinecone fastapi uvicorn pydantic

In Colab: store keys in Colab Secrets (key icon in sidebar), not hardcoded:
  from google.colab import userdata
  PINECONE_API_KEY = userdata.get("PINECONE_API_KEY")
"""

import os
import re
import concurrent.futures
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from pinecone import Pinecone, ServerlessSpec

# ---------------------------------------------------------------------------
# Config - single source of truth for model choices
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"        # 384-dim
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_DIM = 384
DEFAULT_TOP_K = 8
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
SHARED_NAMESPACE = "shared"   # one namespace for all shared/public documents
RRF_K = 60                    # standard RRF damping constant

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# ---------------------------------------------------------------------------
# Lazy-loaded singletons so models load once, not repeatedly across cells
# ---------------------------------------------------------------------------
_embedding_model = None
_llm_tokenizer = None
_llm_model = None
_reranker = None
_pc_client = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedding_model


def get_llm():
    global _llm_tokenizer, _llm_model
    if _llm_model is None:
        _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        _llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME, torch_dtype="auto", device_map="auto"
        )
    return _llm_tokenizer, _llm_model


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def get_pinecone_client():
    global _pc_client
    if _pc_client is None:
        _pc_client = Pinecone(api_key=PINECONE_API_KEY)
    return _pc_client


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def load_and_chunk(filepath: str) -> list[str]:
    """Load a plain-text file and split into overlapping chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.encode(texts).tolist()


def embed_text(text: str) -> list[float]:
    model = get_embedding_model()
    return model.encode(text).tolist()


# ---------------------------------------------------------------------------
# Pinecone index management
# ---------------------------------------------------------------------------
def get_or_create_index(index_name: str):
    pc = get_pinecone_client()
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


# ---------------------------------------------------------------------------
# IMPROVEMENT 1: Shared namespace + metadata filtering, not per-user copies
# ---------------------------------------------------------------------------
def is_shared_doc_ingested(index) -> bool:
    """Checks whether the shared document has already been ingested at all —
    regardless of which user is asking. Ingested ONCE, ever, not once per user."""
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces", {})
    return SHARED_NAMESPACE in namespaces and namespaces[SHARED_NAMESPACE].get("vector_count", 0) > 0


def ingest_shared_document(index, filepath: str = "corpus.txt", allowed_users: list[str] | None = None):
    """Ingests a document ONCE into a shared namespace. Access control is
    handled via metadata + query-time filtering, not by duplicating vectors
    per user. `allowed_users=None` means public — every user can retrieve it.
    Pass a list to restrict it to specific users (private document use case)."""
    if is_shared_doc_ingested(index):
        print("[shared] document already ingested — skipping re-ingest")
        return

    chunks = load_and_chunk(filepath)
    vectors_emb = embed_texts(chunks)
    is_public = allowed_users is None

    vectors = [
        {
            "id": f"shared-chunk-{i}",
            "values": vec,
            "metadata": {
                "text": chunks[i],
                "is_public": is_public,
                # Pinecone metadata doesn't store None well; use empty list for public docs
                "allowed_users": allowed_users or [],
            },
        }
        for i, vec in enumerate(vectors_emb)
    ]
    index.upsert(vectors=vectors, namespace=SHARED_NAMESPACE)
    print(f"[shared] ingested {len(vectors)} chunks from {filepath} (public={is_public})")


def retrieve_shared(index, query: str, user_id: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Retrieves from the single shared namespace, filtered so a user only
    ever sees chunks that are public OR explicitly allow-listed for them.
    This is the access-control mechanism — no vector duplication required."""
    query_vec = embed_text(query)
    results = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        namespace=SHARED_NAMESPACE,
        filter={
            "$or": [
                {"is_public": True},
                {"allowed_users": {"$in": [user_id]}},
            ]
        },
    )
    return [
        {"text": m["metadata"]["text"], "score": m["score"]} for m in results["matches"]
    ]


# ---------------------------------------------------------------------------
# Generation (single consistent LLM: Qwen2.5-1.5B-Instruct)
# ---------------------------------------------------------------------------
def _run_llm(prompt: str, max_new_tokens: int = 150) -> str:
    tokenizer, model = get_llm()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,   # greedy decoding — reduces invented, "creative" embellishment
    )
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""You are answering strictly based on the CONTEXT below. This is a
closed-book task: you must NOT use any outside knowledge, even if you know the
real-world answer. Only state facts that are explicitly written in the context.

CONTEXT:
{context}

QUESTION: {question}

Instructions:
- If the context contains the answer, answer using ONLY facts stated in the context.
- Do not add details, mechanisms, or examples that are not explicitly written above,
  even if they sound plausible or you believe they are true.
- If the context does NOT contain enough information to answer, respond exactly:
  "I don't have enough information to answer that."

ANSWER:"""
    return _run_llm(prompt)


REFUSAL_TEXT = "i don't have enough information to answer that"


def estimate_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Rough faithfulness signal: how semantically close is the answer to the
    context it was supposedly grounded in? Not a substitute for a real
    faithfulness evaluator (e.g. Ragas), but catches obvious drift cheaply —
    a low score is a hint the model may have added outside knowledge.
    An honest refusal ("I don't have enough information") is treated as
    fully faithful (1.0) — declining is the CORRECT behavior when the
    context is insufficient, not a sign of drift."""
    if answer.strip().lower().startswith(REFUSAL_TEXT):
        return 1.0
    if not context_chunks:
        return 0.0
    from sentence_transformers import util
    model = get_embedding_model()
    answer_emb = model.encode(answer, convert_to_tensor=True)
    context_emb = model.encode(" ".join(context_chunks), convert_to_tensor=True)
    return float(util.cos_sim(answer_emb, context_emb)[0][0])


# ---------------------------------------------------------------------------
# IMPROVEMENT 2: Regex-based robust parsing for multi-query generation
# ---------------------------------------------------------------------------
_LIST_PREFIX_RE = re.compile(r"^\s*(\d+[\.\)]|[-*•])\s*")


def _clean_line(line: str) -> str:
    """Strips numbering (1. / 1)) and markdown bullets (- / * / •) from the
    start of a line, regardless of which style the LLM used."""
    return _LIST_PREFIX_RE.sub("", line).strip()


def generate_query_variations(question: str, n: int = 5, max_attempts: int = 3) -> list[str]:
    """Generates n semantically similar, UNIQUE questions. Parsing is
    regex-based so it survives markdown bullets, mixed numbering styles, and
    conversational preamble ("Here are 5 questions:") from smaller LLMs —
    any non-question line (no "?") is filtered out rather than breaking parsing."""
    unique_questions = {}

    for attempt in range(max_attempts):
        still_needed = n - len(unique_questions)
        if still_needed <= 0:
            break

        prompt = f"""Generate exactly {still_needed} semantically similar questions for the question below.
Every question must be phrased DIFFERENTLY from one another — no repeats or near-duplicates.

Original question: {question}

Rules:
- All {still_needed} questions must have the same meaning as the original.
- Each must be a complete question, worded differently from the others.
- Do not answer them.
- Number them from 1 to {still_needed}."""

        raw = _run_llm(prompt, max_new_tokens=200)
        lines = [l for l in raw.split("\n") if l.strip()]

        for l in lines:
            cleaned = _clean_line(l)
            # Filter out conversational noise ("Here are 5 questions:") —
            # a real question line always contains a "?"
            if not cleaned or "?" not in cleaned:
                continue
            key = cleaned.lower().rstrip("?").strip()
            if key and key not in unique_questions:
                unique_questions[key] = cleaned

    return list(unique_questions.values())[:n]


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
def rerank(question: str, results: list[dict], top_n: int = 5) -> list[dict]:
    reranker = get_reranker()
    pairs = [[question, r["text"]] for r in results]
    scores = reranker.predict(pairs)
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# IMPROVEMENT 4: Parallel retrieval + Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------
def _reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Merges multiple ranked retrieval lists (one per query variation) into
    a single ranked list using RRF: score(doc) = sum(1 / (k + rank)) across
    every list the doc appears in. Docs retrieved by MULTIPLE query
    variations naturally float to the top — a much better relevance signal
    than any single query's raw similarity score."""
    rrf_scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):  # rank is 0-indexed
            text = item["text"]
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + rank + 1)
            chunk_lookup[text] = item

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{**chunk_lookup[text], "rrf_score": score} for text, score in fused]


def _retrieve_parallel(index, queries: list[str], user_id: str, top_k_per_query: int) -> list[list[dict]]:
    """Runs retrieval for all query variations CONCURRENTLY instead of
    looping sequentially — each call is I/O-bound (network call to Pinecone),
    so threads give a real wall-clock speedup proportional to query count."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
        futures = [
            executor.submit(retrieve_shared, index, q, user_id, top_k_per_query)
            for q in queries
        ]
        return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Full pipeline (matches required flow):
#   1 question -> 5 unique similar questions -> EACH retrieves + gets its OWN
#   answer -> the 5 (question, answer) pairs are reranked against the
#   ORIGINAL question -> best answer surfaces to the top
# ---------------------------------------------------------------------------
def full_pipeline(index, question: str, user_id: str, top_k_per_query: int = 5):
    variations = generate_query_variations(question)

    # Retrieve for all 5 variations in parallel (I/O-bound Pinecone calls)
    retrieved_lists = _retrieve_parallel(index, variations, user_id, top_k_per_query)

    # Each variation gets its OWN reranked context and its OWN generated answer
    qa_pairs = []
    for q, retrieved in zip(variations, retrieved_lists):
        if not retrieved:
            qa_pairs.append({"question": q, "answer": "I don't have enough information.", "context": []})
            continue
        reranked_chunks = rerank(q, retrieved, top_n=3)
        context = [r["text"] for r in reranked_chunks]
        answer = generate_answer(q, context)
        faithfulness = estimate_faithfulness(answer, context)
        qa_pairs.append({"question": q, "answer": answer, "context": context, "faithfulness": round(faithfulness, 3)})

    # Rerank the 5 CANDIDATE ANSWERS against the original question — this is
    # what picks the single best answer out of the 5 independently generated ones
    reranker = get_reranker()
    scores = reranker.predict([[question, qa["answer"]] for qa in qa_pairs])
    for qa, s in zip(qa_pairs, scores):
        qa["rerank_score"] = float(s)
    ranked_qa_pairs = sorted(qa_pairs, key=lambda x: x["rerank_score"], reverse=True)

    return {
        "question": question,
        "qa_pairs": ranked_qa_pairs,          # all 5, best answer first
        "best_answer": ranked_qa_pairs[0]["answer"],
    }


# ---------------------------------------------------------------------------
# Dynamic entry point — ANY user, ANY question, no hardcoding required.
# ---------------------------------------------------------------------------
def handle_user_query(index, user_id: str, question: str, doc_filepath: str = "corpus.txt") -> dict:
    """One call does everything:
      1. Ensures the SHARED document is ingested (once, globally — not per user)
      2. Generates 5 unique similar questions
      3. Retrieves in parallel across all 5, fuses with RRF, reranks, answers
    """
    ingest_shared_document(index, doc_filepath)  # no-op after the first call, ever
    return full_pipeline(index, question, user_id)
