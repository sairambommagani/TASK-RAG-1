"""
evaluate.py — v2

IMPROVEMENT 3: Cosine-similarity based evaluation instead of substring
matching. Substring checks fail on correct answers phrased differently
(e.g. "one and a half million km" vs "1.5 million km") — semantic
similarity catches these correctly.

For full RAG evaluation rigor beyond this script (Faithfulness, Answer
Relevance, Context Precision), integrate a framework like Ragas or TruLens —
noted as a next step in README.md rather than added here, to avoid an extra
heavy dependency in a demo script.

Usage:
    from evaluate import run_evaluation
    run_evaluation(index)
"""

from sentence_transformers import util
from rag_pipeline import handle_user_query, get_embedding_model

SIMILARITY_THRESHOLD = 0.6  # cosine similarity above this = considered correct

# Each entry: the question, and a full ground-truth answer (not just a
# keyword) to compare against semantically.
TEST_CASES = [
    {
        "question": "Which rocket is the most powerful?",
        "ground_truth": "The Saturn V rocket is the most powerful rocket ever successfully flown.",
    },
    {
        "question": "What year did humans first land on the Moon?",
        "ground_truth": "Humans first landed on the Moon in 1969, during the Apollo 11 mission.",
    },
    {
        "question": "How fast must a rocket travel to escape Earth's gravity?",
        "ground_truth": "A rocket must reach escape velocity, about 11.2 km/s, to escape Earth's gravity.",
    },
    {
        "question": "What is the Karman line?",
        "ground_truth": "The Karman line is 100 km above sea level, the recognized boundary of space.",
    },
    {
        "question": "How far is the James Webb Space Telescope from Earth?",
        "ground_truth": "The James Webb Space Telescope orbits about 1.5 million km from Earth, near the L2 point.",
    },
    {
        "question": "What altitude does the International Space Station orbit at?",
        "ground_truth": "The International Space Station orbits at roughly 400 km altitude.",
    },
    {
        "question": "What company pioneered reusable rocket boosters?",
        "ground_truth": "SpaceX pioneered reusable rocket boosters at scale with the Falcon 9.",
    },
    {
        "question": "What program aims to return humans to the Moon?",
        "ground_truth": "The Artemis program aims to return humans to the Moon.",
    },
]


def semantic_similarity(text_a: str, text_b: str) -> float:
    model = get_embedding_model()
    emb_a = model.encode(text_a, convert_to_tensor=True)
    emb_b = model.encode(text_b, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b)[0][0])


def run_evaluation(index, user_id: str = "eval_user"):
    results = []
    for case in TEST_CASES:
        result = handle_user_query(index, user_id, case["question"])
        similarity = semantic_similarity(result["best_answer"], case["ground_truth"])
        passed = similarity >= SIMILARITY_THRESHOLD
        results.append({
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "answer": result["best_answer"],
            "similarity": round(similarity, 3),
            "passed": passed,
        })

    passed_count = sum(r["passed"] for r in results)
    avg_similarity = sum(r["similarity"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {passed_count}/{len(results)} passed "
          f"(avg similarity: {avg_similarity:.3f}, threshold: {SIMILARITY_THRESHOLD})")
    print(f"{'='*60}\n")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] ({r['similarity']:.3f}) {r['question']}")
        if not r["passed"]:
            print(f"       Ground truth: {r['ground_truth']}")
            print(f"       Got:          {r['answer'][:150]}")
        print()

    return results


if __name__ == "__main__":
    from rag_pipeline import get_or_create_index
    index = get_or_create_index("rag-multiuser-demo")
    run_evaluation(index)
