"""
run_demo.py
Shows the full required flow: 1 question -> 5 similar questions, EACH with
its own answer -> reranked -> best answer highlighted. Works identically for
any number of users, all sharing the same document.
"""

from rag_pipeline import get_or_create_index, handle_user_query

INDEX_NAME = "rag-multiuser-demo"


def print_result(user_id: str, result: dict):
    print(f"\n{'='*70}")
    print(f"USER: {user_id}")
    print(f"ORIGINAL QUESTION: {result['question']}")
    print(f"{'='*70}")

    for i, qa in enumerate(result["qa_pairs"], 1):
        marker = "  <-- BEST" if qa["answer"] == result["best_answer"] else ""
        print(f"\n[{i}] Similar question: {qa['question']}")
        print(f"    Answer: {qa['answer']}")
        print(f"    Rerank score: {qa['rerank_score']:.4f} | Faithfulness: {qa['faithfulness']:.3f}{marker}")

    print(f"\n>>> BEST ANSWER: {result['best_answer']}\n")


if __name__ == "__main__":
    index = get_or_create_index(INDEX_NAME)

    # Dynamic: any number of users, any questions
    incoming_questions = [
        ("user_1", "Which rocket is the most powerful?"),
        ("user_2", "How do astronauts survive re-entry?"),
        ("user_3", "What is a gravity assist maneuver?"),
    ]

    for user_id, question in incoming_questions:
        result = handle_user_query(index, user_id, question)
        print_result(user_id, result)
