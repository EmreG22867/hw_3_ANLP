from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import ollama

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

# Each entry is (chunk text, embedding vector)
VECTOR_DB: list[tuple[str, list[float]]] = []


def load_dataset(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def embed_text(text: str) -> list[float]:
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    embeddings = response.get("embeddings")
    if not embeddings:
        raise RuntimeError("Ollama returned no embedding.")
    return list(embeddings[0])


def add_chunk_to_database(chunk: str) -> None:
    VECTOR_DB.append((chunk, embed_text(chunk)))


def build_vector_database(dataset: Iterable[str]) -> None:
    VECTOR_DB.clear()
    for chunk in dataset:
        add_chunk_to_database(chunk)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector lengths differ: {len(a)} != {len(b)}")
    if not a:
        raise ValueError("Cosine similarity is undefined for empty vectors.")

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")
    return dot_product / (norm_a * norm_b)


def run_small_cosine_tests() -> None:
    assert math.isclose(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)


def retrieve(query: str, top_n: int = 3) -> list[tuple[str, float]]:
    """return top-N chunks sorted by descending cosine similarity"""
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    if not VECTOR_DB:
        raise RuntimeError("The vector database is empty. Build it before retrieval.")

    query_embedding = embed_text(query)
    similarities = [
        (chunk, cosine_similarity(query_embedding, embedding))
        for chunk, embedding in VECTOR_DB
    ]
    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_n]


def build_grounded_prompt(retrieved: list[tuple[str, float]]) -> str:
    """Construct a prompt that limits the answer to retrieved evidence."""
    context = "\n".join(f"- {chunk}" for chunk, _score in retrieved)
    return f"""You are a grounded question-answering assistant.

    Use only the context below to answer the user's question.
If the context does not contain enough evidence, say that the answer is not in the knowledge base.
When records conflict, prefer a clearly dated newer record and explain the update briefly.

Context:
{context}
"""


def generate_answer(query: str, system_prompt: str) -> None:
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        stream=True,
    )
    print("\nAnswer:")
    for response_chunk in stream:
        print(response_chunk["message"]["content"], end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base", default="cat-facts.txt")
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    run_small_cosine_tests()
    dataset = load_dataset(args.knowledge_base)
    if not dataset:
        raise ValueError(f"No non-empty chunks found in {args.knowledge_base!r}.")

    print(f"Loaded {len(dataset)} chunks from {args.knowledge_base}")
    print(f"First two chunks: {dataset[:2]}")
    print("Building vector database ...")
    build_vector_database(dataset)
    print(f"Stored {len(VECTOR_DB)} embeddings.")

    query = input("Ask a question: ").strip()
    if not query:
        raise ValueError("The query must not be empty.")

    retrieved = retrieve(query, top_n=args.top_n)
    print("\nRetrieved knowledge:")
    for chunk, score in retrieved:
        print(f"- ({score:.3f}) {chunk}")

    generate_answer(query, build_grounded_prompt(retrieved))


if __name__ == "__main__":
    main()
