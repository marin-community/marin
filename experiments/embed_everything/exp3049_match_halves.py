# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce Test Case 1 from the Luxical blog post: matching document halves.

Methodology (from https://www.datologyai.com/blog/introducing-luxical-embeddings):
1. Sample N documents from FineWeb.
2. Split each document at the midpoint into two halves → 2N half-documents.
3. Embed all halves with each model.
4. For each half, rank all other halves by cosine similarity.
5. Measure how often the matching half appears in the top-k nearest neighbors
   at various retrieval windows (top-1, top-0.01%, top-0.1%, top-1%).
6. Compare retrieval accuracy and throughput across models.

Usage:
    # Run locally (small sample for testing):
    uv run python experiments/embed_everything/exp3049_match_halves.py --n_docs 100

    # Full reproduction:
    uv run python experiments/embed_everything/exp3049_match_halves.py --n_docs 50000

    # With StepSpec DAG:
    uv run python experiments/embed_everything/exp3049_match_halves.py --use_steps
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "DatologyAI/luxical-one",
    "sentence-transformers/all-MiniLM-L6-v2",
]

RETRIEVAL_WINDOWS = [1, 0.01, 0.1, 1.0]
"""Retrieval windows: top-1 (absolute), then top-x% of the corpus."""


@dataclass
class HalfDocument:
    doc_id: str
    half: str  # "a" or "b"
    text: str


@dataclass
class MatchHalvesResult:
    """Results for a single model on the match-halves benchmark."""

    model_name: str
    n_docs: int
    n_halves: int
    embedding_dim: int
    embed_time_seconds: float
    throughput_docs_per_sec: float
    retrieval_accuracy: dict[str, float]
    """Maps window label (e.g. "top-1", "top-0.01%") to accuracy (fraction of correct matches)."""


def sample_fineweb_documents(n_docs: int, seed: int = 42, min_chars: int = 200) -> list[dict]:
    """Stream n_docs documents from FineWeb, filtering for minimum length.

    Args:
        n_docs: Number of documents to sample.
        seed: Random seed for shuffling.
        min_chars: Minimum character length to include a document.

    Returns:
        List of dicts with "id" and "text" fields.
    """
    import datasets

    datasets.disable_caching()
    logger.info(f"Streaming {n_docs} documents from FineWeb (min_chars={min_chars})")

    ds = datasets.load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    docs = []
    for row in ds:
        text = row.get("text", "")
        if len(text) < min_chars:
            continue
        docs.append({"id": row.get("id", f"doc-{len(docs)}"), "text": text})
        if len(docs) >= n_docs:
            break

    logger.info(f"Sampled {len(docs)} documents from FineWeb")
    return docs


def split_into_halves(docs: list[dict]) -> list[HalfDocument]:
    """Split each document at the character midpoint into two halves.

    Returns a list of 2*len(docs) HalfDocument objects. The two halves of
    document i have doc_id=docs[i]["id"] and half="a"/"b".
    """
    halves = []
    for doc in docs:
        text = doc["text"]
        mid = len(text) // 2
        halves.append(HalfDocument(doc_id=doc["id"], half="a", text=text[:mid]))
        halves.append(HalfDocument(doc_id=doc["id"], half="b", text=text[mid:]))
    return halves


def embed_halves(
    halves: list[HalfDocument],
    model_name: str,
    batch_size: int = 64,
) -> tuple[np.ndarray, float]:
    """Embed all half-documents and return (embeddings, elapsed_seconds).

    Args:
        halves: List of HalfDocument objects.
        model_name: HuggingFace model name for sentence-transformers.
        batch_size: Encoding batch size.

    Returns:
        Tuple of (embeddings array [N, D], time in seconds).
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading model {model_name}")
    model = SentenceTransformer(model_name)

    texts = [h.text for h in halves]
    logger.info(f"Encoding {len(texts)} half-documents with {model_name}")

    start = time.monotonic()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    elapsed = time.monotonic() - start

    return np.array(embeddings, dtype=np.float32), elapsed


def evaluate_retrieval(
    embeddings: np.ndarray,
    halves: list[HalfDocument],
    windows: list[float] = RETRIEVAL_WINDOWS,
) -> dict[str, float]:
    """Evaluate retrieval accuracy: how often the matching half is in the top-k.

    For each half-document, we rank all *other* halves by cosine similarity
    (embeddings are assumed to be L2-normalized, so dot product = cosine).
    We check whether the matching half (same doc_id, opposite half label)
    appears within the top-k nearest neighbors at each retrieval window.

    Args:
        embeddings: L2-normalized embeddings, shape [2N, D].
        halves: Corresponding HalfDocument objects.
        windows: List of retrieval windows. Values >= 1 are treated as absolute
            counts (e.g. 1 = top-1). Values < 1 are treated as percentages of
            the corpus (e.g. 0.01 = top 0.01%).

    Returns:
        Dict mapping window label to accuracy (fraction of halves whose match
        was found within the window).
    """
    n = len(halves)
    assert embeddings.shape[0] == n

    # Build ground-truth match index: for each half i, match_idx[i] is the
    # index of its matching half (same doc_id, opposite half label).
    doc_id_to_indices: dict[str, list[int]] = {}
    for i, h in enumerate(halves):
        doc_id_to_indices.setdefault(h.doc_id, []).append(i)

    match_idx = np.full(n, -1, dtype=np.int64)
    for indices in doc_id_to_indices.values():
        assert len(indices) == 2, f"Expected exactly 2 halves per doc, got {len(indices)}"
        match_idx[indices[0]] = indices[1]
        match_idx[indices[1]] = indices[0]

    assert np.all(match_idx >= 0), "Some halves have no match"

    # Compute cosine similarity matrix (dot product of normalized vectors)
    # Process in chunks to avoid OOM on large corpora
    chunk_size = 1000
    results: dict[str, int] = {_window_label(w): 0 for w in windows}

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # [chunk_size, D] @ [D, N] → [chunk_size, N]
        sims = embeddings[start:end] @ embeddings.T

        for i_local in range(end - start):
            i_global = start + i_local
            row = sims[i_local].copy()
            # Exclude self-similarity
            row[i_global] = -np.inf

            # Argsort descending
            ranked = np.argsort(-row)
            target = match_idx[i_global]
            rank_of_target = int(np.where(ranked == target)[0][0])

            for w in windows:
                k = _window_to_k(w, n - 1)  # n-1 because we exclude self
                if rank_of_target < k:
                    results[_window_label(w)] += 1

    accuracy = {label: count / n for label, count in results.items()}
    return accuracy


def _window_to_k(window: float, corpus_size: int) -> int:
    """Convert a retrieval window to an absolute k value."""
    if window >= 1:
        return int(window)
    return max(1, int(corpus_size * window / 100))


def _window_label(window: float) -> str:
    if window >= 1:
        return f"top-{int(window)}"
    return f"top-{window}%"


def run_match_halves(
    output_path: str,
    n_docs: int = 50_000,
    models: list[str] | None = None,
    batch_size: int = 64,
    seed: int = 42,
) -> list[MatchHalvesResult]:
    """Run the full match-halves benchmark.

    Args:
        output_path: Directory to write results.
        n_docs: Number of FineWeb documents to sample.
        models: List of HuggingFace model names. Defaults to DEFAULT_MODELS.
        batch_size: Encoding batch size.
        seed: Random seed for document sampling.

    Returns:
        List of MatchHalvesResult, one per model.
    """
    if models is None:
        models = list(DEFAULT_MODELS)

    os.makedirs(output_path, exist_ok=True)

    # Step 1: Sample documents
    docs = sample_fineweb_documents(n_docs=n_docs, seed=seed)

    # Step 2: Split into halves
    halves = split_into_halves(docs)
    logger.info(f"Created {len(halves)} half-documents from {len(docs)} documents")

    # Save halves for reproducibility
    halves_file = os.path.join(output_path, "halves.jsonl")
    with open(halves_file, "w") as f:
        for h in halves:
            f.write(json.dumps(asdict(h)) + "\n")

    all_results = []

    for model_name in models:
        logger.info(f"\n{'='*60}\nEvaluating {model_name}\n{'='*60}")

        # Step 3: Embed
        embeddings, elapsed = embed_halves(halves, model_name, batch_size=batch_size)
        throughput = len(halves) / elapsed

        # Step 4: Evaluate retrieval
        accuracy = evaluate_retrieval(embeddings, halves)

        result = MatchHalvesResult(
            model_name=model_name,
            n_docs=len(docs),
            n_halves=len(halves),
            embedding_dim=embeddings.shape[1],
            embed_time_seconds=round(elapsed, 2),
            throughput_docs_per_sec=round(throughput, 1),
            retrieval_accuracy=accuracy,
        )
        all_results.append(result)

        logger.info(f"Model: {model_name}")
        logger.info(f"  Throughput: {throughput:.1f} docs/sec ({elapsed:.2f}s for {len(halves)} halves)")
        for label, acc in accuracy.items():
            logger.info(f"  {label}: {acc:.4f} ({acc*100:.2f}%)")

    # Save results
    results_file = os.path.join(output_path, "results.json")
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    return all_results


def build_steps(
    n_docs: int = 50_000,
    models: list[str] | None = None,
    batch_size: int = 64,
    seed: int = 42,
) -> list:
    """Build StepSpec DAG for the match-halves benchmark."""
    from marin.execution.step_spec import StepSpec

    if models is None:
        models = list(DEFAULT_MODELS)

    return [
        StepSpec(
            name="embed_everything/match_halves",
            hash_attrs={"n_docs": n_docs, "models": models, "seed": seed},
            fn=lambda output_path: run_match_halves(
                output_path=output_path,
                n_docs=n_docs,
                models=models,
                batch_size=batch_size,
                seed=seed,
            ),
        )
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Reproduce Luxical blog Test Case 1: match document halves")
    parser.add_argument("--n_docs", type=int, default=50_000, help="Number of FineWeb documents to sample")
    parser.add_argument("--models", nargs="+", default=None, help="Models to evaluate (HuggingFace names)")
    parser.add_argument("--batch_size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_path", type=str, default="output/match_halves", help="Output directory")
    parser.add_argument("--use_steps", action="store_true", help="Run via StepSpec DAG instead of directly")
    args = parser.parse_args()

    if args.use_steps:
        from marin.execution.step_runner import StepRunner

        steps = build_steps(n_docs=args.n_docs, models=args.models, batch_size=args.batch_size, seed=args.seed)
        StepRunner().run(steps)
    else:
        run_match_halves(
            output_path=args.output_path,
            n_docs=args.n_docs,
            models=args.models,
            batch_size=args.batch_size,
            seed=args.seed,
        )
