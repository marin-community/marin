# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure pinned Harrier embedding throughput on one GPU or TPU worker."""

import argparse
import json
import os
import time
from pathlib import Path

import jax
import numpy as np
from marin.processing.embeddings.harrier import HARRIER_DIM, HARRIER_MAX_TOKENS, HARRIER_REPO, HARRIER_REVISION
from marin.processing.embeddings.harrier_jax import HarrierJaxConfig, load_harrier_jax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("gpu", "tpu"))
    parser.add_argument("--documents", type=int, default=128)
    parser.add_argument("--words-per-document", type=int, default=256)
    args = parser.parse_args()
    if args.documents <= 0 or args.words_per_document <= 0:
        raise ValueError("Documents and words per document must be positive")

    text = " ".join(["harrier"] * args.words_per_document)
    texts = [text] * args.documents
    config = HarrierJaxConfig(
        model=HARRIER_REPO,
        revision=HARRIER_REVISION,
        tokenizer_revision=HARRIER_REVISION,
        dimensions=HARRIER_DIM,
        max_tokens=HARRIER_MAX_TOKENS,
        compute_dtype="bfloat16",
        platform=args.platform,
    )
    started = time.perf_counter()
    embedder = load_harrier_jax(config)
    load_seconds = time.perf_counter() - started
    token_count = len(embedder.tokenizer(text)["input_ids"])
    if token_count > config.max_tokens:
        raise ValueError(f"Benchmark text has {token_count} tokens, above the configured maximum")
    embedder.embed(texts)
    started = time.perf_counter()
    vectors = embedder.embed(texts)
    seconds = time.perf_counter() - started
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or not np.allclose(norms, 1, atol=1e-3):
        raise ValueError("Harrier produced nonfinite or unnormalized vectors")
    result = {
        "platform": args.platform,
        "devices": len(jax.local_devices()),
        "documents": args.documents,
        "tokens_per_document": token_count,
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(seconds, 3),
        "documents_per_second": round(args.documents / seconds, 3),
        "tokens_per_second": round(args.documents * token_count / seconds, 3),
    }
    result_json = json.dumps(result, sort_keys=True)
    print(result_json)
    if output_dir := os.environ.get("IRIS_OUTPUT_DIR"):
        Path(output_dir, "benchmark.json").write_text(result_json + "\n")


if __name__ == "__main__":
    main()
