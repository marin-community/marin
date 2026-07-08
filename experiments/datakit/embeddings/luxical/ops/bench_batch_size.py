# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep batch (window) size for the native Luxical-One Embedder on Iris cpu=8.

Reads ~10K texts from the cached ``nemotron_cc_v2/high_quality`` normalized
parquet, warms the embedder + numba JIT, then encodes the same pool at each
window size and reports docs/s + chars/s. The optimum picks the
``window(N)`` size for the Zephyr-based embed pipeline.

Uses the native ``luxical.embedder.Embedder`` (no sentence-transformers wrapper);
prior bench on the wrapper got 952 docs/s at batch=1024. Native is expected
to be substantially faster per the M4 macbook number in #5410 (~2700 docs/s).

Submit:

    uv run iris --cluster=marin job run --no-wait \\
        --cpu=8 --memory=16G --extra=cpu --extra=embed \\
        --enable-extra-resources \\
        --job-name "bench-batch-size-$(date +%Y%m%d-%H%M%S)" \\
        -- python -m experiments.datakit.embeddings.luxical.ops.bench_batch_size
"""

import json
import logging
import os
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rigging.filesystem import StoragePath

logger = logging.getLogger(__name__)

LUXICAL_REPO = "DatologyAI/luxical-one"
LUXICAL_WEIGHTS_FILE = "luxical_one_rc4.npz"
NORMALIZED_DIR = "gs://marin-eu-west4/normalized/nemotron_cc_v2/high_quality_b451aefe/outputs/main"
N_DOCS = 10_000
WARMUP_DOCS = 1_000
WINDOW_SIZES: tuple[int, ...] = (64, 256, 1024, 4096, 10_000)
RESULT_URI = "gs://marin-eu-west4/tmp/ttl=7d/rav/clustering-full-smoke/bench_batch_size_native.json"

# Thread caps so cpu=8 actually uses 8 (matching the production Zephyr worker config).
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(var, "8")


def _load_texts() -> list[str]:
    shard = f"{NORMALIZED_DIR}/part-00000-of-04136.parquet"
    logger.info("Loading first %d texts from %s", N_DOCS, shard)
    pf = pq.ParquetFile(shard)
    texts: list[str] = []
    for i in range(pf.num_row_groups):
        if len(texts) >= N_DOCS:
            break
        rg = pf.read_row_group(i, columns=["text"]).to_pylist()
        for row in rg:
            texts.append(row["text"])
            if len(texts) >= N_DOCS:
                break
    return texts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from luxical.embedder import Embedder  # noqa: PLC0415  # optional dep: luxical

    texts = _load_texts()
    char_counts = [len(t) for t in texts]
    total_chars = sum(char_counts)
    logger.info("Loaded %d texts (%.1f MB UTF-8)", len(texts), total_chars / 1024 / 1024)

    t0 = time.monotonic()
    npz_path = hf_hub_download(repo_id=LUXICAL_REPO, filename=LUXICAL_WEIGHTS_FILE)
    download_s = time.monotonic() - t0
    t0 = time.monotonic()
    embedder = Embedder.load(npz_path)
    load_s = time.monotonic() - t0
    logger.info("Weights downloaded in %.1fs, embedder loaded in %.1fs", download_s, load_s)

    # Warmup so numba JIT + tokenizer caches are hot for the timed runs.
    t0 = time.monotonic()
    _ = embedder(texts[:WARMUP_DOCS], progress_bars=False)
    warmup_s = time.monotonic() - t0
    logger.info("Warmup on %d docs in %.1fs", WARMUP_DOCS, warmup_s)

    results: list[dict] = []
    for w in WINDOW_SIZES:
        # Process the full N_DOCS pool in ceil(N/w) chunks of size w; sum the time.
        t0 = time.monotonic()
        for i in range(0, len(texts), w):
            chunk = texts[i : i + w]
            _ = embedder(chunk, progress_bars=False)
        encode_s = time.monotonic() - t0
        docs_per_sec = len(texts) / encode_s
        chars_per_sec = total_chars / encode_s
        results.append(
            {
                "batch_size": w,
                "n_docs": len(texts),
                "encode_s": encode_s,
                "docs_per_sec": docs_per_sec,
                "chars_per_sec": chars_per_sec,
            }
        )
        logger.info(
            "window=%5d  encode=%.2fs  %.1f docs/s  %.2f MB/s",
            w,
            encode_s,
            docs_per_sec,
            chars_per_sec / 1024 / 1024,
        )

    output = {
        "api": "native luxical.embedder.Embedder",
        "repo": LUXICAL_REPO,
        "weights": LUXICAL_WEIGHTS_FILE,
        "n_docs": len(texts),
        "total_chars": total_chars,
        "p50_chars": int(np.median(char_counts)),
        "p95_chars": int(np.quantile(char_counts, 0.95)),
        "download_s": download_s,
        "load_s": load_s,
        "warmup_s": warmup_s,
        "results": results,
    }
    StoragePath(RESULT_URI).write_text(json.dumps(output, indent=2))
    logger.info("Wrote results to %s", RESULT_URI)


if __name__ == "__main__":
    main()
