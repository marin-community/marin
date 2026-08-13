# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preflight the corpus scorer on a node before handing it the fleet.

Answers the two questions that decide whether a scoring run is real:

* **Does JAX see the accelerators?** A worker venv resolved from the repo's declared
  deps is not guaranteed to carry a CUDA jaxlib for every architecture, and a
  CPU-only jaxlib does not fail -- it runs the forward on cores perhaps fifty times
  slower and reports success. This logs the backend and the device list, and exits
  non-zero when no accelerator is attached.
* **Is the staged model the one we think it is?** Reports each artifact's size and
  the sha256 over the directory's files in name order, so a swapped or truncated
  checkpoint is caught before 14 billion documents are scored against it.

Then it runs one real forward through the deployment path (:func:`load_folded_scorer`
plus :func:`predict`) so a broken CUDA build surfaces here rather than inside the run.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --gpu 4 --extra gpu --cpu 8 --memory 32g --enable-extra-resources \\
        -- python -m experiments.datakit.scripts.score_corpus_preflight
"""

import argparse
import hashlib
import json
import logging
import time

import fsspec
import jax
import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.score_corpus import (
    DEFAULT_FOLDED_DIR,
    EMBED_DIM,
    load_folded_scorer,
)

logger = logging.getLogger(__name__)

EXPECTED_SHA256 = "e1be09c903bf7a926046bac97d40db050ca399fbb336c7541df42dc8cf6eda10"
EXPECTED_BYTES = 167_519_131
PROBE_ROWS = 256


def digest_dir(model_dir: str) -> dict:
    """Per-file sha256 plus every composite convention a caller might have meant.

    A model directory has no canonical digest, so a single number cannot be checked
    against a quoted one: it depends on whether the files were concatenated, and in
    what order, and whether names were mixed in. Report the per-file digests and all
    the usual composites, and let the match fall out.
    """
    fs = fsspec.filesystem("s3")
    paths = sorted(p for p in fs.find(model_dir.removeprefix("s3://")) if not p.endswith("/"))
    payloads = {p.rsplit("/", 1)[-1]: fs.cat(p) for p in paths}
    per_file = {name: hashlib.sha256(body).hexdigest() for name, body in payloads.items()}
    # The pinning convention: fold each file's *raw* 32-byte digest into one hash,
    # keyed by its NUL-terminated path relative to the model root, over the
    # bytewise-sorted relative paths. Path-keyed, so a rename changes the digest.
    root = model_dir.removeprefix("s3://").rstrip("/")
    rels = sorted(p.removeprefix(root).lstrip("/") for p in paths)
    rel_payload = {p.removeprefix(root).lstrip("/"): fs.cat(p) for p in paths}
    folded = hashlib.sha256()
    for rel in rels:
        folded.update(rel.encode() + b"\0" + hashlib.sha256(rel_payload[rel]).digest())
    by_name = sorted(payloads)
    by_size = sorted(payloads, key=lambda n: (len(payloads[n]), n))
    concat_name = hashlib.sha256(b"".join(payloads[n] for n in by_name)).hexdigest()
    concat_size = hashlib.sha256(b"".join(payloads[n] for n in by_size)).hexdigest()
    tree = hashlib.sha256("".join(f"{n}:{per_file[n]}\n" for n in by_name).encode()).hexdigest()
    digest_tree = hashlib.sha256("".join(per_file[n] for n in by_name).encode()).hexdigest()
    return {
        "files": {n: len(b) for n, b in payloads.items()},
        "total_bytes": sum(len(b) for b in payloads.values()),
        "per_file_sha256": per_file,
        "relative_paths": rels,
        "sha256": folded.hexdigest(),
        "candidates": {
            "relpath_nul_rawdigest": folded.hexdigest(),
            "concat_by_name": concat_name,
            "concat_by_size": concat_size,
            "name_and_digest_lines": tree,
            "concatenated_digests": digest_tree,
            **{f"file:{n}": h for n, h in per_file.items()},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_FOLDED_DIR)
    ap.add_argument(
        "--digest-only",
        action="store_true",
        help="report the staged digests and skip the device check and forward (runs on CPU)",
    )
    args = ap.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    backend = jax.default_backend() if not args.digest_only else "skipped"
    devices = [] if args.digest_only else jax.devices()
    logger.info("PREFLIGHT jax %s backend=%s devices=%s", jax.__version__, backend, devices)
    logger.info("PREFLIGHT device_kinds=%s", sorted({d.device_kind for d in devices}))

    staged = digest_dir(args.model_dir)
    logger.info("PREFLIGHT model %s", json.dumps(staged, indent=2))
    matched = [name for name, value in staged["candidates"].items() if value == EXPECTED_SHA256]
    digest_ok = bool(matched)
    bytes_ok = staged["total_bytes"] == EXPECTED_BYTES
    logger.info("PREFLIGHT digest conventions matching the expected value: %s", matched or "none")

    forward_seconds, scores = 0.0, np.zeros(0, dtype=np.float32)
    if not args.digest_only:
        scorer = load_folded_scorer(args.model_dir)
        rng = np.random.default_rng(0)
        ids = rng.integers(0, scorer.model.config.vocab_size, size=(PROBE_ROWS, scorer.max_tokens), dtype=np.int32)
        embed = rng.normal(size=(PROBE_ROWS, EMBED_DIM)).astype(np.float32)
        embed /= np.maximum(np.linalg.norm(embed, axis=1, keepdims=True), 1e-6)
        t0 = time.monotonic()
        scores = predict(scorer.model, ids, batch_size=PROBE_ROWS, doc_embed=embed)
        forward_seconds = time.monotonic() - t0

    result = {
        "jax_version": jax.__version__,
        "backend": backend,
        "num_devices": len(devices),
        "device_kinds": sorted({d.device_kind for d in devices}),
        "model_dir": args.model_dir,
        "model_files": staged["files"],
        "model_total_bytes": staged["total_bytes"],
        "model_sha256": staged["sha256"],
        "digest_matches": digest_ok,
        "digest_conventions_matched": matched,
        "digest_candidates": staged["candidates"],
        "per_file_sha256": staged["per_file_sha256"],
        "relative_paths": staged["relative_paths"],
        "byte_count_matches": bytes_ok,
        "probe_forward_seconds": forward_seconds,
        "probe_score_mean": float(scores.mean()) if len(scores) else None,
        "probe_score_std": float(scores.std()) if len(scores) else None,
    }
    logger.info("PREFLIGHT result %s", json.dumps(result, indent=2))
    if backend == "cpu" and not args.digest_only:
        raise RuntimeError(f"jax backend is CPU-only ({devices}); the forward would run on cores")
    if not digest_ok:
        raise RuntimeError(f"model digest {staged['sha256']} != expected {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()
