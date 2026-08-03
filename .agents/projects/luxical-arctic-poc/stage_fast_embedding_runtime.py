# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a pinned FastTransformer runtime before release evaluation."""

import argparse
import io
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
from evaluate_fast_student import TRAINING_ROOT, load_student
from fast_student import LUXICAL_TOKENIZER_NAME
from publish_fast_embedding_bundle import (
    MINIMUM_PARITY_COSINE,
    MODEL_FILENAME,
    OUTPUT_DIMENSION,
    SMOKE_TEXTS,
    TOKEN_REMAP_FILENAME,
    TOKENIZER_FILENAME,
    read_json_artifact,
    tokenizer_payload,
    write_once,
)
from rigging.filesystem import StoragePath

from experiments.datakit.cluster.quality.fast_transformer.model import (
    ACCELERATOR_COMPUTE_DTYPE_NAME,
    CPU_COMPUTE_DTYPE_NAME,
    FastTransformerConfig,
)
from experiments.datakit.embeddings.fast_transformer.embedder import (
    MANIFEST_FILENAME,
    FastEmbeddingModel,
    FastEmbeddingRuntimeManifest,
    payload_sha256,
)

CANDIDATE_REPORT_FILENAME = "candidate-report.json"
RESULT_FILE = Path("/tmp/luxical-fast-embedding-runtime")

logger = logging.getLogger(__name__)


def main() -> None:
    """Stage exact runtime files and verify research-to-loader parity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-name", required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    training_report_url = f"{TRAINING_ROOT}/{args.training_name}/{args.rung}/training.json"
    training_report, training_payload = read_json_artifact(training_report_url)
    expected_identity = {
        "config_name": args.config,
        "training_name": args.training_name,
        "rung": args.rung,
    }
    if any(training_report.get(name) != value for name, value in expected_identity.items()):
        raise ValueError("The training report does not identify the requested runtime")
    if training_report["validation_decision"]["passed"] is not True:
        raise ValueError("The training report did not pass private validation")

    model_payload = StoragePath(training_report["final_model_url"]).read_bytes()
    remap_payload = StoragePath(training_report["raw_to_compact_url"]).read_bytes()
    if payload_sha256(model_payload) != training_report["final_model_sha256"]:
        raise ValueError("The training model digest does not match")
    if payload_sha256(remap_payload) != training_report["raw_to_compact_sha256"]:
        raise ValueError("The training token-remap digest does not match")
    remap = np.load(io.BytesIO(remap_payload), allow_pickle=False)
    tokenizer = tokenizer_payload()
    config = FastTransformerConfig(**training_report["config"])
    manifest = FastEmbeddingRuntimeManifest(
        model_filename=MODEL_FILENAME,
        model_sha256=payload_sha256(model_payload),
        token_remap_filename=TOKEN_REMAP_FILENAME,
        token_remap_sha256=payload_sha256(remap_payload),
        tokenizer_filename=TOKENIZER_FILENAME,
        tokenizer_sha256=payload_sha256(tokenizer),
        tokenizer_name=LUXICAL_TOKENIZER_NAME,
        raw_vocab_size=len(remap),
        config=config,
        output_dimension=OUTPUT_DIMENSION,
        characters_per_region=config.max_tokens,
        cpu_compute_dtype=CPU_COMPUTE_DTYPE_NAME,
        accelerator_compute_dtype=ACCELERATOR_COMPUTE_DTYPE_NAME,
        training_report_url=training_report_url,
        training_report_sha256=payload_sha256(training_payload),
    )
    manifest_payload = manifest.model_dump_json(indent=2).encode()
    manifest_sha256 = payload_sha256(manifest_payload)
    output_root = StoragePath(args.output_root)
    output_root.mkdirs()
    write_once(output_root / MODEL_FILENAME, model_payload)
    write_once(output_root / TOKEN_REMAP_FILENAME, remap_payload)
    write_once(output_root / TOKENIZER_FILENAME, tokenizer)
    write_once(output_root / MANIFEST_FILENAME, manifest_payload)

    with tempfile.TemporaryDirectory() as temporary_directory:
        research_student, _ = load_student(args.config, args.training_name, args.rung, Path(temporary_directory))
        runtime_student = FastEmbeddingModel.load_runtime(args.output_root, manifest_sha256)
        research_vectors = research_student(SMOKE_TEXTS)
        runtime_vectors = runtime_student(SMOKE_TEXTS)
    parity_cosines = np.sum(research_vectors * runtime_vectors, axis=1)
    parity_cosine_minimum = float(parity_cosines.min())
    finite = bool(np.isfinite(runtime_vectors).all())
    unique = len(np.unique(np.round(runtime_vectors, 6), axis=0)) == len(runtime_vectors)
    if not finite or not unique or parity_cosine_minimum < MINIMUM_PARITY_COSINE:
        raise ValueError("The staged runtime failed its parity smoke")
    report = {
        "bundle_root": args.output_root,
        "manifest_sha256": manifest_sha256,
        "model_sha256": manifest.model_sha256,
        "training_report_url": training_report_url,
        "training_report_sha256": manifest.training_report_sha256,
        "smoke_documents": len(SMOKE_TEXTS),
        "finite": finite,
        "unique_6dp": unique,
        "research_to_runtime_minimum_cosine": parity_cosine_minimum,
    }
    report_payload = json.dumps(report, indent=2, sort_keys=True).encode()
    write_once(output_root / CANDIDATE_REPORT_FILENAME, report_payload)
    RESULT_FILE.write_text(json.dumps(report, sort_keys=True))
    logger.info("FAST_EMBEDDING_RUNTIME=%s", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
