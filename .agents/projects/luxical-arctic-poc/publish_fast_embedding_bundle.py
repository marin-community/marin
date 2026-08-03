# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish and verify one production FastTransformer embedding bundle."""

import argparse
import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_fast_student import TRAINING_ROOT, load_student
from fast_student import BASELINE_FILE, BASELINE_REPO, BASELINE_REVISION, LUXICAL_TOKENIZER_NAME
from huggingface_hub import hf_hub_download
from rigging.filesystem import StoragePath
from verify_blind_neighborhood_with_claude import review_package_sha256

from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformerConfig
from experiments.datakit.embeddings.fast_transformer.embedder import (
    MANIFEST_FILENAME,
    FastEmbeddingBundleManifest,
    FastEmbeddingModel,
    payload_sha256,
)

MODEL_FILENAME = "model.eqx"
TOKEN_REMAP_FILENAME = "raw-to-compact.npy"
TOKENIZER_FILENAME = "tokenizer.json"
RELEASE_REPORT_FILENAME = "release-report.json"
OUTPUT_DIMENSION = 256
MINIMUM_CPU_SPEED_RATIO = 0.85
MINIMUM_PARITY_COSINE = 0.999999
BLIND_REVIEW_MODEL = "claude-opus-5"
RESULT_FILE = Path("/tmp/luxical-fast-embedding-release")
SMOKE_TEXTS = [
    "A short account of market structure and company finance.",
    "def merge_rows(rows):\n    return sorted(rows, key=lambda row: row['id'])",
    "日本語で書かれた技術文書と実行手順です。",
    "هذا نص عربي عن التاريخ والثقافة والتعليم.",
    '{"schema": {"type": "object"}, "required": ["name", "value"]}',
    "The experiment measures protein binding under three temperature settings.",
    "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id;",
    "First disconnect the supply. Then remove the access panel.",
]

logger = logging.getLogger(__name__)


def read_json_artifact(url: str) -> tuple[dict[str, Any], bytes]:
    """Return one JSON artifact and its exact payload."""
    payload = StoragePath(url).read_bytes()
    return json.loads(payload), payload


def release_evidence_decision(
    training_report: dict[str, Any],
    evaluation_report: dict[str, Any],
    speed_report: dict[str, Any],
    blind_review_report: dict[str, Any],
    *,
    config_name: str,
    training_name: str,
    rung: str,
    student_model: str,
    blind_package_sha256: str,
) -> dict[str, bool]:
    """Return the fixed release decisions for one exact student."""
    variant = evaluation_report["variants"]["compact"]
    evaluation_identity = (
        evaluation_report["student_config"] == config_name
        and evaluation_report["student_training_name"] == training_name
        and evaluation_report["student_rung"] == rung
        and evaluation_report["student_model"] == student_model
        and evaluation_report["documents"] >= 10_000
        and evaluation_report["label_version"] == "adjudicated"
        and bool(evaluation_report["adjudication_review_url"])
        and evaluation_report["source_metadata_used_as_quality_target"] is False
        and evaluation_report["model_metadata"][student_model]["final_model_sha256"]
        == training_report["final_model_sha256"]
    )
    speed_identity = (
        speed_report["config_name"] == config_name
        and speed_report["teacher"] == training_name
        and speed_report["rung"] == rung
        and speed_report["training_report"]["final_model_sha256"] == training_report["final_model_sha256"]
    )
    blind_identity = (
        blind_review_report["student_model"] == student_model
        and blind_review_report["claude_model"] == BLIND_REVIEW_MODEL
        and blind_review_report["overall"]["documents"] == 200
        and blind_review_report["package_sha256"] == blind_package_sha256
    )
    training_validation = training_report["validation_decision"]
    return {
        "training_validation": bool(training_validation) and all(bool(value) for value in training_validation.values()),
        "evaluation_identity": evaluation_identity,
        "parent_semantics": bool(variant["parent"]["student_all_gates_passed"]),
        "leaf_semantics": bool(variant["leaf"]["student_all_gates_passed"]),
        "form_semantics": bool(variant["form"]["student_all_gates_passed"]),
        "fixed_40_buckets": bool(variant["production_buckets"]["student_all_gates_passed"]),
        "speed_identity": speed_identity,
        "speed_stability": speed_report["measurement_valid"] is True,
        "cpu_speed": float(speed_report["student_to_baseline_ratio"]) >= MINIMUM_CPU_SPEED_RATIO,
        "blind_identity": blind_identity,
        "blind_overall": bool(blind_review_report["overall"]["release_gate_passed"]),
        "blind_code": bool(blind_review_report["code"]["release_gate_passed"]),
        "blind_non_english": bool(blind_review_report["non_english"]["release_gate_passed"]),
        "blind_other": bool(blind_review_report["other"]["release_gate_passed"]),
    }


def write_once(path: StoragePath, payload: bytes) -> None:
    """Write one immutable file or verify its existing payload."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"The release file at {path} already has different bytes")
        return
    path.write_bytes(payload)


def tokenizer_payload() -> bytes:
    """Return the tokenizer state from the pinned Luxical artifact."""
    baseline_path = hf_hub_download(repo_id=BASELINE_REPO, filename=BASELINE_FILE, revision=BASELINE_REVISION)
    with np.load(baseline_path, allow_pickle=False) as archive:
        return archive["tokenizer"].tobytes()


def main() -> None:
    """Validate evidence, publish a bundle, and do an exact parity smoke."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-name", required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--evaluation-report-url", required=True)
    parser.add_argument("--speed-report-url", required=True)
    parser.add_argument("--blind-review-report-url", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    training_report_url = f"{TRAINING_ROOT}/{args.training_name}/{args.rung}/training.json"
    training_report, training_payload = read_json_artifact(training_report_url)
    evaluation_report, evaluation_payload = read_json_artifact(args.evaluation_report_url)
    speed_report, speed_payload = read_json_artifact(args.speed_report_url)
    blind_review_report, blind_review_payload = read_json_artifact(args.blind_review_report_url)
    blind_package_url = evaluation_report["blind_neighborhood_package_url"]
    blind_package = json.loads(StoragePath(blind_package_url).read_text(compression="gzip"))
    blind_package_digest = review_package_sha256(blind_package)
    decision = release_evidence_decision(
        training_report,
        evaluation_report,
        speed_report,
        blind_review_report,
        config_name=args.config,
        training_name=args.training_name,
        rung=args.rung,
        student_model=args.student_model,
        blind_package_sha256=blind_package_digest,
    )
    if not all(decision.values()):
        failed = sorted(name for name, passed in decision.items() if not passed)
        raise ValueError(f"The student failed release evidence gates: {failed}")

    model_payload = StoragePath(training_report["final_model_url"]).read_bytes()
    remap_payload = StoragePath(training_report["raw_to_compact_url"]).read_bytes()
    if payload_sha256(model_payload) != training_report["final_model_sha256"]:
        raise ValueError("The training model digest does not match")
    if payload_sha256(remap_payload) != training_report["raw_to_compact_sha256"]:
        raise ValueError("The training token-remap digest does not match")
    remap = np.load(io.BytesIO(remap_payload), allow_pickle=False)
    tokenizer = tokenizer_payload()
    config = FastTransformerConfig(**training_report["config"])
    manifest = FastEmbeddingBundleManifest(
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
        training_report_url=training_report_url,
        training_report_sha256=payload_sha256(training_payload),
        evaluation_report_url=args.evaluation_report_url,
        evaluation_report_sha256=payload_sha256(evaluation_payload),
        speed_report_url=args.speed_report_url,
        speed_report_sha256=payload_sha256(speed_payload),
        blind_review_report_url=args.blind_review_report_url,
        blind_review_report_sha256=payload_sha256(blind_review_payload),
        blind_review_package_url=blind_package_url,
        blind_review_package_sha256=blind_package_digest,
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
        production_student = FastEmbeddingModel.load(args.output_root, manifest_sha256)
        research_vectors = research_student(SMOKE_TEXTS)
        production_vectors = production_student(SMOKE_TEXTS)
    parity_cosines = np.sum(research_vectors * production_vectors, axis=1)
    parity_cosine_minimum = float(parity_cosines.min())
    finite = bool(np.isfinite(production_vectors).all())
    unique = len(np.unique(np.round(production_vectors, 6), axis=0)) == len(production_vectors)
    if not finite or not unique or parity_cosine_minimum < MINIMUM_PARITY_COSINE:
        raise ValueError("The production bundle failed its parity smoke")
    report = {
        "bundle_root": args.output_root,
        "manifest_sha256": manifest_sha256,
        "model_sha256": manifest.model_sha256,
        "release_evidence_decision": decision,
        "smoke_documents": len(SMOKE_TEXTS),
        "finite": finite,
        "unique_6dp": unique,
        "research_to_production_minimum_cosine": parity_cosine_minimum,
    }
    report_payload = json.dumps(report, indent=2, sort_keys=True).encode()
    write_once(output_root / RELEASE_REPORT_FILENAME, report_payload)
    RESULT_FILE.write_text(json.dumps(report, sort_keys=True))
    logger.info("FAST_EMBEDDING_RELEASE=%s", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
