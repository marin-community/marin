# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit an internal semantic projection on a complete assignment prefix."""

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_fast_student import load_student
from fast_student import FastStudent
from glm_hierarchical_labels import OUTPUT_ROOT, VALIDATION_REPAIR_PREFIX, HierarchicalAssignment
from glm_semantic_labels import SampleDocument, read_jsonl
from label_frozen_hierarchy_training import identity_digest
from ladder_config import write_json
from rigging.filesystem import StoragePath
from semantic_embedding_metrics import normalize_embeddings
from train_fast_student import OUTPUT_ROOT as TRAINING_ROOT
from train_semantic_projection import (
    BASE_RUNG,
    BASE_TRAINING_NAME,
    CONFIDENCE_DROP_FRACTION,
    HIERARCHY_RUN_ID,
    HIERARCHY_VARIANT,
    SemanticLabels,
    evaluation_metrics,
    fit_projection_minibatches,
    fold_embedding_projection,
    integer_labels,
    retained_train_validation_indices,
    select_projection_mix,
    validation_decision,
)
from train_semantic_projection_large import (
    BATCH_SIZE,
    EXCLUDED_EVALUATION_RUN_ID,
    LABEL_RUN_ID,
    VALIDATION_FRACTION,
    file_text_sha256,
)

EXPECTED_FULL_SAMPLE_DOCUMENTS = 50_000
MINIMUM_PREFIX_DOCUMENTS = 1_000
RESULT_FILE = Path("/tmp/luxical-semantic-projection-prefix")

logger = logging.getLogger(__name__)


def assignment_prefix(assignment_root: StoragePath, expected_documents: int) -> list[HierarchicalAssignment]:
    """Return one complete and ordered prefix from saved assignment files."""
    paths = (assignment_root / "assignments" / "*.jsonl.gz").glob()
    assignments = [HierarchicalAssignment(**row) for path in paths for row in read_jsonl(path)]
    return validated_assignment_prefix(assignments, expected_documents)


def validated_assignment_prefix(
    assignments: list[HierarchicalAssignment], expected_documents: int
) -> list[HierarchicalAssignment]:
    """Select and validate an assignment prefix from loaded rows."""
    prefix = sorted(
        (row for row in assignments if row.sample_index < expected_documents),
        key=lambda row: row.sample_index,
    )
    if [row.sample_index for row in prefix] != list(range(expected_documents)):
        raise ValueError("The semantic assignment prefix is not complete and ordered")
    return prefix


def validated_prefix_inputs(
    training_root: StoragePath, expected_documents: int
) -> tuple[list[SampleDocument], list[HierarchicalAssignment], dict[str, Any]]:
    """Return a disjoint sample prefix and its complete assignments."""
    config_text = (training_root / "run-config.json").read_text()
    config = json.loads(config_text)
    documents = [SampleDocument(**row) for row in read_jsonl(training_root / "sample-private.jsonl.gz")]
    documents.sort(key=lambda row: row.sample_index)
    if config.get("purpose") != "semantic_projection_training":
        raise ValueError("The GLM label run is not a semantic-projection training set")
    if config.get("excluded_evaluation_run_id") != EXCLUDED_EVALUATION_RUN_ID:
        raise ValueError("The GLM label run did not exclude the fixed evaluation set")
    if len(documents) != EXPECTED_FULL_SAMPLE_DOCUMENTS:
        raise ValueError("The GLM label run does not contain the full 50,000-document sample")
    if [row.sample_index for row in documents] != list(range(EXPECTED_FULL_SAMPLE_DOCUMENTS)):
        raise ValueError("The GLM training sample indices are not complete")
    if identity_digest(documents) != config.get("training_identity_sha256"):
        raise ValueError("The GLM training identity digest does not match its config")
    if expected_documents < MINIMUM_PREFIX_DOCUMENTS or expected_documents >= len(documents):
        raise ValueError("The internal prefix size is outside its fixed range")
    selected_documents = documents[:expected_documents]
    assignments = assignment_prefix(training_root, expected_documents)
    metadata = {
        "run_config_url": str(training_root / "run-config.json"),
        "run_config_sha256": file_text_sha256(config_text),
        "full_training_identity_sha256": config["training_identity_sha256"],
    }
    return selected_documents, assignments, metadata


def main() -> None:
    """Fit and report one internal prefix rung without held-out evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-documents", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    training_root = OUTPUT_ROOT / HIERARCHY_RUN_ID / HIERARCHY_VARIANT / LABEL_RUN_ID
    documents, assignments, label_metadata = validated_prefix_inputs(training_root, args.prefix_documents)
    repaired_indices = {row.sample_index for row in assignments if row.rationale.startswith(VALIDATION_REPAIR_PREFIX)}
    confidences = np.asarray([row.confidence for row in assignments])
    leaf_names = [row.primary_leaf_id for row in assignments]
    training_indices, validation_indices, confidence_cutoff = retained_train_validation_indices(
        confidences,
        leaf_names,
        validation_fraction=VALIDATION_FRACTION,
        drop_fraction=CONFIDENCE_DROP_FRACTION,
    )
    retained_indices = set(training_indices.tolist()) | set(validation_indices.tolist())
    if repaired_indices & retained_indices:
        raise ValueError("A validation-repaired GLM label entered the retained prefix")
    labels = SemanticLabels(
        parent=integer_labels([row.primary_parent_id for row in assignments]),
        leaf=integer_labels(leaf_names),
        form=integer_labels([row.form_id for row in assignments]),
    )
    sources = integer_labels([row.source for row in documents])
    texts = [row.text for row in documents]

    with tempfile.TemporaryDirectory() as temporary_directory:
        student, base_report = load_student("full", BASE_TRAINING_NAME, BASE_RUNG, Path(temporary_directory))
        base_vectors = student(texts)
        training_labels = SemanticLabels(
            parent=labels.parent[training_indices],
            leaf=labels.leaf[training_indices],
            form=labels.form[training_indices],
        )
        raw_projection, history, minibatch_audit = fit_projection_minibatches(
            base_vectors[training_indices],
            training_labels,
            sources[training_indices],
            batch_size=BATCH_SIZE,
            epochs=10,
        )
        validation_labels = SemanticLabels(
            parent=labels.parent[validation_indices],
            leaf=labels.leaf[validation_indices],
            form=labels.form[validation_indices],
        )
        base_validation = evaluation_metrics(
            base_vectors[validation_indices],
            validation_labels,
            sources[validation_indices],
        )
        selected_alpha, projection, projection_mix_candidates = select_projection_mix(
            base_vectors,
            raw_projection,
            validation_indices,
            validation_labels,
            sources,
            base_validation,
        )
        folded_model = fold_embedding_projection(student.model, projection)
        folded_student = FastStudent(folded_model, student.raw_to_compact, student.tokenizer_name)
        folded_vectors = folded_student(texts)
        projected_vectors = normalize_embeddings(base_vectors @ projection)
        folded_cosines = np.sum(projected_vectors * normalize_embeddings(folded_vectors), axis=1)
        folded_cosine_minimum = float(folded_cosines.min())
        projected_validation = evaluation_metrics(
            folded_vectors[validation_indices],
            validation_labels,
            sources[validation_indices],
        )
        decision = validation_decision(base_validation, projected_validation, folded_cosine_minimum)

    output_root = f"{TRAINING_ROOT}/full-glm-semantic-projection-prefix/prefix-{args.prefix_documents}"
    report = {
        "purpose": "internal_prefix_validation",
        "heldout_evaluation_used": False,
        "prefix_documents": args.prefix_documents,
        "training_rows": len(training_indices),
        "validation_rows": len(validation_indices),
        "dropped_rows": args.prefix_documents - len(training_indices) - len(validation_indices),
        "confidence_cutoff": confidence_cutoff,
        "validation_repair_count": len(repaired_indices),
        "label_run_id": LABEL_RUN_ID,
        "label_artifacts": label_metadata,
        "base_training_name": BASE_TRAINING_NAME,
        "base_rung": BASE_RUNG,
        "base_model_sha256": base_report["final_model_sha256"],
        "selected_projection_alpha": selected_alpha,
        "projection_mix_candidates": projection_mix_candidates,
        "base_validation": base_validation,
        "projected_validation": projected_validation,
        "validation_decision": decision,
        "minibatch": minibatch_audit,
        "history": history,
    }
    report_url = f"{output_root}/validation.json"
    write_json(report_url, report)
    summary = {
        "report_url": report_url,
        "prefix_documents": args.prefix_documents,
        "selected_projection_alpha": selected_alpha,
        "base_validation": base_validation,
        "projected_validation": projected_validation,
        "validation_decision": decision,
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("GLM_SEMANTIC_PROJECTION_PREFIX=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
