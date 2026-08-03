# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit a folded semantic projection on the disjoint 50K GLM label set."""

import hashlib
import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_fast_student import load_student
from evaluate_hierarchical_embeddings import hierarchical_assignments
from fast_student import FastStudent
from glm_hierarchical_labels import MAXIMUM_VALIDATION_REPAIR_FRACTION, OUTPUT_ROOT, VALIDATION_REPAIR_PREFIX
from glm_semantic_labels import SampleDocument, read_jsonl
from label_frozen_hierarchy_training import identity_digest
from ladder_config import write_json
from rigging.filesystem import StoragePath
from semantic_embedding_metrics import normalize_embeddings
from train_fast_student import OUTPUT_ROOT as TRAINING_ROOT
from train_fast_student import upload_array, upload_model
from train_semantic_projection import (
    BASE_RUNG,
    BASE_TRAINING_NAME,
    CONFIDENCE_DROP_FRACTION,
    FORM_WEIGHT,
    HIERARCHY_RUN_ID,
    HIERARCHY_VARIANT,
    IDENTITY_WEIGHT,
    LEAF_WEIGHT,
    LEARNING_RATE,
    MAXIMUM_VALIDATION_LEVEL_LOSS,
    MINIMUM_EFFECTIVE_RANK_FRACTION,
    MINIMUM_FOLDED_COSINE,
    MINIMUM_TOTAL_VARIANCE,
    MINIMUM_VALIDATION_SEMANTIC_DELTA,
    ORTHOGONAL_WEIGHT,
    PARENT_WEIGHT,
    PROJECTION_MIX_ALPHAS,
    SPREAD_COVARIANCE_WEIGHT,
    SPREAD_STANDARD_DEVIATION_TARGET,
    SPREAD_WEIGHT,
    TEMPERATURE,
    TRAINING_NAME,
    SemanticLabels,
    evaluation_metrics,
    fit_projection_minibatches,
    fold_embedding_projection,
    integer_labels,
    retained_train_validation_indices,
    select_projection_mix,
    validation_decision,
)

from experiments.datakit.cluster.quality.fast_transformer.model import count_params

LABEL_RUN_ID = "projection-train-50000-20260803-001"
EXCLUDED_EVALUATION_RUN_ID = "heldout-10000-20260802-001"
RUNG = "training-50k-v1"
VALIDATION_FRACTION = 0.05
BATCH_SIZE = 1_024
EPOCHS = 10
EXPECTED_DOCUMENTS = 50_000
RESULT_FILE = Path("/tmp/luxical-semantic-projection-large")

logger = logging.getLogger(__name__)


def file_text_sha256(text: str) -> str:
    """Return the SHA-256 digest of text."""
    return hashlib.sha256(text.encode()).hexdigest()


def validated_training_documents(
    training_root: StoragePath,
) -> tuple[list[SampleDocument], dict[str, Any], dict[str, str]]:
    """Return a complete, disjoint, and pinned GLM training sample."""
    config_text = (training_root / "run-config.json").read_text()
    summary_text = (training_root / "summary.json").read_text()
    config = json.loads(config_text)
    summary = json.loads(summary_text)
    documents = [SampleDocument(**row) for row in read_jsonl(training_root / "sample-private.jsonl.gz")]
    documents.sort(key=lambda row: row.sample_index)
    if config.get("purpose") != "semantic_projection_training":
        raise ValueError("The GLM label run is not a semantic-projection training set")
    if config.get("excluded_evaluation_run_id") != EXCLUDED_EVALUATION_RUN_ID:
        raise ValueError("The GLM label run did not exclude the fixed evaluation set")
    if len(documents) != EXPECTED_DOCUMENTS or summary.get("documents") != EXPECTED_DOCUMENTS:
        raise ValueError("The GLM label run does not contain exactly 50,000 documents")
    if not summary.get("complete"):
        raise ValueError("The GLM label run is not complete")
    repair_count = int(summary.get("validation_repair_count", -1))
    if repair_count < 0 or repair_count > EXPECTED_DOCUMENTS * MAXIMUM_VALIDATION_REPAIR_FRACTION:
        raise ValueError("The GLM label run has an invalid validation-repair count")
    if [row.sample_index for row in documents] != list(range(EXPECTED_DOCUMENTS)):
        raise ValueError("The GLM training sample indices are not complete")
    if identity_digest(documents) != config.get("training_identity_sha256"):
        raise ValueError("The GLM training identity digest does not match its config")
    metadata = {
        "run_config_url": str(training_root / "run-config.json"),
        "run_config_sha256": file_text_sha256(config_text),
        "summary_url": str(training_root / "summary.json"),
        "summary_sha256": file_text_sha256(summary_text),
    }
    return documents, config, metadata


def main() -> None:
    """Fit, fold, upload, and report the 50K semantic projection."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    training_root = OUTPUT_ROOT / HIERARCHY_RUN_ID / HIERARCHY_VARIANT / LABEL_RUN_ID
    documents, label_config, label_metadata = validated_training_documents(training_root)
    assignments = hierarchical_assignments(training_root, documents)
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
        raise ValueError("A validation-repaired GLM label entered the retained sample")
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
            epochs=EPOCHS,
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
        projected_vectors = normalize_embeddings(base_vectors @ projection)
        folded_model = fold_embedding_projection(student.model, projection)
        folded_student = FastStudent(folded_model, student.raw_to_compact, student.tokenizer_name)
        folded_vectors = folded_student(texts)
        folded_cosines = np.sum(
            normalize_embeddings(projected_vectors) * normalize_embeddings(folded_vectors),
            axis=1,
        )
        folded_cosine_minimum = float(folded_cosines.min())
        projected_validation = evaluation_metrics(
            folded_vectors[validation_indices],
            validation_labels,
            sources[validation_indices],
        )
        decision = validation_decision(base_validation, projected_validation, folded_cosine_minimum)
        output_root = f"{TRAINING_ROOT}/{TRAINING_NAME}/{RUNG}"
        model_url = f"{output_root}/model.eqx"
        projection_url = f"{output_root}/projection.npy"
        raw_projection_url = f"{output_root}/raw-projection.npy"
        model_sha256 = upload_model(folded_model, model_url)
        projection_sha256 = upload_array(projection, projection_url)
        raw_projection_sha256 = upload_array(raw_projection, raw_projection_url)

    report = {
        "config_name": "full",
        "training_name": TRAINING_NAME,
        "rung": RUNG,
        "training_rows": len(training_indices),
        "validation_rows": len(validation_indices),
        "confidence_cutoff": confidence_cutoff,
        "dropped_rows": int(len(documents) - len(training_indices) - len(validation_indices)),
        "validation_repair_count": len(repaired_indices),
        "label_run_id": LABEL_RUN_ID,
        "label_run_config": label_config,
        "label_artifacts": label_metadata,
        "hierarchy_run_id": HIERARCHY_RUN_ID,
        "hierarchy_variant": HIERARCHY_VARIANT,
        "excluded_evaluation_run_id": EXCLUDED_EVALUATION_RUN_ID,
        "base_training_name": BASE_TRAINING_NAME,
        "base_rung": BASE_RUNG,
        "base_model_sha256": base_report["final_model_sha256"],
        "final_model_url": model_url,
        "final_model_sha256": model_sha256,
        "final_projection_url": projection_url,
        "final_projection_sha256": projection_sha256,
        "raw_projection_url": raw_projection_url,
        "raw_projection_sha256": raw_projection_sha256,
        "raw_to_compact_url": base_report["raw_to_compact_url"],
        "raw_to_compact_sha256": base_report["raw_to_compact_sha256"],
        "parameters": count_params(folded_model),
        "config": asdict(folded_model.backbone.config),
        "objective": {
            "learning_rate": LEARNING_RATE,
            "temperature": TEMPERATURE,
            "parent_weight": PARENT_WEIGHT,
            "leaf_weight": LEAF_WEIGHT,
            "form_weight": FORM_WEIGHT,
            "identity_weight": IDENTITY_WEIGHT,
            "orthogonal_weight": ORTHOGONAL_WEIGHT,
            "spread_weight": SPREAD_WEIGHT,
            "spread_standard_deviation_target": SPREAD_STANDARD_DEVIATION_TARGET,
            "spread_covariance_weight": SPREAD_COVARIANCE_WEIGHT,
            "minimum_validation_semantic_delta": MINIMUM_VALIDATION_SEMANTIC_DELTA,
            "maximum_validation_level_loss": MAXIMUM_VALIDATION_LEVEL_LOSS,
            "minimum_effective_rank_fraction": MINIMUM_EFFECTIVE_RANK_FRACTION,
            "minimum_total_variance": MINIMUM_TOTAL_VARIANCE,
            "minimum_folded_cosine": MINIMUM_FOLDED_COSINE,
            "projection_mix_alphas": list(PROJECTION_MIX_ALPHAS),
            "validation_fraction": VALIDATION_FRACTION,
            "confidence_drop_fraction": CONFIDENCE_DROP_FRACTION,
            "minibatch": minibatch_audit,
        },
        "selected_projection_alpha": selected_alpha,
        "projection_mix_candidates": projection_mix_candidates,
        "base_validation": base_validation,
        "projected_validation": projected_validation,
        "validation_decision": decision,
        "history": history,
    }
    report_url = f"{output_root}/training.json"
    write_json(report_url, report)
    summary = {
        "report_url": report_url,
        "model_url": model_url,
        "model_sha256": model_sha256,
        "training_rows": len(training_indices),
        "validation_rows": len(validation_indices),
        "selected_projection_alpha": selected_alpha,
        "base_validation": base_validation,
        "projected_validation": projected_validation,
        "validation_decision": decision,
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("GLM_SEMANTIC_PROJECTION_LARGE=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
