# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune the 30M Arctic student on the disjoint 50K GLM labels."""

import json
import logging
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from evaluate_fast_student import load_student
from evaluate_hierarchical_embeddings import hierarchical_assignments
from fast_student import packed_document_ids
from glm_hierarchical_labels import OUTPUT_ROOT, VALIDATION_REPAIR_PREFIX
from ladder_config import SEED, write_json
from semantic_embedding_metrics import normalize_embeddings
from train_fast_student import OUTPUT_ROOT as TRAINING_ROOT
from train_fast_student import upload_model
from train_semantic_projection import (
    BASE_RUNG,
    BASE_TRAINING_NAME,
    CONFIDENCE_DROP_FRACTION,
    FORM_WEIGHT,
    HIERARCHY_RUN_ID,
    HIERARCHY_VARIANT,
    IDENTITY_WEIGHT,
    LEAF_WEIGHT,
    PARENT_WEIGHT,
    PROJECTION_MIX_ALPHAS,
    SPREAD_COVARIANCE_WEIGHT,
    SPREAD_STANDARD_DEVIATION_TARGET,
    SPREAD_WEIGHT,
    SemanticLabels,
    evaluation_metrics,
    integer_labels,
    retained_train_validation_indices,
    semantic_validation_decision,
    supervised_contrastive_loss,
)
from train_semantic_projection_large import (
    LABEL_RUN_ID,
    VALIDATION_FRACTION,
    validated_training_documents,
)

from experiments.datakit.cluster.quality.fast_transformer.embedding import embedding_spread_loss, predict_embeddings
from experiments.datakit.cluster.quality.fast_transformer.model import FastEmbeddingTransformer, count_params

TRAINING_NAME = "full-glm-semantic-finetune"
RUNG = "training-50k-rank-preserving-v2"
BATCH_SIZE = 1_024
EPOCHS = 3
LEARNING_RATE = 5e-5
GRADIENT_CLIP = 1.0
MIN_BASE_RANK_RETENTION = 0.75
RESULT_FILE = Path("/tmp/luxical-semantic-finetune-large")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelMixCandidate:
    """Store one base-to-fine-tuned model validation result."""

    alpha: float
    metrics: dict[str, Any]
    provisional_decision: dict[str, Any]


def rank_preservation_decision(
    base_validation: dict[str, Any],
    candidate_validation: dict[str, Any],
) -> dict[str, Any]:
    """Return the base-rank gate for one model mix."""
    base_rank = float(base_validation["geometry"]["effective_rank_fraction"])
    candidate_rank = float(candidate_validation["geometry"]["effective_rank_fraction"])
    rank_retention = candidate_rank / base_rank
    minimum_candidate_rank = MIN_BASE_RANK_RETENTION * base_rank
    rank_preserved = candidate_rank >= minimum_candidate_rank or math.isclose(
        candidate_rank,
        minimum_candidate_rank,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    return {
        "base_rank_fraction": base_rank,
        "candidate_rank_fraction": candidate_rank,
        "base_rank_retention": rank_retention,
        "minimum_base_rank_retention": MIN_BASE_RANK_RETENTION,
        "rank_preserved": rank_preserved,
    }


def model_mix_validation_decision(
    base_validation: dict[str, Any],
    candidate_validation: dict[str, Any],
) -> dict[str, Any]:
    """Return semantic and base-rank gates for one model mix."""
    decision = semantic_validation_decision(base_validation, candidate_validation)
    rank_decision = rank_preservation_decision(base_validation, candidate_validation)
    return {
        **decision,
        **rank_decision,
        "semantic_gates_passed": decision["passed"],
        "passed": decision["passed"] and rank_decision["rank_preserved"],
    }


def best_passing_model_mix(candidates: list[ModelMixCandidate]) -> ModelMixCandidate:
    """Return the passing model mix with the largest semantic gain."""
    passed = [row for row in candidates if row.provisional_decision["passed"]]
    if not passed:
        raise ValueError("No model mix passed the private semantic and rank gates")
    return max(passed, key=lambda row: row.provisional_decision["semantic_mean_delta"])


def semantic_model_loss(
    model: FastEmbeddingTransformer,
    ids: jax.Array,
    base_vectors: jax.Array,
    labels: SemanticLabels,
    source_ids: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return the hierarchy, base-anchor, and spread objective."""
    vectors = model(ids, key=key, inference=False)
    vectors /= jnp.maximum(jnp.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    base_vectors /= jnp.maximum(jnp.linalg.norm(base_vectors, axis=1, keepdims=True), 1e-12)
    parent = supervised_contrastive_loss(vectors, jnp.asarray(labels.parent), source_ids, 0.1)
    leaf = supervised_contrastive_loss(vectors, jnp.asarray(labels.leaf), source_ids, 0.1)
    form = supervised_contrastive_loss(vectors, jnp.asarray(labels.form), source_ids, 0.1)
    identity = jnp.mean(1.0 - jnp.sum(vectors * base_vectors, axis=1))
    spread = embedding_spread_loss(
        vectors,
        SPREAD_STANDARD_DEVIATION_TARGET,
        SPREAD_COVARIANCE_WEIGHT,
    )
    semantic = PARENT_WEIGHT * parent + LEAF_WEIGHT * leaf + FORM_WEIGHT * form
    total = semantic + IDENTITY_WEIGHT * identity + SPREAD_WEIGHT * spread
    return total, {
        "parent": parent,
        "leaf": leaf,
        "form": form,
        "semantic": semantic,
        "identity": identity,
        "spread": spread,
    }


def fit_semantic_model(
    model: FastEmbeddingTransformer,
    ids: np.ndarray,
    base_vectors: np.ndarray,
    labels: SemanticLabels,
    source_ids: np.ndarray,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
) -> tuple[FastEmbeddingTransformer, list[dict[str, float]], dict[str, int]]:
    """Fine-tune one model with deterministic fixed-size minibatches."""
    rows = len(ids)
    if rows < 1 or batch_size < 1 or epochs < 1:
        raise ValueError("Fine-tuning rows, batch size, and epochs must be positive")
    if base_vectors.shape != (rows, model.output_dim):
        raise ValueError("Base vectors do not match the fine-tuning rows")
    if labels.parent.shape != (rows,) or labels.leaf.shape != (rows,) or labels.form.shape != (rows,):
        raise ValueError("Semantic labels do not match the fine-tuning rows")
    if source_ids.shape != (rows,):
        raise ValueError("Source IDs do not match the fine-tuning rows")

    optimizer = optax.chain(optax.clip_by_global_norm(GRADIENT_CLIP), optax.adam(LEARNING_RATE))
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(
        current_model: FastEmbeddingTransformer,
        current_optimizer_state: optax.OptState,
        batch_ids: jax.Array,
        batch_base_vectors: jax.Array,
        batch_labels: SemanticLabels,
        batch_source_ids: jax.Array,
        key: jax.Array,
    ):
        (loss, components), gradients = eqx.filter_value_and_grad(semantic_model_loss, has_aux=True)(
            current_model,
            batch_ids,
            batch_base_vectors,
            batch_labels,
            batch_source_ids,
            key,
        )
        updates, next_optimizer_state = optimizer.update(
            gradients,
            current_optimizer_state,
            eqx.filter(current_model, eqx.is_inexact_array),
        )
        return eqx.apply_updates(current_model, updates), next_optimizer_state, loss, components

    batches_per_epoch = math.ceil(rows / batch_size)
    padded_rows_per_epoch = batches_per_epoch * batch_size
    generator = np.random.default_rng(SEED + 710_000)
    key = jax.random.PRNGKey(SEED + 710_001)
    history = []
    update_index = 0
    for epoch_index in range(epochs):
        order = generator.permutation(rows)
        padded_order = np.resize(order, padded_rows_per_epoch)
        for start in range(0, padded_rows_per_epoch, batch_size):
            update_index += 1
            indices = padded_order[start : start + batch_size]
            key, step_key = jax.random.split(key)
            batch_labels = SemanticLabels(
                parent=jnp.asarray(labels.parent[indices]),
                leaf=jnp.asarray(labels.leaf[indices]),
                form=jnp.asarray(labels.form[indices]),
            )
            model, optimizer_state, loss, components = step(
                model,
                optimizer_state,
                jnp.asarray(ids[indices]),
                jnp.asarray(base_vectors[indices]),
                batch_labels,
                jnp.asarray(source_ids[indices]),
                step_key,
            )
            if update_index == 1 or update_index % 25 == 0 or update_index == batches_per_epoch * epochs:
                row = {
                    "step": float(update_index),
                    "epoch": float(epoch_index + 1),
                    "loss": float(loss),
                }
                row.update({name: float(value) for name, value in components.items()})
                if not np.isfinite(list(row.values())).all():
                    raise ValueError(f"Fine-tuning returned a non-finite value at update {update_index}")
                history.append(row)
                logger.info(
                    "Semantic update %d/%d: %s",
                    update_index,
                    batches_per_epoch * epochs,
                    json.dumps(row, sort_keys=True),
                )
    audit = {
        "rows": rows,
        "batch_size": batch_size,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "padded_rows_per_epoch": padded_rows_per_epoch,
        "updates": update_index,
    }
    return model, history, audit


def interpolate_models(
    base_model: FastEmbeddingTransformer,
    fine_tuned_model: FastEmbeddingTransformer,
    alpha: float,
) -> FastEmbeddingTransformer:
    """Interpolate equal model trees without an inference-time operation."""
    if not 0 <= alpha <= 1:
        raise ValueError("The model mix must be from zero through one")
    if alpha == 0:
        return base_model
    if alpha == 1:
        return fine_tuned_model
    base_parameters, base_static = eqx.partition(base_model, eqx.is_inexact_array)
    tuned_parameters, tuned_static = eqx.partition(fine_tuned_model, eqx.is_inexact_array)
    if not eqx.tree_equal(base_static, tuned_static):
        raise ValueError("The base and fine-tuned model structures differ")
    mixed_parameters = jax.tree.map(
        lambda base, tuned: None if base is None else base + alpha * (tuned - base),
        base_parameters,
        tuned_parameters,
        is_leaf=lambda value: value is None,
    )
    return eqx.combine(mixed_parameters, base_static)


def select_model_mix(
    base_model: FastEmbeddingTransformer,
    fine_tuned_model: FastEmbeddingTransformer,
    validation_ids: np.ndarray,
    validation_labels: SemanticLabels,
    validation_sources: np.ndarray,
    base_validation: dict[str, Any],
) -> tuple[float, FastEmbeddingTransformer, list[dict[str, Any]]]:
    """Select the best base-to-fine-tuned mix that passes validation."""
    candidates = []
    for alpha in PROJECTION_MIX_ALPHAS:
        candidate_model = interpolate_models(base_model, fine_tuned_model, alpha)
        candidate_vectors = predict_embeddings(candidate_model, validation_ids, batch_size=4_096)
        metrics = evaluation_metrics(candidate_vectors, validation_labels, validation_sources)
        decision = model_mix_validation_decision(base_validation, metrics)
        candidates.append(ModelMixCandidate(alpha, metrics, decision))
    logger.info("MODEL_MIX_CANDIDATES=%s", json.dumps([asdict(row) for row in candidates], sort_keys=True))
    selected = best_passing_model_mix(candidates)
    return (
        selected.alpha,
        interpolate_models(base_model, fine_tuned_model, selected.alpha),
        [asdict(row) for row in candidates],
    )


def main() -> None:
    """Fine-tune, mix, upload, and report the 50K semantic model."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    if jax.default_backend() != "gpu":
        raise ValueError(f"Semantic fine-tuning requires a GPU backend, got {jax.default_backend()}")
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
        base_student, base_report = load_student(
            "full",
            BASE_TRAINING_NAME,
            BASE_RUNG,
            Path(temporary_directory),
        )
        base_vectors = base_student(texts)
        model_config = base_student.model.backbone.config
        ids = packed_document_ids(
            texts,
            base_student.raw_to_compact,
            base_student.tokenizer_name,
            max_tokens=model_config.max_tokens,
            characters_per_source_window=model_config.max_tokens,
        )
        training_labels = SemanticLabels(
            parent=labels.parent[training_indices],
            leaf=labels.leaf[training_indices],
            form=labels.form[training_indices],
        )
        fine_tuned_model, history, minibatch_audit = fit_semantic_model(
            base_student.model,
            ids[training_indices],
            base_vectors[training_indices],
            training_labels,
            sources[training_indices],
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
        selected_alpha, selected_model, mix_candidates = select_model_mix(
            base_student.model,
            fine_tuned_model,
            ids[validation_indices],
            validation_labels,
            sources[validation_indices],
            base_validation,
        )
        selected_vectors = predict_embeddings(selected_model, ids, batch_size=4_096)
        selected_validation = evaluation_metrics(
            selected_vectors[validation_indices],
            validation_labels,
            sources[validation_indices],
        )
        decision = model_mix_validation_decision(base_validation, selected_validation)
        normalized_base = normalize_embeddings(base_vectors)
        normalized_selected = normalize_embeddings(selected_vectors)
        base_cosines = np.sum(normalized_base * normalized_selected, axis=1)
        output_root = f"{TRAINING_ROOT}/{TRAINING_NAME}/{RUNG}"
        model_url = f"{output_root}/model.eqx"
        raw_model_url = f"{output_root}/raw-fine-tuned-model.eqx"
        model_sha256 = upload_model(selected_model, model_url)
        raw_model_sha256 = upload_model(fine_tuned_model, raw_model_url)

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
        "base_training_name": BASE_TRAINING_NAME,
        "base_rung": BASE_RUNG,
        "base_model_sha256": base_report["final_model_sha256"],
        "final_model_url": model_url,
        "final_model_sha256": model_sha256,
        "raw_fine_tuned_model_url": raw_model_url,
        "raw_fine_tuned_model_sha256": raw_model_sha256,
        "raw_to_compact_url": base_report["raw_to_compact_url"],
        "raw_to_compact_sha256": base_report["raw_to_compact_sha256"],
        "parameters": count_params(selected_model),
        "config": asdict(selected_model.backbone.config),
        "objective": {
            "learning_rate": LEARNING_RATE,
            "gradient_clip": GRADIENT_CLIP,
            "parent_weight": PARENT_WEIGHT,
            "leaf_weight": LEAF_WEIGHT,
            "form_weight": FORM_WEIGHT,
            "identity_weight": IDENTITY_WEIGHT,
            "spread_weight": SPREAD_WEIGHT,
            "spread_standard_deviation_target": SPREAD_STANDARD_DEVIATION_TARGET,
            "spread_covariance_weight": SPREAD_COVARIANCE_WEIGHT,
            "model_mix_alphas": list(PROJECTION_MIX_ALPHAS),
            "minimum_base_rank_retention": MIN_BASE_RANK_RETENTION,
            "validation_fraction": VALIDATION_FRACTION,
            "confidence_drop_fraction": CONFIDENCE_DROP_FRACTION,
            "minibatch": minibatch_audit,
        },
        "selected_model_alpha": selected_alpha,
        "model_mix_candidates": mix_candidates,
        "minimum_base_cosine": float(base_cosines.min()),
        "mean_base_cosine": float(base_cosines.mean()),
        "base_validation": base_validation,
        "selected_validation": selected_validation,
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
        "selected_model_alpha": selected_alpha,
        "minimum_base_cosine": float(base_cosines.min()),
        "base_validation": base_validation,
        "selected_validation": selected_validation,
        "validation_decision": decision,
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("GLM_SEMANTIC_FINETUNE_LARGE=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
