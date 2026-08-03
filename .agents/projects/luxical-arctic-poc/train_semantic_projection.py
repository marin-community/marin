# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit and fold a GLM-supervised linear head into the 30M student."""

import json
import logging
import math
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from evaluate_fast_student import load_student
from evaluate_hierarchical_embeddings import hierarchical_assignments
from evaluate_semantic_embeddings import semantic_sample
from fast_student import FastStudent
from glm_hierarchical_labels import OUTPUT_ROOT
from glm_semantic_labels import stable_order
from ladder_config import SEED, write_json
from semantic_embedding_metrics import normalize_embeddings
from sklearn.metrics import f1_score
from train_fast_student import OUTPUT_ROOT as TRAINING_ROOT
from train_fast_student import upload_array, upload_model

from experiments.datakit.cluster.quality.fast_transformer.embedding import embedding_spread_loss
from experiments.datakit.cluster.quality.fast_transformer.model import FastEmbeddingTransformer, count_params

BASE_TRAINING_NAME = "full"
BASE_RUNG = "30m"
TRAINING_NAME = "full-glm-semantic-projection"
RUNG = "pilot-1k-mix-v2"
HIERARCHY_RUN_ID = "hierarchy-1000-20260802-002"
HIERARCHY_VARIANT = "compact"
CONFIDENCE_DROP_FRACTION = 0.05
VALIDATION_FRACTION = 0.20
STEPS = 500
LEARNING_RATE = 1e-3
TEMPERATURE = 0.1
PARENT_WEIGHT = 0.5
LEAF_WEIGHT = 1.0
FORM_WEIGHT = 0.25
IDENTITY_WEIGHT = 2.0
ORTHOGONAL_WEIGHT = 1.0
SPREAD_WEIGHT = 1.0
SPREAD_STANDARD_DEVIATION_TARGET = 0.04
SPREAD_COVARIANCE_WEIGHT = 0.1
MINIMUM_VALIDATION_SEMANTIC_DELTA = 0.005
MAXIMUM_VALIDATION_LEVEL_LOSS = 0.01
MINIMUM_EFFECTIVE_RANK_FRACTION = 0.25
MINIMUM_TOTAL_VARIANCE = 0.50
MINIMUM_FOLDED_COSINE = 0.999
PROJECTION_MIX_ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
RESULT_FILE = Path("/tmp/luxical-semantic-projection")

logger = logging.getLogger(__name__)


class SemanticLabels(NamedTuple):
    """Store integer hierarchy labels for one document table."""

    parent: jax.Array | np.ndarray
    leaf: jax.Array | np.ndarray
    form: jax.Array | np.ndarray


@dataclass(frozen=True)
class ProjectionMixCandidate:
    """Store one identity-mix validation result."""

    alpha: float
    metrics: dict[str, Any]
    provisional_decision: dict[str, Any]


def integer_labels(values: list[str]) -> np.ndarray:
    """Return stable integer IDs for one string-label column."""
    ids = {value: index for index, value in enumerate(sorted(set(values)))}
    return np.asarray([ids[value] for value in values], dtype=np.int32)


def retained_train_validation_indices(
    confidences: np.ndarray,
    leaf_labels: list[str],
    validation_fraction: float = VALIDATION_FRACTION,
    drop_fraction: float = CONFIDENCE_DROP_FRACTION,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Drop the least-confident tail and return a leaf-stratified split."""
    if confidences.ndim != 1 or len(confidences) != len(leaf_labels):
        raise ValueError("Confidence and leaf-label rows differ")
    if not 0 <= drop_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions are invalid")
    drop_count = int(np.floor(len(confidences) * drop_fraction))
    ranked = sorted(range(len(confidences)), key=lambda index: (confidences[index], stable_order(str(index))))
    dropped = set(ranked[:drop_count])
    retained = [index for index in range(len(confidences)) if index not in dropped]
    cutoff = float(min(confidences[index] for index in retained))
    by_leaf: dict[str, list[int]] = defaultdict(list)
    for index in retained:
        by_leaf[leaf_labels[index]].append(index)
    validation = []
    for label in sorted(by_leaf):
        rows = sorted(by_leaf[label], key=lambda index: stable_order(str(index)))
        validation_count = max(1, round(len(rows) * validation_fraction)) if len(rows) >= 5 else 0
        validation.extend(rows[:validation_count])
    validation_set = set(validation)
    training = [index for index in retained if index not in validation_set]
    if not training or not validation:
        raise ValueError("The semantic split has an empty side")
    return np.asarray(sorted(training)), np.asarray(sorted(validation)), cutoff


def supervised_contrastive_loss(
    vectors: jax.Array,
    labels: jax.Array,
    source_ids: jax.Array,
    temperature: float,
) -> jax.Array:
    """Pull equal-label rows together across sources."""
    if vectors.ndim != 2 or labels.shape != (vectors.shape[0],) or source_ids.shape != labels.shape:
        raise ValueError("Semantic vectors, labels, and sources are not aligned")
    if temperature <= 0:
        raise ValueError("The semantic temperature must be positive")
    vectors = vectors / jnp.maximum(jnp.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    cross_source = source_ids[:, None] != source_ids[None, :]
    positives = cross_source & (labels[:, None] == labels[None, :])
    logits = vectors @ vectors.T / temperature
    masked_logits = jnp.where(cross_source, logits, -1e9)
    log_probabilities = masked_logits - jax.nn.logsumexp(masked_logits, axis=1, keepdims=True)
    positive_counts = jnp.sum(positives, axis=1)
    row_losses = -jnp.sum(jnp.where(positives, log_probabilities, 0.0), axis=1) / jnp.maximum(positive_counts, 1)
    valid_rows = positive_counts > 0
    return jnp.sum(jnp.where(valid_rows, row_losses, 0.0)) / jnp.maximum(jnp.sum(valid_rows), 1)


def projection_loss(
    projection: jax.Array,
    base_vectors: jax.Array,
    labels: SemanticLabels,
    source_ids: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return the fixed hierarchy, anchor, and spread objective."""
    projected = base_vectors @ projection
    projected /= jnp.maximum(jnp.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
    parent = supervised_contrastive_loss(projected, jnp.asarray(labels.parent), source_ids, TEMPERATURE)
    leaf = supervised_contrastive_loss(projected, jnp.asarray(labels.leaf), source_ids, TEMPERATURE)
    form = supervised_contrastive_loss(projected, jnp.asarray(labels.form), source_ids, TEMPERATURE)
    identity = jnp.mean(1.0 - jnp.sum(projected * base_vectors, axis=1))
    identity_matrix = jnp.eye(projection.shape[0], dtype=projection.dtype)
    orthogonal = jnp.mean(jnp.square(projection.T @ projection - identity_matrix))
    spread = embedding_spread_loss(
        projected,
        SPREAD_STANDARD_DEVIATION_TARGET,
        SPREAD_COVARIANCE_WEIGHT,
    )
    semantic = PARENT_WEIGHT * parent + LEAF_WEIGHT * leaf + FORM_WEIGHT * form
    total = semantic + IDENTITY_WEIGHT * identity + ORTHOGONAL_WEIGHT * orthogonal + SPREAD_WEIGHT * spread
    return total, {
        "parent": parent,
        "leaf": leaf,
        "form": form,
        "semantic": semantic,
        "identity": identity,
        "orthogonal": orthogonal,
        "spread": spread,
    }


def geometry_metrics(vectors: np.ndarray) -> dict[str, float]:
    """Return finite, unique, variance, and effective-rank metrics."""
    vectors = normalize_embeddings(vectors)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(vectors) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    probabilities = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    positive = probabilities > 0
    effective_rank = float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))
    return {
        "finite_fraction": float(np.isfinite(vectors).all(axis=1).mean()),
        "unique_fraction_4dp": float(len(np.unique(np.round(vectors, 4), axis=0)) / len(vectors)),
        "total_variance": float(np.var(vectors, axis=0).sum()),
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank / vectors.shape[1],
    }


def cross_source_macro_f1(vectors: np.ndarray, labels: np.ndarray, sources: np.ndarray) -> float:
    """Return nearest-label macro-F1 with same-source rows excluded."""
    vectors = normalize_embeddings(vectors)
    similarity = vectors @ vectors.T
    allowed = sources[:, None] != sources[None, :]
    if not np.all(allowed.any(axis=1)):
        raise ValueError("A semantic row has no cross-source neighbor candidate")
    similarity = np.where(allowed, similarity, -np.inf)
    neighbor_labels = labels[np.argmax(similarity, axis=1)]
    return float(f1_score(labels, neighbor_labels, average="macro", zero_division=0))


def evaluation_metrics(
    vectors: np.ndarray,
    labels: SemanticLabels,
    sources: np.ndarray,
) -> dict[str, Any]:
    """Return pilot semantic and vector-health metrics."""
    return {
        "geometry": geometry_metrics(vectors),
        "parent_macro_f1": cross_source_macro_f1(vectors, labels.parent, sources),
        "leaf_macro_f1": cross_source_macro_f1(vectors, labels.leaf, sources),
        "form_macro_f1": cross_source_macro_f1(vectors, labels.form, sources),
    }


def validation_decision(base: dict[str, Any], projected: dict[str, Any], folded_cosine_minimum: float) -> dict[str, Any]:
    """Return the predeclared pilot decision and its inputs."""
    levels = ("parent_macro_f1", "leaf_macro_f1", "form_macro_f1")
    deltas = {level: float(projected[level] - base[level]) for level in levels}
    semantic_mean_delta = float(np.mean(list(deltas.values())))
    geometry = projected["geometry"]
    gates = {
        "semantic_mean_delta": semantic_mean_delta >= MINIMUM_VALIDATION_SEMANTIC_DELTA,
        "no_level_regression": min(deltas.values()) >= -MAXIMUM_VALIDATION_LEVEL_LOSS,
        "finite": float(geometry["finite_fraction"]) == 1.0,
        "unique": float(geometry["unique_fraction_4dp"]) >= 0.99,
        "effective_rank_fraction": float(geometry["effective_rank_fraction"]) >= MINIMUM_EFFECTIVE_RANK_FRACTION,
        "total_variance": float(geometry["total_variance"]) >= MINIMUM_TOTAL_VARIANCE,
        "folded_parity": folded_cosine_minimum >= MINIMUM_FOLDED_COSINE,
    }
    return {
        "semantic_deltas": deltas,
        "semantic_mean_delta": semantic_mean_delta,
        "folded_cosine_minimum": folded_cosine_minimum,
        "gates": gates,
        "passed": all(gates.values()),
    }


def fold_embedding_projection(
    model: FastEmbeddingTransformer,
    projection: np.ndarray,
) -> FastEmbeddingTransformer:
    """Fold one square output projection into an embedding head."""
    if projection.shape != (model.output_dim, model.output_dim):
        raise ValueError("The semantic projection shape does not match the embedding output")
    folded_head = model.embedding_head @ jnp.asarray(projection)
    return eqx.tree_at(lambda candidate: candidate.embedding_head, model, folded_head)


def projection_mix(projection: np.ndarray, alpha: float) -> np.ndarray:
    """Shrink one learned projection toward identity."""
    if projection.ndim != 2 or projection.shape[0] != projection.shape[1]:
        raise ValueError("The semantic projection must be square")
    if not 0 <= alpha <= 1:
        raise ValueError("The semantic projection mix must be from zero through one")
    identity = np.eye(projection.shape[0], dtype=projection.dtype)
    return identity + alpha * (projection - identity)


def select_projection_mix(
    base_vectors: np.ndarray,
    raw_projection: np.ndarray,
    validation_indices: np.ndarray,
    validation_labels: SemanticLabels,
    sources: np.ndarray,
    base_validation: dict[str, Any],
) -> tuple[float, np.ndarray, list[dict[str, Any]]]:
    """Select the best pilot mix that passes the provisional gates."""
    candidates = []
    validation_vectors = base_vectors[validation_indices]
    validation_sources = sources[validation_indices]
    for alpha in PROJECTION_MIX_ALPHAS:
        candidate_projection = projection_mix(raw_projection, alpha)
        candidate_vectors = normalize_embeddings(validation_vectors @ candidate_projection)
        metrics = evaluation_metrics(
            candidate_vectors,
            validation_labels,
            validation_sources,
        )
        decision = validation_decision(base_validation, metrics, folded_cosine_minimum=1.0)
        candidates.append(ProjectionMixCandidate(alpha, metrics, decision))
    passed = [row for row in candidates if row.provisional_decision["passed"]]
    selection_pool = passed if passed else candidates
    selected = max(selection_pool, key=lambda row: row.provisional_decision["semantic_mean_delta"])
    selected_alpha = selected.alpha
    return selected_alpha, projection_mix(raw_projection, selected_alpha), [asdict(row) for row in candidates]


def fit_projection(
    base_vectors: np.ndarray,
    labels: SemanticLabels,
    source_ids: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Fit the fixed identity-started projection."""
    dimension = base_vectors.shape[1]
    projection = jnp.eye(dimension)
    base = jnp.asarray(normalize_embeddings(base_vectors))
    sources = jnp.asarray(source_ids)
    optimizer = optax.adam(LEARNING_RATE)
    state = optimizer.init(projection)

    @jax.jit
    def step(current_projection: jax.Array, current_state: optax.OptState):
        (loss, components), gradient = jax.value_and_grad(projection_loss, has_aux=True)(
            current_projection,
            base,
            labels,
            sources,
        )
        updates, next_state = optimizer.update(gradient, current_state, current_projection)
        return optax.apply_updates(current_projection, updates), next_state, loss, components

    history = []
    for step_index in range(1, STEPS + 1):
        projection, state, loss, components = step(projection, state)
        if step_index == 1 or step_index % 50 == 0:
            row = {"step": float(step_index), "loss": float(loss)}
            row.update({name: float(value) for name, value in components.items()})
            history.append(row)
            logger.info("Projection step %d/%d: %s", step_index, STEPS, json.dumps(row, sort_keys=True))
    result = np.asarray(projection)
    if not np.isfinite(result).all():
        raise ValueError("The semantic projection contains non-finite values")
    return result, history


def fit_projection_minibatches(
    base_vectors: np.ndarray,
    labels: SemanticLabels,
    source_ids: np.ndarray,
    batch_size: int,
    epochs: int,
) -> tuple[np.ndarray, list[dict[str, float]], dict[str, int]]:
    """Fit an identity-started projection with deterministic minibatches."""
    rows, dimension = base_vectors.shape
    if rows < 1 or batch_size < 1 or epochs < 1:
        raise ValueError("Projection rows, batch size, and epochs must be positive")
    if labels.parent.shape != (rows,) or labels.leaf.shape != (rows,) or labels.form.shape != (rows,):
        raise ValueError("Projection labels do not match the base-vector rows")
    if source_ids.shape != (rows,):
        raise ValueError("Projection sources do not match the base-vector rows")

    projection = jnp.eye(dimension)
    base = jnp.asarray(normalize_embeddings(base_vectors))
    device_labels = SemanticLabels(
        parent=jnp.asarray(labels.parent),
        leaf=jnp.asarray(labels.leaf),
        form=jnp.asarray(labels.form),
    )
    sources = jnp.asarray(source_ids)
    optimizer = optax.adam(LEARNING_RATE)
    state = optimizer.init(projection)

    @jax.jit
    def step(
        current_projection: jax.Array,
        current_state: optax.OptState,
        all_vectors: jax.Array,
        all_labels: SemanticLabels,
        all_sources: jax.Array,
        indices: jax.Array,
    ):
        batch_labels = SemanticLabels(
            parent=all_labels.parent[indices],
            leaf=all_labels.leaf[indices],
            form=all_labels.form[indices],
        )
        (loss, components), gradient = jax.value_and_grad(projection_loss, has_aux=True)(
            current_projection,
            all_vectors[indices],
            batch_labels,
            all_sources[indices],
        )
        updates, next_state = optimizer.update(gradient, current_state, current_projection)
        return optax.apply_updates(current_projection, updates), next_state, loss, components

    batches_per_epoch = math.ceil(rows / batch_size)
    padded_rows_per_epoch = batches_per_epoch * batch_size
    generator = np.random.default_rng(SEED + 700_000)
    history = []
    update_index = 0
    for epoch_index in range(epochs):
        order = generator.permutation(rows)
        padded_order = np.resize(order, padded_rows_per_epoch)
        for start in range(0, padded_rows_per_epoch, batch_size):
            update_index += 1
            indices = jnp.asarray(padded_order[start : start + batch_size])
            projection, state, loss, components = step(
                projection,
                state,
                base,
                device_labels,
                sources,
                indices,
            )
            if update_index == 1 or update_index % 50 == 0 or update_index == batches_per_epoch * epochs:
                row = {
                    "step": float(update_index),
                    "epoch": float(epoch_index + 1),
                    "loss": float(loss),
                }
                row.update({name: float(value) for name, value in components.items()})
                history.append(row)
                logger.info(
                    "Projection update %d/%d: %s",
                    update_index,
                    batches_per_epoch * epochs,
                    json.dumps(row, sort_keys=True),
                )
    result = np.asarray(projection)
    if not np.isfinite(result).all():
        raise ValueError("The minibatch semantic projection contains non-finite values")
    audit = {
        "rows": rows,
        "batch_size": batch_size,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "padded_rows_per_epoch": padded_rows_per_epoch,
        "updates": update_index,
    }
    return result, history, audit


def main() -> None:
    """Fit, fold, upload, and report one semantic projection."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    documents, _, _ = semantic_sample()
    assignment_root = OUTPUT_ROOT / HIERARCHY_RUN_ID / HIERARCHY_VARIANT
    assignments = hierarchical_assignments(assignment_root, documents)
    confidences = np.asarray([row.confidence for row in assignments])
    leaf_names = [row.primary_leaf_id for row in assignments]
    training_indices, validation_indices, confidence_cutoff = retained_train_validation_indices(
        confidences,
        leaf_names,
    )
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
        raw_projection, history = fit_projection(
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
        "hierarchy_run_id": HIERARCHY_RUN_ID,
        "hierarchy_variant": HIERARCHY_VARIANT,
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
            "steps": STEPS,
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
        "base_validation": base_validation,
        "projected_validation": projected_validation,
        "validation_decision": decision,
        "selected_projection_alpha": selected_alpha,
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("GLM_SEMANTIC_PROJECTION=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
