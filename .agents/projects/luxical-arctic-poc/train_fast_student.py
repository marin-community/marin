# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train one source-balanced fast-transformer embedding student rung."""

import argparse
import hashlib
import json
import logging
import math
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq
from build_manifest import allocate_balanced_quotas
from fast_student import FastStudent
from ladder_config import MANIFEST_ROOT, SEED
from luxical.training import dequantize_8bit_uniform_scalar_quantized
from rigging.filesystem import atomic_rename

from experiments.datakit.cluster.quality.fast_transformer.embedding import (
    embedding_distillation_loss,
    predict_embeddings,
    source_conditioned_geometry_loss,
)
from experiments.datakit.cluster.quality.fast_transformer.inference import data_parallel_shardings
from experiments.datakit.cluster.quality.fast_transformer.model import FastEmbeddingTransformer, count_params

PREPARED_MANIFEST_URL = f"{MANIFEST_ROOT}/fast-student/prepared-3m/manifest.json"
OUTPUT_ROOT = f"{MANIFEST_ROOT}/fast-student"
RUNG_TARGETS = {"64k": 65_536, "750k": 750_000, "3m": 3_000_000}
BATCH_SIZE = 4_096
EPOCHS = 3
LEARNING_RATE = 5e-4
WARMUP_FRACTION = 0.05
WEIGHT_DECAY = 0.05
GRADIENT_CLIP = 1.0
LOSS_TEMPERATURE = 3.0
DIRECT_COSINE_WEIGHT = 1.0
SOURCE_GEOMETRY_WEIGHTS = {
    "baseline": 0.0,
    "source-geometry-w0.25": 0.25,
    "source-geometry-w0.5": 0.5,
    "source-geometry-w1": 1.0,
}
TEACHER_QUANTIZATION_LIMIT = 0.3
TEACHER_DIMENSION = 256
AUDIT_ROWS = 2_048
RESULT_FILE = Path("/tmp/luxical-fast-student-train")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def read_json(url: str) -> dict[str, Any]:
    """Read one JSON object from private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path) as file:
        return json.load(file)


def write_json(url: str, value: dict[str, Any]) -> None:
    """Write one JSON object atomically."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(value, file, indent=2, sort_keys=True)


def load_numpy(url: str) -> np.ndarray:
    """Load one NumPy array from private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path, "rb") as file:
        return np.load(file)


def load_training_arrays(
    prepared: dict[str, Any], target: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Load one exact source-balanced rung from the prepared 3M arrays."""
    capacities = {source: int(result["rows"]) for source, result in prepared["sources"].items()}
    quotas = allocate_balanced_quotas(capacities, target)
    id_chunks = []
    teacher_chunks = []
    source_id_chunks = []
    for source_id, (source, result) in enumerate(sorted(prepared["sources"].items())):
        filesystem, path = fsspec.core.url_to_fs(result["output_url"])
        table = pq.read_table(
            path,
            filesystem=filesystem,
            columns=["train_rank", "ids", "embedding"],
            filters=[("train_rank", "<", quotas[source])],
        )
        if len(table) != quotas[source]:
            raise ValueError(f"Prepared source {source} returned {len(table)} rows; expected {quotas[source]}")
        ids = table["ids"].combine_chunks()
        embeddings = table["embedding"].combine_chunks()
        id_chunks.append(ids.values.to_numpy(zero_copy_only=False).reshape(len(table), ids.type.list_size))
        quantized = embeddings.values.to_numpy(zero_copy_only=False).reshape(len(table), TEACHER_DIMENSION)
        teacher_chunks.append(dequantize_8bit_uniform_scalar_quantized(quantized, TEACHER_QUANTIZATION_LIMIT))
        source_id_chunks.append(np.full(len(table), source_id, dtype=np.int32))
        logger.info("Loaded source %d/%d: %s (%d rows)", source_id + 1, len(quotas), source, len(table))
    all_ids = np.concatenate(id_chunks).astype(np.int32, copy=False)
    all_teacher = np.concatenate(teacher_chunks).astype(np.float32, copy=False)
    all_source_ids = np.concatenate(source_id_chunks)
    all_teacher /= np.linalg.norm(all_teacher, axis=1, keepdims=True).clip(min=1e-12)
    if len(all_ids) != target or all_teacher.shape != (target, TEACHER_DIMENSION):
        raise ValueError(f"Loaded array shapes do not match target: {all_ids.shape}, {all_teacher.shape}")
    if not np.isfinite(all_teacher).all():
        raise ValueError("Teacher arrays contain non-finite values")
    return all_ids, all_teacher, all_source_ids, quotas


def embedding_audit(model: FastEmbeddingTransformer, ids: np.ndarray) -> dict[str, float]:
    """Measure finite, uniqueness, variance, and concentration checks."""
    vectors = predict_embeddings(model, ids[:AUDIT_ROWS], batch_size=AUDIT_ROWS)
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    cosine = normalized @ normalized.T
    off_diagonal = cosine[~np.eye(len(cosine), dtype=bool)]
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = np.square(singular_values) / max(1, len(vectors) - 1)
    probabilities = eigenvalues / max(float(eigenvalues.sum()), 1e-30)
    positive = probabilities > 0
    return {
        "finite_fraction": float(np.isfinite(vectors).mean()),
        "unique_fraction_6dp": float(len(np.unique(np.round(vectors, 6), axis=0)) / len(vectors)),
        "median_dimension_variance": float(np.median(np.var(vectors, axis=0))),
        "minimum_dimension_variance": float(np.min(np.var(vectors, axis=0))),
        "cosine_mean": float(np.mean(off_diagonal)),
        "cosine_standard_deviation": float(np.std(off_diagonal)),
        "cosine_p99": float(np.quantile(off_diagonal, 0.99)),
        "total_variance": float(eigenvalues.sum()),
        "effective_rank": float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))),
    }


def upload_model(model: FastEmbeddingTransformer, url: str) -> str:
    """Serialize and upload one Equinox model atomically."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_path = Path(temporary_directory) / "model.eqx"
        eqx.tree_serialise_leaves(local_path, model)
        digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
        with atomic_rename(path, fs=filesystem) as temporary_path:
            filesystem.put(str(local_path), temporary_path)
    return digest


def train(
    model: FastEmbeddingTransformer,
    ids: np.ndarray,
    teacher: np.ndarray,
    source_ids: np.ndarray,
    source_geometry_weight: float,
    output_root: str,
) -> tuple[FastEmbeddingTransformer, list[dict[str, Any]]]:
    """Train with the Luxical Gram-KL objective and save each epoch."""
    device_count, replicated, batch_sharding = data_parallel_shardings()
    if BATCH_SIZE % device_count:
        raise ValueError(f"Batch size {BATCH_SIZE} is not divisible by {device_count} devices")
    if len(source_ids) != len(ids):
        raise ValueError(f"Source rows {len(source_ids)} do not match input rows {len(ids)}")
    steps_per_epoch = math.ceil(len(ids) / BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=LEARNING_RATE * 0.05,
        peak_value=LEARNING_RATE,
        warmup_steps=max(1, int(total_steps * WARMUP_FRACTION)),
        decay_steps=total_steps,
        end_value=LEARNING_RATE * 0.05,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP),
        optax.adamw(schedule, weight_decay=WEIGHT_DECAY),
    )
    model = jax.device_put(model, replicated)
    optimizer_state = jax.device_put(optimizer.init(eqx.filter(model, eqx.is_inexact_array)), replicated)

    @eqx.filter_jit
    def step(current_model, current_optimizer_state, batch_ids, batch_teacher, batch_source_ids, key):
        def loss_function(candidate):
            prediction = candidate(batch_ids, key=key, inference=False)
            distillation = embedding_distillation_loss(
                prediction,
                batch_teacher,
                LOSS_TEMPERATURE,
                DIRECT_COSINE_WEIGHT,
            )
            source_geometry = (
                source_conditioned_geometry_loss(prediction, batch_teacher, batch_source_ids)
                if source_geometry_weight
                else jnp.asarray(0.0)
            )
            return distillation + source_geometry_weight * source_geometry, (distillation, source_geometry)

        (loss, components), gradients = eqx.filter_value_and_grad(loss_function, has_aux=True)(current_model)
        updates, next_optimizer_state = optimizer.update(
            gradients,
            current_optimizer_state,
            eqx.filter(current_model, eqx.is_inexact_array),
        )
        return eqx.apply_updates(current_model, updates), next_optimizer_state, loss, components

    rng = np.random.default_rng(SEED)
    key = jax.random.PRNGKey(SEED + 1)
    history = []
    global_step = 0
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        permutation = rng.permutation(len(ids))
        padded_rows = steps_per_epoch * BATCH_SIZE - len(permutation)
        if padded_rows:
            permutation = np.concatenate([permutation, permutation[:padded_rows]])
        epoch_losses = []
        epoch_distillation_losses = []
        epoch_source_geometry_losses = []
        for batch_index in range(steps_per_epoch):
            selected = permutation[batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE]
            key, step_key = jax.random.split(key)
            batch_ids = jax.device_put(jnp.asarray(ids[selected]), batch_sharding)
            batch_teacher = jax.device_put(jnp.asarray(teacher[selected]), batch_sharding)
            batch_source_ids = jax.device_put(jnp.asarray(source_ids[selected]), batch_sharding)
            model, optimizer_state, loss, components = step(
                model,
                optimizer_state,
                batch_ids,
                batch_teacher,
                batch_source_ids,
                step_key,
            )
            loss_value = float(loss)
            distillation_loss = float(components[0])
            source_geometry_loss = float(components[1])
            if not np.isfinite(loss_value):
                raise ValueError(f"Training loss is non-finite at step {global_step + 1}")
            epoch_losses.append(loss_value)
            epoch_distillation_losses.append(distillation_loss)
            epoch_source_geometry_losses.append(source_geometry_loss)
            global_step += 1
            if global_step == 1 or global_step % 10 == 0:
                logger.info("Epoch %d step %d/%d loss %.8f", epoch + 1, global_step, total_steps, loss_value)
        audit = embedding_audit(model, ids)
        if (
            audit["finite_fraction"] != 1.0
            or audit["unique_fraction_6dp"] < 0.99
            or audit["total_variance"] <= 1e-6
            or audit["effective_rank"] < 2.0
            or audit["cosine_standard_deviation"] <= 1e-4
            or audit["cosine_p99"] >= 0.9999
        ):
            raise ValueError(f"Embedding audit failed after epoch {epoch + 1}: {audit}")
        model_url = f"{output_root}/model-epoch-{epoch + 1}.eqx"
        model_sha256 = upload_model(model, model_url)
        epoch_result = {
            "epoch": epoch + 1,
            "steps": global_step,
            "first_loss": epoch_losses[0],
            "final_loss": epoch_losses[-1],
            "mean_loss": float(np.mean(epoch_losses)),
            "final_distillation_loss": epoch_distillation_losses[-1],
            "mean_distillation_loss": float(np.mean(epoch_distillation_losses)),
            "final_source_geometry_loss": epoch_source_geometry_losses[-1],
            "mean_source_geometry_loss": float(np.mean(epoch_source_geometry_losses)),
            "elapsed_seconds": time.perf_counter() - started,
            "model_url": model_url,
            "model_sha256": model_sha256,
            "embedding_audit": audit,
        }
        history.append(epoch_result)
        write_json(f"{output_root}/training-epoch-{epoch + 1}.json", epoch_result)
        logger.info("Completed epoch %d: %s", epoch + 1, json.dumps(epoch_result, sort_keys=True))
    return model, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", choices=tuple(RUNG_TARGETS), required=True)
    parser.add_argument("--config", choices=("full", "slim"), required=True)
    parser.add_argument("--treatment", choices=tuple(SOURCE_GEOMETRY_WEIGHTS), default="baseline")
    return parser.parse_args()


def main() -> None:
    """Train and save one fixed fast-student ladder rung."""
    arguments = parse_args()
    if jax.default_backend() != "gpu":
        raise ValueError(f"Fast student training requires a GPU backend, got {jax.default_backend()}")
    target = RUNG_TARGETS[arguments.rung]
    prepared = read_json(PREPARED_MANIFEST_URL)
    raw_to_compact = load_numpy(prepared["raw_to_compact_url"])
    ids, teacher, source_ids, quotas = load_training_arrays(prepared, target)
    student = FastStudent.random(arguments.config, raw_to_compact, seed=SEED)
    training_name = (
        arguments.config if arguments.treatment == "baseline" else f"{arguments.config}-{arguments.treatment}"
    )
    source_geometry_weight = SOURCE_GEOMETRY_WEIGHTS[arguments.treatment]
    output_root = f"{OUTPUT_ROOT}/{training_name}/{arguments.rung}"
    model, history = train(
        student.model,
        ids,
        teacher,
        source_ids,
        source_geometry_weight,
        output_root,
    )
    report = {
        "rung": arguments.rung,
        "config_name": arguments.config,
        "training_name": training_name,
        "treatment": arguments.treatment,
        "training_rows": target,
        "source_quotas": quotas,
        "prepared_manifest_url": PREPARED_MANIFEST_URL,
        "prepared_manifest_sha256": prepared["manifest_sha256"],
        "raw_to_compact_url": prepared["raw_to_compact_url"],
        "raw_to_compact_sha256": prepared["raw_to_compact_sha256"],
        "config": asdict(model.backbone.config),
        "parameters": count_params(model),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "loss_temperature": LOSS_TEMPERATURE,
        "direct_cosine_weight": DIRECT_COSINE_WEIGHT,
        "source_geometry_weight": source_geometry_weight,
        "learning_rate": LEARNING_RATE,
        "warmup_fraction": WARMUP_FRACTION,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "history": history,
        "final_model_url": history[-1]["model_url"],
        "final_model_sha256": history[-1]["model_sha256"],
    }
    report_url = f"{output_root}/training.json"
    write_json(report_url, report)
    summary = {
        "rung": arguments.rung,
        "config_name": arguments.config,
        "training_name": training_name,
        "treatment": arguments.treatment,
        "training_rows": target,
        "report_url": report_url,
        "model_url": report["final_model_url"],
        "model_sha256": report["final_model_sha256"],
        "first_loss": history[0]["first_loss"],
        "final_loss": history[-1]["final_loss"],
        "embedding_audit": history[-1]["embedding_audit"],
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("FAST_STUDENT_TRAIN=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
