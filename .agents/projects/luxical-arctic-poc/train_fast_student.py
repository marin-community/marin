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
from dataclasses import asdict, dataclass
from enum import StrEnum
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
from fast_student import MAX_TOKENS, FastStudent
from fast_student_training_data import (
    MaterializedTrainingRows,
    SourceTrainingRows,
    StagedTrainingRows,
    stage_training_rows,
)
from ladder_config import MANIFEST_ROOT, SEED, read_json, write_json
from luxical.training import dequantize_8bit_uniform_scalar_quantized
from rigging.filesystem import atomic_rename

from experiments.datakit.cluster.quality.fast_transformer.embedding import (
    embedding_distillation_loss,
    predict_embeddings,
    projected_embedding_distillation_loss,
    source_conditioned_geometry_loss,
)
from experiments.datakit.cluster.quality.fast_transformer.inference import data_parallel_shardings
from experiments.datakit.cluster.quality.fast_transformer.model import FastEmbeddingTransformer, count_params

PREPARED_MANIFEST_URL = f"{MANIFEST_ROOT}/fast-student/prepared-3m/manifest.json"
OUTPUT_ROOT = f"{MANIFEST_ROOT}/fast-student"
RUNG_TARGETS = {"64k": 65_536, "750k": 750_000, "3m": 3_000_000, "10m": 10_000_000, "30m": 30_000_000}
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
AUDIT_ROWS = 2_048
STAGING_CHUNK_ROWS = BATCH_SIZE
TRAINING_BLOCK_ROWS = 16 * BATCH_SIZE
MAXIMUM_SOURCE_ROWS_IN_MEMORY = 262_144
RESULT_FILE = Path("/tmp/luxical-fast-student-train")


@dataclass(frozen=True)
class TeacherSpec:
    """Define one aligned training teacher."""

    dimension: int
    source_root: str | None
    audit_url: str | None
    artifact_suffix: str | None


class TrainingLayout(StrEnum):
    """Select the host-memory layout for training rows."""

    MATERIALIZED = "materialized"
    STAGED = "staged"


TEACHERS = {
    "arctic-medium-256": TeacherSpec(
        dimension=256,
        source_root=None,
        audit_url=None,
        artifact_suffix=None,
    ),
    "qwen3-embedding-0.6b-1024": TeacherSpec(
        dimension=1_024,
        source_root=f"{MANIFEST_ROOT}/teacher-qwen3-embedding-0.6b-1024-train-750k-v1/sources",
        audit_url=f"{MANIFEST_ROOT}/teacher-qwen3-embedding-0.6b-1024-train-750k-v1/audit.json",
        artifact_suffix="qwen3-06b-1024-crossdim",
    ),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def load_numpy(url: str) -> np.ndarray:
    """Load one NumPy array from private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path, "rb") as file:
        return np.load(file)


def load_training_arrays(
    prepared: dict[str, Any], target: int, teacher_spec: TeacherSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Load one exact source-balanced rung from the prepared 3M arrays."""
    capacities = {source: int(result["rows"]) for source, result in prepared["sources"].items()}
    quotas = allocate_balanced_quotas(capacities, target)
    id_chunks = []
    teacher_chunks = []
    source_id_chunks = []
    for source_id, (source, result) in enumerate(sorted(prepared["sources"].items())):
        filesystem, path = fsspec.core.url_to_fs(result["output_url"])
        prepared_table = pq.read_table(
            path,
            filesystem=filesystem,
            columns=["raw_sha256", "train_rank", "ids", "embedding"],
            filters=[("train_rank", "<", quotas[source])],
        ).sort_by("train_rank")
        if len(prepared_table) != quotas[source]:
            raise ValueError(f"Prepared source {source} returned {len(prepared_table)} rows; expected {quotas[source]}")
        if teacher_spec.source_root is None:
            teacher_table = prepared_table
        else:
            teacher_url = f"{teacher_spec.source_root}/{Path(result['output_url']).name}"
            teacher_filesystem, teacher_path = fsspec.core.url_to_fs(teacher_url)
            teacher_table = pq.read_table(
                teacher_path,
                filesystem=teacher_filesystem,
                columns=["raw_sha256", "train_rank", "embedding"],
                filters=[("train_rank", "<", quotas[source])],
            ).sort_by("train_rank")
            if len(teacher_table) != quotas[source]:
                raise ValueError(
                    f"Teacher source {source} returned {len(teacher_table)} rows; expected {quotas[source]}"
                )
            if prepared_table["raw_sha256"].to_pylist() != teacher_table["raw_sha256"].to_pylist():
                raise ValueError(f"Teacher hashes are not aligned for {source}")
            if prepared_table["train_rank"].to_pylist() != teacher_table["train_rank"].to_pylist():
                raise ValueError(f"Teacher ranks are not aligned for {source}")
        ids = prepared_table["ids"].combine_chunks()
        embeddings = teacher_table["embedding"].combine_chunks()
        id_chunks.append(ids.values.to_numpy(zero_copy_only=False).reshape(len(prepared_table), ids.type.list_size))
        quantized = embeddings.values.to_numpy(zero_copy_only=False).reshape(len(prepared_table), teacher_spec.dimension)
        teacher_chunks.append(dequantize_8bit_uniform_scalar_quantized(quantized, TEACHER_QUANTIZATION_LIMIT))
        source_id_chunks.append(np.full(len(prepared_table), source_id, dtype=np.int32))
        logger.info(
            "Loaded source %d/%d: %s (%d rows)",
            source_id + 1,
            len(quotas),
            source,
            len(prepared_table),
        )
    all_ids = np.concatenate(id_chunks).astype(np.int32, copy=False)
    all_teacher = np.concatenate(teacher_chunks).astype(np.float32, copy=False)
    all_source_ids = np.concatenate(source_id_chunks)
    all_teacher /= np.linalg.norm(all_teacher, axis=1, keepdims=True).clip(min=1e-12)
    if len(all_ids) != target or all_teacher.shape != (target, teacher_spec.dimension):
        raise ValueError(f"Loaded array shapes do not match target: {all_ids.shape}, {all_teacher.shape}")
    if not np.isfinite(all_teacher).all():
        raise ValueError("Teacher arrays contain non-finite values")
    return all_ids, all_teacher, all_source_ids, quotas


def staged_sources(
    prepared: dict[str, Any],
    quotas: dict[str, int],
    teacher_spec: TeacherSpec,
) -> list[SourceTrainingRows]:
    """Return ordered source slices for disk staging."""
    sources = []
    for source_id, (source, result) in enumerate(sorted(prepared["sources"].items())):
        teacher_url = None
        if teacher_spec.source_root is not None:
            teacher_url = f"{teacher_spec.source_root}/{Path(result['output_url']).name}"
        sources.append(
            SourceTrainingRows(
                source=source,
                source_id=source_id,
                prepared_url=result["output_url"],
                teacher_url=teacher_url,
                rows=quotas[source],
            )
        )
    return sources


def training_rows(
    prepared: dict[str, Any],
    target: int,
    teacher_spec: TeacherSpec,
    layout: TrainingLayout,
    staging_directory: Path,
) -> tuple[MaterializedTrainingRows | StagedTrainingRows, dict[str, int]]:
    """Load the original arrays or stage bounded disk-backed arrays."""
    if layout == TrainingLayout.MATERIALIZED:
        ids, teacher, source_ids, quotas = load_training_arrays(prepared, target, teacher_spec)
        return MaterializedTrainingRows(ids, teacher, source_ids), quotas
    capacities = {source: int(result["rows"]) for source, result in prepared["sources"].items()}
    quotas = allocate_balanced_quotas(capacities, target)
    staged = stage_training_rows(
        staged_sources(prepared, quotas, teacher_spec),
        staging_directory,
        id_width=MAX_TOKENS,
        teacher_dimension=teacher_spec.dimension,
        teacher_quantization_limit=TEACHER_QUANTIZATION_LIMIT,
        chunk_rows=STAGING_CHUNK_ROWS,
        maximum_source_rows=MAXIMUM_SOURCE_ROWS_IN_MEMORY,
        seed=SEED,
    )
    if staged.rows != target:
        raise ValueError(f"Staged {staged.rows} rows; expected {target}")
    return staged, quotas


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


def upload_array(values: np.ndarray, url: str) -> str:
    filesystem, path = fsspec.core.url_to_fs(url)
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_path = Path(temporary_directory) / "array.npy"
        np.save(local_path, values)
        digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
        with atomic_rename(path, fs=filesystem) as temporary_path:
            filesystem.put(str(local_path), temporary_path)
    return digest


def initial_projection(student_dimension: int, teacher_dimension: int) -> jax.Array | None:
    """Return a train-only cross-dimension projection when dimensions differ."""
    if student_dimension == teacher_dimension:
        return None
    scale = math.sqrt(2.0 / (student_dimension + teacher_dimension))
    return jax.random.normal(jax.random.PRNGKey(SEED + 2), (student_dimension, teacher_dimension)) * scale


def weight_decay_mask(model: FastEmbeddingTransformer, projection: jax.Array | None):
    """Apply AdamW decay to model arrays, but not to the scale-invariant alignment head."""
    filtered_model = eqx.filter(model, eqx.is_inexact_array)
    model_mask = jax.tree.map(
        lambda value: value is not None,
        filtered_model,
        is_leaf=lambda value: value is None,
    )
    return model_mask, False if projection is not None else None


def train(
    model: FastEmbeddingTransformer,
    rows: MaterializedTrainingRows | StagedTrainingRows,
    source_geometry_weight: float,
    output_root: str,
    teacher_dimension: int,
) -> tuple[FastEmbeddingTransformer, jax.Array | None, list[dict[str, Any]]]:
    """Train with the Luxical Gram-KL objective and save each epoch."""
    device_count, replicated, batch_sharding = data_parallel_shardings()
    if BATCH_SIZE % device_count:
        raise ValueError(f"Batch size {BATCH_SIZE} is not divisible by {device_count} devices")
    steps_per_epoch = math.ceil(rows.rows / BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=LEARNING_RATE * 0.05,
        peak_value=LEARNING_RATE,
        warmup_steps=max(1, int(total_steps * WARMUP_FRACTION)),
        decay_steps=total_steps,
        end_value=LEARNING_RATE * 0.05,
    )
    model = jax.device_put(model, replicated)
    projection = initial_projection(model.output_dim, teacher_dimension)
    if projection is not None:
        projection = jax.device_put(projection, replicated)
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP),
        optax.adamw(schedule, weight_decay=WEIGHT_DECAY, mask=weight_decay_mask(model, projection)),
    )
    parameters = (model, projection)
    optimizer_state = jax.device_put(optimizer.init(eqx.filter(parameters, eqx.is_inexact_array)), replicated)

    @eqx.filter_jit
    def step(current_parameters, current_optimizer_state, batch_ids, batch_teacher, batch_source_ids, key):
        def loss_function(candidate_parameters):
            candidate_model, candidate_projection = candidate_parameters
            prediction = candidate_model(batch_ids, key=key, inference=False)
            if candidate_projection is None:
                distillation = embedding_distillation_loss(
                    prediction,
                    batch_teacher,
                    LOSS_TEMPERATURE,
                    DIRECT_COSINE_WEIGHT,
                )
            else:
                distillation = projected_embedding_distillation_loss(
                    prediction,
                    batch_teacher,
                    candidate_projection,
                    LOSS_TEMPERATURE,
                    DIRECT_COSINE_WEIGHT,
                )
            source_geometry = (
                source_conditioned_geometry_loss(prediction, batch_teacher, batch_source_ids)
                if source_geometry_weight
                else jnp.asarray(0.0)
            )
            return distillation + source_geometry_weight * source_geometry, (distillation, source_geometry)

        (loss, components), gradients = eqx.filter_value_and_grad(loss_function, has_aux=True)(current_parameters)
        updates, next_optimizer_state = optimizer.update(
            gradients,
            current_optimizer_state,
            eqx.filter(current_parameters, eqx.is_inexact_array),
        )
        return eqx.apply_updates(current_parameters, updates), next_optimizer_state, loss, components

    key = jax.random.PRNGKey(SEED + 1)
    audit_ids = rows.audit_ids(AUDIT_ROWS)
    history = []
    global_step = 0
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        epoch_losses = []
        epoch_distillation_losses = []
        epoch_source_geometry_losses = []
        for batch in rows.epoch_batches(epoch, BATCH_SIZE, TRAINING_BLOCK_ROWS, SEED):
            key, step_key = jax.random.split(key)
            batch_ids = jax.device_put(jnp.asarray(batch.ids), batch_sharding)
            batch_teacher = jax.device_put(jnp.asarray(batch.teacher), batch_sharding)
            batch_source_ids = jax.device_put(jnp.asarray(batch.source_ids), batch_sharding)
            parameters, optimizer_state, loss, components = step(
                parameters,
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
        if len(epoch_losses) != steps_per_epoch:
            raise ValueError(f"Epoch {epoch + 1} returned {len(epoch_losses)} batches; expected {steps_per_epoch}")
        model, projection = parameters
        audit = embedding_audit(model, audit_ids)
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
        projection_url = None
        projection_sha256 = None
        if projection is not None:
            projection_url = f"{output_root}/alignment-epoch-{epoch + 1}.npy"
            projection_sha256 = upload_array(np.asarray(projection), projection_url)
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
            "alignment_url": projection_url,
            "alignment_sha256": projection_sha256,
            "embedding_audit": audit,
        }
        history.append(epoch_result)
        write_json(f"{output_root}/training-epoch-{epoch + 1}.json", epoch_result)
        logger.info("Completed epoch %d: %s", epoch + 1, json.dumps(epoch_result, sort_keys=True))
    return model, projection, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", choices=tuple(RUNG_TARGETS), required=True)
    parser.add_argument("--config", choices=("full", "slim"), required=True)
    parser.add_argument("--treatment", choices=tuple(SOURCE_GEOMETRY_WEIGHTS), default="baseline")
    parser.add_argument("--teacher", choices=tuple(TEACHERS), required=True)
    parser.add_argument("--training-layout", choices=tuple(TrainingLayout), type=TrainingLayout, required=True)
    parser.add_argument("--prepared-manifest-url", default=PREPARED_MANIFEST_URL)
    return parser.parse_args()


def training_name(config_name: str, treatment: str, teacher_spec: TeacherSpec) -> str:
    """Return the artifact name for one training configuration."""
    parts = [config_name]
    if teacher_spec.artifact_suffix is not None:
        parts.append(teacher_spec.artifact_suffix)
    if treatment != "baseline":
        parts.append(treatment)
    return "-".join(parts)


def main() -> None:
    """Train and save one fixed fast-student ladder rung."""
    arguments = parse_args()
    if jax.default_backend() != "gpu":
        raise ValueError(f"Fast student training requires a GPU backend, got {jax.default_backend()}")
    target = RUNG_TARGETS[arguments.rung]
    prepared = read_json(arguments.prepared_manifest_url)
    teacher_spec = TEACHERS[arguments.teacher]
    if teacher_spec.audit_url is not None:
        audit = read_json(teacher_spec.audit_url)
        if audit["manifest_sha256"] != prepared["manifest_sha256"]:
            raise ValueError("The teacher audit has a different manifest digest")
        if audit["teacher_dimension"] != teacher_spec.dimension:
            raise ValueError("The teacher audit has a different embedding dimension")
        if audit["row_count"] < target:
            raise ValueError(f"The teacher audit has {audit['row_count']} rows; requested {target}")
    if arguments.treatment != "baseline" and teacher_spec.dimension != 256:
        raise ValueError("Source-geometry treatments require equal student and teacher dimensions")
    raw_to_compact = load_numpy(prepared["raw_to_compact_url"])
    with tempfile.TemporaryDirectory(prefix="luxical-fast-student-training-") as staging_directory:
        rows, quotas = training_rows(
            prepared,
            target,
            teacher_spec,
            arguments.training_layout,
            Path(staging_directory),
        )
        memory_report = rows.memory_report(TRAINING_BLOCK_ROWS)
        student = FastStudent.random(arguments.config, raw_to_compact, seed=SEED)
        artifact_name = training_name(arguments.config, arguments.treatment, teacher_spec)
        source_geometry_weight = SOURCE_GEOMETRY_WEIGHTS[arguments.treatment]
        output_root = f"{OUTPUT_ROOT}/{artifact_name}/{arguments.rung}"
        model, projection, history = train(
            student.model,
            rows,
            source_geometry_weight,
            output_root,
            teacher_spec.dimension,
        )
    report = {
        "rung": arguments.rung,
        "config_name": arguments.config,
        "training_name": artifact_name,
        "treatment": arguments.treatment,
        "teacher": {"name": arguments.teacher, **asdict(teacher_spec)},
        "training_layout": arguments.training_layout,
        "training_memory": memory_report,
        "training_rows": target,
        "source_quotas": quotas,
        "prepared_manifest_url": arguments.prepared_manifest_url,
        "prepared_manifest_sha256": prepared["manifest_sha256"],
        "raw_to_compact_url": prepared["raw_to_compact_url"],
        "raw_to_compact_sha256": prepared["raw_to_compact_sha256"],
        "config": asdict(model.backbone.config),
        "parameters": count_params(model),
        "training_alignment_parameters": int(projection.size) if projection is not None else 0,
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
        "final_alignment_url": history[-1]["alignment_url"],
        "final_alignment_sha256": history[-1]["alignment_sha256"],
    }
    report_url = f"{output_root}/training.json"
    write_json(report_url, report)
    summary = {
        "rung": arguments.rung,
        "config_name": arguments.config,
        "training_name": artifact_name,
        "treatment": arguments.treatment,
        "teacher": arguments.teacher,
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
