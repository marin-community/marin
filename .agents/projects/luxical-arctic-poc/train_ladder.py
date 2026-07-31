# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train one Arctic-distilled Luxical ladder rung."""

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from ladder_config import (
    MANIFEST_ROOT,
    SEED,
    TEACHER_ID,
    TEACHER_REVISION,
    TRAIN_TARGET_3M,
    TRAIN_TARGET_750K,
)
from luxical.csr_matrix_utils import csr_matrix_to_torch
from luxical.embedder import Embedder
from luxical.sparse_to_dense_neural_nets import SparseToDenseEmbedder
from luxical.training import contrastive_distillation_loss, dequantize_8bit_uniform_scalar_quantized, equal_beta_adamw
from rigging.filesystem import atomic_rename

MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
TEACHER_ROOT = f"{MANIFEST_ROOT}/teacher-arctic-v1"
TEACHER_AUDIT_URL = f"{TEACHER_ROOT}/audit.json"
OUTPUT_ROOT = f"{MANIFEST_ROOT}/students"
BASELINE_REPO = "DatologyAI/luxical-one"
BASELINE_FILE = "luxical_one_rc4.npz"
BASELINE_REVISION = "474cfeb959dd473b3d1cd61da630f566037e69e2"
EXPECTED_LAYER_SHAPES = (
    (96, 2_000_000),
    (3_072, 96),
    (3_072, 3_072),
    (192, 3_072),
)
TRAIN_BATCH_SIZE = 12_288
NUM_EPOCHS = 3
LOSS_TEMPERATURE = 3.0
LEARNING_RATE = 1e-2
WARMUP_FRACTION = 0.05
DECAY_FRACTION = 0.1
OPTIMIZER_BETA = 0.9
OPTIMIZER_EPSILON = 1e-8
TEACHER_QUANTIZATION_LIMIT = 0.3
TEACHER_EMBEDDING_DIMENSION = 256
RESULT_FILE = Path("/tmp/luxical-arctic-train-rung")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def read_json(url: str) -> dict[str, Any]:
    """Read one JSON object from private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path) as file:
        return json.load(file)


def selected_mask(table: pa.Table, rung_column: str) -> pa.Array | pa.ChunkedArray:
    """Return the training selection for one ladder rung."""
    return pc.and_(pc.equal(table["split"], "train"), table[rung_column])


def teacher_output_url(manifest_output_url: str) -> str:
    """Return the teacher file paired with one manifest source file."""
    return f"{TEACHER_ROOT}/sources/{Path(manifest_output_url).name}"


def load_training_arrays(
    manifest: dict[str, Any],
    rung_column: str,
) -> tuple[pa.ChunkedArray, pa.ChunkedArray]:
    """Load aligned text and quantized teacher arrays."""
    text_chunks = []
    embedding_chunks = []
    loaded_rows = 0
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Loading source %d/%d: %s", index, len(manifest["sources"]), source)
        manifest_filesystem, manifest_path = fsspec.core.url_to_fs(result["output_url"])
        source_table = pq.read_table(
            manifest_path,
            filesystem=manifest_filesystem,
            columns=["raw_sha256", "split", rung_column, "text"],
        )
        source_table = source_table.filter(selected_mask(source_table, rung_column))

        teacher_url = teacher_output_url(result["output_url"])
        teacher_filesystem, teacher_path = fsspec.core.url_to_fs(teacher_url)
        if not teacher_filesystem.exists(teacher_path):
            raise FileNotFoundError(f"Missing teacher output: {teacher_url}")
        teacher_table = pq.read_table(
            teacher_path,
            filesystem=teacher_filesystem,
            columns=["raw_sha256", "split", rung_column, "embedding"],
        )
        teacher_table = teacher_table.filter(selected_mask(teacher_table, rung_column))
        if len(source_table) != len(teacher_table):
            raise ValueError(f"Source and teacher row counts differ for {source}")
        if source_table["raw_sha256"].to_pylist() != teacher_table["raw_sha256"].to_pylist():
            raise ValueError(f"Source and teacher rows are not aligned for {source}")

        text_chunks.extend(source_table["text"].chunks)
        embedding_chunks.extend(teacher_table["embedding"].chunks)
        loaded_rows += len(source_table)
    logger.info("Loaded %d aligned training rows", loaded_rows)
    texts = pc.cast(pa.chunked_array(text_chunks), pa.large_string())
    return texts, pa.chunked_array(embedding_chunks)


def new_student() -> Embedder:
    """Create a random student with the exact Luxical-One runtime structure."""
    baseline_path = hf_hub_download(
        repo_id=BASELINE_REPO,
        filename=BASELINE_FILE,
        revision=BASELINE_REVISION,
    )
    baseline = Embedder.load(baseline_path)
    layer_shapes = tuple(layer.shape for layer in baseline.bow_to_dense_embedder.layers)
    if layer_shapes != EXPECTED_LAYER_SHAPES:
        raise ValueError(f"Unexpected Luxical-One layer shapes: {layer_shapes}")
    dimensions = (layer_shapes[0][1], *(shape[0] for shape in layer_shapes))
    random_network = SparseToDenseEmbedder.create(
        dims=dimensions,
        seed=SEED,
    )
    return baseline.replace_sparse_to_dense_embedder(random_network)


def learning_rate_scale(step: int, total_steps: int) -> float:
    """Return a warmup, stable, and decay learning-rate scale."""
    warmup_steps = max(1, int(total_steps * WARMUP_FRACTION))
    decay_steps = max(1, int(total_steps * DECAY_FRACTION))
    if step <= warmup_steps:
        return step / warmup_steps
    if step <= total_steps - decay_steps:
        return 1.0
    return max(0.0, (total_steps - step) / decay_steps)


def arrow_teacher_batch(
    embeddings: pa.ChunkedArray,
    indices: np.ndarray[Any, np.dtype[np.int64]],
) -> np.ndarray:
    """Take and dequantize one teacher batch."""
    selected = pc.take(embeddings, pa.array(indices)).combine_chunks()
    quantized = selected.values.to_numpy(zero_copy_only=False).reshape(
        len(indices),
        TEACHER_EMBEDDING_DIMENSION,
    )
    teacher = dequantize_8bit_uniform_scalar_quantized(
        quantized,
        TEACHER_QUANTIZATION_LIMIT,
    )
    teacher /= np.linalg.norm(teacher, axis=1, keepdims=True).clip(min=1e-12)
    if not np.isfinite(teacher).all():
        raise ValueError("A dequantized teacher batch contains non-finite values")
    return teacher


def train_student(
    student: Embedder,
    texts: pa.ChunkedArray,
    teacher_embeddings: pa.ChunkedArray,
) -> tuple[Embedder, list[float]]:
    """Train the sparse-to-dense network with global row shuffling."""
    if len(texts) != len(teacher_embeddings):
        raise ValueError("Text and teacher row counts differ")
    device = torch.device("cuda")
    module = student.bow_to_dense_embedder.to_torch(device=device)
    optimizer = equal_beta_adamw(
        module.parameters(),
        lr=LEARNING_RATE,
        beta=OPTIMIZER_BETA,
        eps=OPTIMIZER_EPSILON,
        weight_decay=0.0,
    )
    steps_per_epoch = math.ceil(len(texts) / TRAIN_BATCH_SIZE)
    total_steps = NUM_EPOCHS * steps_per_epoch
    rng = np.random.default_rng(SEED)
    losses = []
    step = 0
    for epoch in range(NUM_EPOCHS):
        epoch_indices = rng.permutation(len(texts))
        for start in range(0, len(texts), TRAIN_BATCH_SIZE):
            indices = epoch_indices[start : start + TRAIN_BATCH_SIZE]
            batch_texts = pc.take(texts, pa.array(indices)).to_pylist()
            bow = student.bow_from_texts(batch_texts)
            tfidf = csr_matrix_to_torch(student.tfidf_from_bow(bow)).to(device)
            target = torch.from_numpy(arrow_teacher_batch(teacher_embeddings, indices)).to(device)
            prediction = module(tfidf)
            loss = contrastive_distillation_loss(prediction, target, LOSS_TEMPERATURE)
            if not torch.isfinite(loss):
                raise ValueError(f"Training loss is non-finite at step {step + 1}")
            loss.backward()
            step += 1
            current_learning_rate = LEARNING_RATE * learning_rate_scale(step, total_steps)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
            logger.info(
                "Epoch %d step %d/%d loss %.6f learning_rate %.6g",
                epoch + 1,
                step,
                total_steps,
                losses[-1],
                current_learning_rate,
            )
    trained_network = SparseToDenseEmbedder.from_torch(module)
    return student.replace_sparse_to_dense_embedder(trained_network), losses


def upload_file(local_path: Path, url: str) -> None:
    """Upload one local file atomically."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        filesystem.put(str(local_path), temporary_path)


def write_json(url: str, value: dict[str, Any]) -> None:
    """Write one JSON object atomically."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(value, file, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", choices=("750k", "3m"), required=True)
    return parser.parse_args()


def main() -> None:
    """Train and save one fixed ladder rung."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    torch.manual_seed(SEED)
    arguments = parse_args()
    rung_column = "in_750k" if arguments.rung == "750k" else "in_3m"
    target_rows = TRAIN_TARGET_750K if arguments.rung == "750k" else TRAIN_TARGET_3M
    manifest = read_json(MANIFEST_URL)
    teacher_audit = read_json(TEACHER_AUDIT_URL)
    if teacher_audit["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("The teacher audit has a different manifest digest")
    if teacher_audit["teacher_id"] != TEACHER_ID:
        raise ValueError(f"The teacher audit has teacher {teacher_audit['teacher_id']}")
    if teacher_audit["teacher_revision"] != TEACHER_REVISION:
        raise ValueError(f"The teacher audit has revision {teacher_audit['teacher_revision']}")
    texts, teacher_embeddings = load_training_arrays(manifest, rung_column)
    if len(texts) != target_rows:
        raise ValueError(f"Loaded {len(texts)} rows for target {target_rows}")
    student = new_student()
    student, losses = train_student(student, texts, teacher_embeddings)

    model_url = f"{OUTPUT_ROOT}/{arguments.rung}/luxical-arctic.npz"
    report_url = f"{OUTPUT_ROOT}/{arguments.rung}/training.json"
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_model_path = Path(temporary_directory) / "luxical-arctic.npz"
        student.save(local_model_path)
        with local_model_path.open("rb") as model_file:
            model_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
        upload_file(local_model_path, model_url)
    report = {
        "rung": arguments.rung,
        "training_rows": len(texts),
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "teacher_audit_url": TEACHER_AUDIT_URL,
        "teacher_id": TEACHER_ID,
        "teacher_revision": TEACHER_REVISION,
        "model_url": model_url,
        "model_sha256": model_sha256,
        "baseline_repo": BASELINE_REPO,
        "baseline_file": BASELINE_FILE,
        "baseline_revision": BASELINE_REVISION,
        "layer_shapes": EXPECTED_LAYER_SHAPES,
        "batch_size": TRAIN_BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "loss_temperature": LOSS_TEMPERATURE,
        "learning_rate": LEARNING_RATE,
        "warmup_fraction": WARMUP_FRACTION,
        "decay_fraction": DECAY_FRACTION,
        "optimizer_beta": OPTIMIZER_BETA,
        "optimizer_epsilon": OPTIMIZER_EPSILON,
        "weight_decay": 0.0,
        "steps": len(losses),
        "first_loss": losses[0],
        "final_loss": losses[-1],
    }
    write_json(report_url, report)
    summary = {
        "rung": arguments.rung,
        "training_rows": len(texts),
        "model_url": model_url,
        "model_sha256": model_sha256,
        "report_url": report_url,
        "steps": len(losses),
        "first_loss": losses[0],
        "final_loss": losses[-1],
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("LUXICAL_ARCTIC_TRAIN_RUNG=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
