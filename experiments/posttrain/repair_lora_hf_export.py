# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repair broken LoRA-derived merged HF exports by transposing corrupted weights.

This is a one-off recovery tool for historical HF exports affected by the old
LoRA merge-axis bug. It does not require retraining or raw trainer checkpoints.
Instead, it reads the broken merged HF export, identifies tensors whose
transpose matches the config-derived expected shape, writes repaired shards, and
copies the rest of the HF metadata to a new output directory.
"""

import argparse
import logging
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass

import fsspec
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from experiments.posttrain.lora_vllm_investigate import (
    LlamaShapeConfig,
    _copy_if_exists,
    _expected_shape,
    _read_json,
    run_shape_check,
    sync_chat_template_into_tokenizer_config,
)

logger = logging.getLogger(__name__)

_ANCILLARY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)

_INDEX_FILE = "model.safetensors.index.json"
_COPY_BUFFER_SIZE = 16 * 1024 * 1024


@dataclass(frozen=True)
class RepairStats:
    shard_count: int
    tensor_count: int
    transposed_count: int
    unchanged_count: int


def _iter_shard_names(index_payload: dict) -> Iterable[str]:
    seen = set()
    for shard_name in index_payload["weight_map"].values():
        if shard_name in seen:
            continue
        seen.add(shard_name)
        yield shard_name


def _fs_exists(path: str) -> bool:
    fs, plain_path = fsspec.core.url_to_fs(path)
    return fs.exists(plain_path)


def _ensure_remote_dir(path: str) -> None:
    fs, plain_path = fsspec.core.url_to_fs(path)
    fs.makedirs(plain_path, exist_ok=True)


def _download_to_local(source_path: str, local_path: str) -> None:
    fs, plain_path = fsspec.core.url_to_fs(source_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    fs.get(plain_path, local_path)


def _upload_from_local(local_path: str, destination_path: str) -> None:
    fs, plain_path = fsspec.core.url_to_fs(destination_path)
    parent = os.path.dirname(plain_path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    fs.put(local_path, plain_path)


def _repair_local_shard(
    local_source_path: str,
    local_output_path: str,
    *,
    shape_config: LlamaShapeConfig,
) -> tuple[int, int, int]:
    repaired_tensors: dict[str, np.ndarray] = {}
    tensor_count = 0
    transposed_count = 0
    unchanged_count = 0
    fatal_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    with safe_open(local_source_path, framework="np") as handle:
        for tensor_name in handle.keys():
            tensor = handle.get_tensor(tensor_name)
            tensor_count += 1
            expected_shape = _expected_shape(shape_config, tensor_name)
            if expected_shape is None:
                repaired_tensors[tensor_name] = tensor
                unchanged_count += 1
                continue

            actual_shape = tuple(int(x) for x in tensor.shape)
            if actual_shape == expected_shape:
                repaired_tensors[tensor_name] = tensor
                unchanged_count += 1
                continue

            if tensor.ndim == 2 and tuple(int(x) for x in tensor.T.shape) == expected_shape:
                repaired_tensors[tensor_name] = np.ascontiguousarray(tensor.T)
                transposed_count += 1
                logger.info(
                    "Transposed tensor %s from %s to %s",
                    tensor_name,
                    actual_shape,
                    expected_shape,
                )
                continue

            fatal_mismatches.append((tensor_name, expected_shape, actual_shape))

    if fatal_mismatches:
        for tensor_name, expected_shape, actual_shape in fatal_mismatches:
            logger.error(
                "Unrepairable tensor mismatch: %s expected=%s actual=%s",
                tensor_name,
                expected_shape,
                actual_shape,
            )
        raise RuntimeError(f"Found {len(fatal_mismatches)} unrepairable tensor mismatches in {local_source_path}")

    save_file(repaired_tensors, local_output_path)
    return tensor_count, transposed_count, unchanged_count


def run_repair(*, source_model_path: str, output_path: str, verify_shapes: bool) -> RepairStats:
    if _fs_exists(output_path):
        raise ValueError(f"Output path already exists: {output_path}")

    config_payload = _read_json(f"{source_model_path}/config.json")
    index_payload = _read_json(f"{source_model_path}/{_INDEX_FILE}")
    shape_config = LlamaShapeConfig.from_json(config_payload)

    _ensure_remote_dir(output_path)

    total_tensors = 0
    total_transposed = 0
    total_unchanged = 0
    shard_names = list(_iter_shard_names(index_payload))

    with tempfile.TemporaryDirectory(prefix="repair_lora_hf_export_") as temp_dir:
        for shard_name in shard_names:
            source_shard_path = f"{source_model_path}/{shard_name}"
            local_source_path = os.path.join(temp_dir, shard_name)
            local_output_path = os.path.join(temp_dir, f"fixed-{shard_name}")

            logger.info("Downloading shard %s", source_shard_path)
            _download_to_local(source_shard_path, local_source_path)

            tensor_count, transposed_count, unchanged_count = _repair_local_shard(
                local_source_path,
                local_output_path,
                shape_config=shape_config,
            )
            total_tensors += tensor_count
            total_transposed += transposed_count
            total_unchanged += unchanged_count

            destination_shard_path = f"{output_path}/{shard_name}"
            logger.info("Uploading repaired shard to %s", destination_shard_path)
            _upload_from_local(local_output_path, destination_shard_path)

        for filename in _ANCILLARY_FILES:
            copied = _copy_if_exists(f"{source_model_path}/{filename}", f"{output_path}/{filename}")
            if copied:
                logger.info("Copied %s", filename)

        _copy_if_exists(f"{source_model_path}/{_INDEX_FILE}", f"{output_path}/{_INDEX_FILE}")
        logger.info("Copied %s", _INDEX_FILE)
        sync_chat_template_into_tokenizer_config(output_path)

    stats = RepairStats(
        shard_count=len(shard_names),
        tensor_count=total_tensors,
        transposed_count=total_transposed,
        unchanged_count=total_unchanged,
    )

    logger.info(
        "Repair complete: shards=%d tensors=%d transposed=%d unchanged=%d",
        stats.shard_count,
        stats.tensor_count,
        stats.transposed_count,
        stats.unchanged_count,
    )

    if verify_shapes:
        logger.info("Running post-repair shape audit for %s", output_path)
        run_shape_check(output_path)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-path", required=True, help="Broken HF export path to repair.")
    parser.add_argument("--output-path", required=True, help="Destination path for the repaired HF export.")
    parser.add_argument(
        "--skip-verify-shapes",
        action="store_true",
        help="Skip the post-repair config-derived shape audit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    run_repair(
        source_model_path=args.source_model_path,
        output_path=args.output_path,
        verify_shapes=not args.skip_verify_shapes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
