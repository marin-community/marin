# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a trained export to the actual masters, not to the untrained weights."""

import hashlib
import json
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from levanter.compat.fsspec_safetensor import TensorRecord
from levanter.compat.hf_export import run_on_export_writer, write_export_json
from levanter.trainer_state import TrainerState
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.identity_context_distill.precision_source_audit import (
    CHUNK_BYTES,
    check_export_auxiliary_metadata,
    check_precision_layout,
    check_source_metadata,
    checkpoint_records,
    open_audit_stream,
)
from experiments.post_training.identity_context_distill.sft import check_export_metadata


def compare_saved_master(
    replicated: jax.Array, record: TensorRecord, source_master_sha256: str, *, frozen_bias: bool
) -> dict:
    """Writer-only comparison of a local replica to every saved tensor byte in chunks."""
    host = np.asarray(replicated.addressable_data(0), order="C")
    if host.dtype != np.float32 or tuple(host.shape) != record.shape:
        raise ValueError(f"Trained master shape or dtype differs: {record.key}")
    master_sha = hashlib.sha256(memoryview(host).cast("B")).hexdigest()
    if frozen_bias and master_sha != source_master_sha256:
        raise ValueError(f"Frozen router master changed: {record.key}")
    flat = host.reshape(-1)
    chunk_elements = CHUNK_BYTES // record.dtype.itemsize
    digest = hashlib.sha256()
    with open_audit_stream(record.file_path) as stream:
        stream.seek(record.byte_start)
        for start in range(0, flat.size, chunk_elements):
            master_chunk = flat[start : start + chunk_elements]
            expected = master_chunk.astype(record.dtype)
            if not np.isfinite(master_chunk).all() or not np.isfinite(expected).all():
                raise ValueError(f"Nonfinite trained/exported tensor: {record.key}")
            payload = stream.read(expected.nbytes)
            if payload != expected.tobytes(order="C"):
                raise ValueError(f"Saved tensor differs from trained master cast: {record.key}")
            digest.update(payload)
    return {"master_sha256": master_sha, "export_sha256": digest.hexdigest(), "frozen_bias": frozen_bias}


def audit_trained_export(
    state: TrainerState,
    export_path: str,
    *,
    source_root: str,
    source_receipt_path: str,
    source_receipt_sha256: str,
    expected_hashes: dict[str, str],
) -> None:
    """Collectively prove BF16/F32 export equality after the native milestone save.

    All ranks enter in matching order. Only process zero reads export bytes, from
    one fully replicated master at a time. The source receipt is the same pinned
    full-source receipt used by the mandatory initialized-master audit.
    """
    tensors = state.model.to_state_dict()

    def prepare():
        check_source_metadata(source_root, expected_hashes)
        receipt_bytes = StoragePath(source_receipt_path).read_bytes()
        if hashlib.sha256(receipt_bytes).hexdigest() != source_receipt_sha256:
            raise ValueError("Source audit receipt hash differs from the launch pin")
        source = json.loads(receipt_bytes)["tensors"]
        exported = checkpoint_records(export_path)
        check_precision_layout(exported)
        if len(tensors) != 502 or tensors.keys() != source.keys() or tensors.keys() != exported.keys():
            raise ValueError("Trained/source/exported tensor keysets differ")
        for name, tensor in tensors.items():
            if tensor.dtype != jnp.float32 or list(tensor.shape) != source[name]["shape"]:
                raise ValueError(f"Trained master shape or dtype differs: {name}")
        original, saved = StoragePath(source_root), StoragePath(export_path)
        check_export_metadata(
            json.loads((original / "config.json").read_bytes()), json.loads((saved / "config.json").read_bytes())
        )
        metadata_additions = check_export_auxiliary_metadata(source_root, export_path)
        return source, exported, metadata_additions

    prepared = run_on_export_writer(prepare)
    results = {}
    for name in sorted(tensors):
        replicated = jax.sharding.reshard(tensors[name], P())
        replicated.block_until_ready()

        def compare(name, replicated):
            assert prepared is not None
            source, exported, _ = prepared
            results[name] = compare_saved_master(
                replicated,
                exported[name],
                source[name]["master_sha256"],
                frozen_bias=name.endswith(".mlp.router.bias"),
            )

        run_on_export_writer(partial(compare, name, replicated))
        del replicated
    run_on_export_writer(
        lambda: write_export_json(
            StoragePath(export_path) / "trained-precision-audit.json",
            {
                "completed_updates": int(state.step),
                "tensor_count": len(results),
                "source_receipt": source_receipt_path,
                "source_receipt_sha256": source_receipt_sha256,
                "source_metadata_sha256": expected_hashes,
                "accepted_metadata_additions": prepared[2],
                "comparison": "476 BF16 saved tensors equal casts of trained masters; 26 FP32 biases equal source",
                "tensors": results,
            },
        )
    )
