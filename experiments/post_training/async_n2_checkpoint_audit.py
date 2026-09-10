# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact byte comparison of a native saved cohort and finalized partition."""

import hashlib

import torch

from experiments.post_training.async_n2_native_audit import _equal


def audit_saved_pending_partition(cohort, actual_partition):
    """Compare the actual native-loaded checkpoint-seven cohort with update eight.

    The caller must bind the source checkpoint and finalized dump byte hashes,
    and load the cohort through the frozen native BufferCheckpointCallback. The
    fresh update-eight dump or the separately released continuation dump can be
    checked; this never compares independent live asynchronous model endpoints.
    """
    geometry = (
        cohort.admission_step,
        cohort.next_update,
        cohort.dp_size,
        cohort.mini_batch_groups,
        cohort.samples_per_prompt,
        len(cohort.groups),
    )
    _equal(geometry, (7, 1, 4, 64, 4, 128), "saved checkpoint-seven cohort geometry")
    expected = cohort.partition(8)
    _equal(set(actual_partition), set(expected), "saved/finalized tensor fields")
    tensor_hashes = {}
    for key in expected:
        left, right = expected[key], actual_partition[key]
        if left is None:
            _equal(right, None, "absent saved tensor")
            continue
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise ValueError("Unexpected finalized batch field type")
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError(f"Saved pending partition tensor changed: {key}")
        if (left.is_floating_point() or left.is_complex()) and not torch.isfinite(left).all():
            raise ValueError(f"Nonfinite saved pending tensor: {key}")
        raw = left.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        actual_raw = right.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        if raw != actual_raw:
            raise ValueError(f"Saved pending partition bytes changed: {key}")
        tensor_hashes[key] = dict(
            saved_sha256=hashlib.sha256(raw).hexdigest(),
            actual_sha256=hashlib.sha256(actual_raw).hexdigest(),
            bytes_equal=True,
            shape=list(left.shape),
            dtype=str(left.dtype),
        )
    for key in ("uids", "async_cohort_admission_step", "async_cohort_update_index", "async_cohort_admission_ages"):
        _equal(actual_partition.metadata[key], expected.metadata[key], f"saved partition metadata {key}")
    uids = expected.metadata["uids"]
    _equal(len(uids), 256, "saved pending response rows")
    _equal(len(set(uids)), 64, "saved pending source groups")
    for start in range(0, 256, 4):
        _equal(uids[start : start + 4], [uids[start]] * 4, "contiguous response group")
    return dict(
        status="ASYNC_N2_SAVED_PENDING_PARTITION_PASS",
        saved_step=7,
        consumed_step=8,
        cohort_groups=128,
        pending_groups=64,
        dp_size=4,
        tensor_hashes=tensor_hashes,
        source_uids=uids[::4],
        scope="exact saved old-logprob/advantage/source partition; not independent end-model equality",
    )
