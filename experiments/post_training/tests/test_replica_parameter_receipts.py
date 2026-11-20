# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from dataclasses import replace

import torch

from experiments.post_training.long_loop.replica_parameter_receipts import (
    ParameterReceipt,
    parameter_sha256,
    replica_differences,
    replica_iteration_differences,
)


def test_parameter_receipt_keeps_bf16_bytes_and_signed_zero():
    bits = torch.tensor([0, -32768, 16256, 32704], dtype=torch.int16)
    parameter = bits.view(torch.bfloat16)
    expected = hashlib.sha256(bits.numpy().tobytes()).hexdigest()
    assert parameter_sha256(parameter, chunk_bytes=3) == expected
    changed = parameter.clone()
    changed[1] = 0
    assert parameter_sha256(changed) != expected


def test_replica_comparison_localizes_expert_group_and_missing_member():
    dense = ParameterReceipt(0, "norm", (2,), "torch.float32", (0, 1), ("model.norm",), 8, "same")
    expert = ParameterReceipt(0, "expert0", (2, 2), "torch.bfloat16", (0, 2), ("hf.expert0",), 8, "same")
    records = [dense, replace(dense, rank=1), expert, replace(expert, rank=2, sha256="changed")]
    differences = replica_differences(records)
    assert [(row["name"], row["group"]) for row in differences] == [("expert0", (0, 2))]
    assert [row["rank"] for row in differences[0]["members"]] == [0, 2]
    missing = replica_differences(records[:3])
    assert [(row["name"], row["group"]) for row in missing] == [("expert0", (0, 2))]


def test_matching_hashes_do_not_hide_wrong_expert_assignment():
    parameter = ParameterReceipt(0, "fc1", (2, 2), "torch.bfloat16", (0, 1), ("expert0",), 8, "same")
    differences = replica_differences([parameter, replace(parameter, rank=1, hf_names=("expert64",))])
    assert differences[0]["name"] == "fc1"
    assert len(differences[0]["members"]) == 2


def test_replica_collective_order_detects_reordered_equal_parameters():
    first = ParameterReceipt(0, "first", (2,), "torch.float32", (0, 1), ("first",), 8, "same")
    second = replace(first, name="second", hf_names=("second",))
    rows = [first, second, replace(first, rank=1), replace(second, rank=1)]
    assert replica_differences(rows) == []
    differences = replica_iteration_differences(rows, {0: ["first", "second"], 1: ["second", "first"]})
    assert differences == [{"group": (0, 1), "source_orders": {0: ["first", "second"], 1: ["second", "first"]}}]
    assert replica_iteration_differences(rows, {0: ["first", "second"], 1: ["first", "second"]}) == []
