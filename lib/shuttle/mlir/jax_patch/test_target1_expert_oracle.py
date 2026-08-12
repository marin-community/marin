# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the Target 1 expert-oracle contract."""

import copy
import json
from pathlib import Path

import pytest
from target1_expert_oracle import load_contract, validate_contract

CONTRACT = Path(__file__).with_name("target1-rowwise-bf16-te-2.17-expert-oracle-v1.json")


def _document() -> dict:
    return json.loads(CONTRACT.read_text())


def _mutated(path: tuple[str | int, ...], value: object) -> dict:
    document = copy.deepcopy(_document())
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return document


def test_contract_closes_oracle_calls_without_claiming_hardware_results():
    contract = load_contract(CONTRACT)

    assert contract["dispatch"] == {
        "key": "boundary",
        "accepted_values": ["forward", "backward_recompute", "composed"],
        "workload_name_dispatch": False,
    }
    assert [call["api"] for call in contract["boundaries"]["backward_recompute"]["timed_calls"]] == [
        "nvte_rmsnorm_fwd",
        "nvte_rmsnorm_bwd",
    ]
    assert contract["boundaries"]["backward_recompute"]["timed_calls"][0]["bindings"][3] == "throwaway_z"
    assert contract["boundaries"]["composed"]["timed_calls"][0]["bindings"][3:5] == ["y", "rsigma"]
    assert contract["hardware_results"] == {
        "status": "blocked_not_executed",
        "required_hardware": ["h100", "gb200_or_b200"],
        "runs": [],
        "missing": [
            "hardware_execution",
            "resolved_binary_and_dependency_identity",
            "queried_workspace_records",
            "numerical_results",
            "latency_results",
        ],
    }
    assert contract["scorecard_effect"]["status_changed"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("dispatch", "key"), "workload_name"),
        (("dispatch", "workload_name_dispatch"), True),
        (("tensors", "shapes", "2048x4096", "x", "dtype"), "float32"),
        (("tensors", "shapes", "7x13", "x", "layout"), "column_major"),
        (("tensors", "shapes", "2048x4096", "x", "strides_elements"), [1, 2048]),
        (("tensors", "shapes", "2048x4096", "rsigma", "dtype"), "bfloat16"),
        (("tensors", "shapes", "7x13", "rsigma", "shape"), [7, 1]),
        (("tensors", "shapes", "7x13", "throwaway_z", "role"), "output"),
        (("workspace", "query_phase"), "inside_timing"),
        (("workspace", "queries", "forward", "bindings", 5), "forward_workspace"),
        (("workspace", "metadata_rule"), "reuse_largest_shape_metadata"),
        (("backend_controls", "forward", "api"), "nvte_enable_cudnn_norm_bwd"),
        (("backend_controls", "independence"), "one_shared_value"),
        (("boundaries", "backward_recompute", "timed_calls", 0, "bindings", 3), "y"),
        (("boundaries", "backward_recompute", "state_policy"), "saved_rsigma"),
        (("boundaries", "composed", "timed_calls", 1, "bindings", 2), "fresh_rsigma"),
        (("timing", "warmup_invocations"), 0),
        (("timing", "measured_invocations"), 49),
        (("timing", "measurement", 3), "cudaStreamSynchronize(stream)"),
        (("timing", "excluded", 2), "workspace_queries_inside_timing"),
        (("provider", "source", "commit"), "0" * 40),
        (("provider", "source_files", "transformer_engine/common/normalization/rmsnorm/rmsnorm_api.cpp"), "0" * 64),
        (("provider", "submodules", "3rdparty/cutlass"), "0" * 40),
        (("api", "signatures", "nvte_rmsnorm_fwd", "sha256"), "0" * 64),
        (("artifact_provenance", "device"), ["model"]),
        (("comparison", "outputs", "backward_recompute"), ["dgamma", "dx"]),
        (("comparison", "contract", "id"), "post_hoc_contract"),
        (("comparison", "threshold_status"), "accepted"),
        (("hardware_results", "runs"), [{"latency_ms": 0.1}]),
        (("hardware_results", "status"), "accepted"),
        (("scorecard_effect", "status_changed"), True),
    ],
    ids=[
        "workload-dispatch-key",
        "workload-dispatch-enabled",
        "public-dtype",
        "layout",
        "strides",
        "rsigma-dtype",
        "rsigma-shape",
        "throwaway-role",
        "workspace-query-timing",
        "workspace-empty-query",
        "workspace-rebinding",
        "forward-backend-api",
        "backend-independence",
        "throwaway-z-binding",
        "backward-state-policy",
        "composed-saved-rsigma",
        "warmup-count",
        "measurement-count",
        "synchronization",
        "timing-exclusion",
        "source-commit",
        "source-file-hash",
        "submodule-commit",
        "api-signature-hash",
        "artifact-provenance",
        "output-roles",
        "comparison-contract",
        "unresolved-thresholds",
        "fabricated-run",
        "hardware-status",
        "scorecard-promotion",
    ],
)
def test_contract_rejects_semantic_drift(path, value):
    with pytest.raises(ValueError, match="drifted"):
        validate_contract(_mutated(path, value))


def test_loader_rejects_duplicate_contract_keys(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}')

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_contract(duplicate)
