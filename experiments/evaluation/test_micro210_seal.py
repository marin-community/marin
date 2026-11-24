# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The seal applies the directive exactly: drop two slots, one image-free swap."""

import base64
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.evaluation.micro210_replacements import humanevalplus_candidate
from scripts.ops.seal_micro210_policy import (
    DROPPED_TASKS,
    SWAP_SOURCE,
    SWAPPED_TASK,
    apply_directive,
    binding_report,
    merge_receipts,
)

SEALED_TASK = f"humanevalplus/{SWAP_SOURCE}"


def _inline_file(path: str, payload: bytes) -> dict:
    return {
        "path": path,
        "source": {"kind": "inline_file", "content_base64": base64.b64encode(payload).decode()},
    }


def _patch_evalplus(monkeypatch, tmp_path) -> None:
    """Serve a tiny local evalplus parquet instead of the networked download."""
    destination = tmp_path / "evalplus.parquet"
    pq.write_table(
        pa.table(
            {
                "task_id": [SWAP_SOURCE],
                "prompt": ["def add(a, b):"],
                "canonical_solution": ["    return a + b"],
                "entry_point": ["add"],
                "test": ["assert add(1, 2) == 3"],
            }
        ),
        destination,
    )
    monkeypatch.setattr(
        "experiments.evaluation.micro210_replacements.hf_hub_download",
        lambda dataset, repo_type, revision, filename: str(destination),
    )


@pytest.fixture
def table() -> pa.Table:
    """Four tasks x three reps shaped like the prepared table: two drop targets, the
    broken HumanEval task, and one unaffected task."""
    config = {"spec": {"doc": {"prompt": "original"}, "grader": {"reference": {"problem": {"prompt": "original"}}}}}
    rows = []
    for task_id, benchmark, source in (
        ("tb2-recovery/terminal-bench/mcmc-sampling-stan", "tb2-recovery", "mcmc-sampling-stan"),
        ("ot-tblite-recovery/pdf-table-parsing", "ot-tblite-recovery", "pdf-table-parsing"),
        (SWAPPED_TASK, "humanevalplus", "HumanEval/130"),
        ("aime24/example-0", "aime24", "example-0"),
    ):
        for repetition_index in range(3):
            rows.append(
                {
                    "task_id": task_id,
                    "source_task_id": source,
                    "benchmark": benchmark,
                    "selection_order": len(rows),
                    "is_micro": True,
                    "repetitions": 1 if benchmark == "humanevalplus" else 3,
                    "task_spec": json.dumps(
                        {
                            "id": task_id,
                            "labels": {},
                            "grader": {"kind": "none", "reason": "Harbor grades the completed trial", "contract": {}},
                            "resources": {
                                "all": [],
                                "worker": [],
                                "oracle": [],
                                "verifier": [_inline_file("config.json", json.dumps(config).encode())],
                            },
                        }
                    ),
                    "task_rollout_spec": json.dumps(
                        {
                            "task": json.loads("{}"),
                            "harness": "harbor_trial" if benchmark != "humanevalplus" else "wheels",
                            "task_machine": None,
                            "verifier_machine": None,
                            "limits": {},
                            "trace": "exact_tokens",
                        }
                    ),
                    "agent": "pi",
                    "validation_profile": "micro210",
                    "repetition_index": repetition_index,
                }
            )
    return pa.Table.from_pylist(rows)


def test_apply_directive_drops_and_swaps(table, monkeypatch, tmp_path):
    _patch_evalplus(monkeypatch, tmp_path)
    _, swaps, sealed_rows = apply_directive(table)
    ids = {row["task_id"] for row in sealed_rows}
    assert all(task_id not in ids for task_id in DROPPED_TASKS)
    assert SWAPPED_TASK not in ids
    assert SEALED_TASK in ids
    assert len(sealed_rows) == len(table) - 6
    spec = json.loads(next(row for row in sealed_rows if row["task_id"] == SEALED_TASK)["task_spec"])
    assert spec["labels"]["micro210_replacement_source"]
    assert swaps["swapped"][0]["replacement"]["task_id"] == SEALED_TASK
    # The unaffected task keeps all three repetitions verbatim.
    assert sum(1 for row in sealed_rows if row["task_id"] == "aime24/example-0") == 3


def test_binding_report_detects_drift(table, monkeypatch, tmp_path):
    _patch_evalplus(monkeypatch, tmp_path)
    _, _, sealed_rows = apply_directive(table)
    prepared_rows = table.to_pylist()
    report = binding_report(prepared_rows, sealed_rows)
    assert report["kept_rows_with_unchanged_spec"] == 3
    drifted = [
        dict(row, task_spec=row["task_spec"] + " ") for row in sealed_rows if row["task_id"] == "aime24/example-0"
    ]
    with pytest.raises(ValueError, match="drifted"):
        binding_report(prepared_rows, drifted)


def _receipt(rows: list[dict], failures: list[dict], digest: str = "digest") -> dict:
    return {
        "prepared_rollouts_sha256": digest,
        "grader_library_manifest_sha256": "manifest",
        "dependencies": [],
        "rows": rows,
        "failures": failures,
        "deferred_harbor_rows": 0,
    }


def _result(task_id: str, repetition_index: int, oracle: float = 1.0, nop: float = 0.0) -> dict:
    return {
        "task_id": task_id,
        "repetition_index": repetition_index,
        "oracle_reward": oracle,
        "nop_reward": nop,
        "judge_required": False,
    }


def test_merge_receipts_binds_a8_and_replacement(table, monkeypatch, tmp_path):
    _patch_evalplus(monkeypatch, tmp_path)
    _, _, sealed_rows = apply_directive(table)
    prepared_rows = table.to_pylist()
    a8_rows = [
        _result(row["task_id"], row["repetition_index"]) for row in prepared_rows if row["task_id"] != SWAPPED_TASK
    ]
    a8_failures = [
        {**_result(SWAPPED_TASK, 0, oracle=0.0), "check": "oracle", "required": 1.0},
        {
            **_result("tb2-recovery/terminal-bench/mcmc-sampling-stan", 0),
            "failed": True,
            "error": "rstan",
            "check": "row",
        },
        {**_result("ot-tblite-recovery/pdf-table-parsing", 1), "failed": True, "error": "camelot", "check": "row"},
    ]
    a8 = _receipt(a8_rows, a8_failures)
    replacement = _receipt([_result(SEALED_TASK, repetition) for repetition in range(3)], [])
    merged = merge_receipts(a8, replacement, sealed_rows, "sealed-digest")
    assert merged["pass"] is True
    assert merged["failures"] == []
    assert len(merged["rows"]) == len(sealed_rows)
    assert merged["prepared_rollouts_sha256"] == "sealed-digest"
    assert all(item["task_id"] != SWAPPED_TASK and item["task_id"] not in DROPPED_TASKS for item in merged["rows"])
    assert sum(1 for item in merged["rows"] if item["task_id"] == SEALED_TASK) == 3


def test_merge_receipts_rejects_stray_a8_failures(table, monkeypatch, tmp_path):
    _patch_evalplus(monkeypatch, tmp_path)
    _, _, sealed_rows = apply_directive(table)
    prepared_rows = table.to_pylist()
    a8_rows = [
        _result(row["task_id"], row["repetition_index"]) for row in prepared_rows if row["task_id"] != SWAPPED_TASK
    ]
    a8 = _receipt(a8_rows, [{**_result("aime24/example-0", 0, oracle=0.0), "check": "oracle", "required": 1.0}])
    replacement = _receipt([_result(SEALED_TASK, repetition) for repetition in range(3)], [])
    with pytest.raises(ValueError, match="outside the directive's scope"):
        merge_receipts(a8, replacement, sealed_rows, "sealed-digest")


def test_merge_receipts_rejects_failed_replacement(table, monkeypatch, tmp_path):
    _patch_evalplus(monkeypatch, tmp_path)
    _, _, sealed_rows = apply_directive(table)
    prepared_rows = table.to_pylist()
    a8_rows = [
        _result(row["task_id"], row["repetition_index"]) for row in prepared_rows if row["task_id"] != SWAPPED_TASK
    ]
    a8 = _receipt(a8_rows, [])
    replacement = _receipt(
        [{**_result(SEALED_TASK, 0, oracle=0.0), "failed": True, "error": "boom"}]
        + [_result(SEALED_TASK, repetition) for repetition in (1, 2)],
        [],
    )
    with pytest.raises(ValueError, match="failed its gate"):
        merge_receipts(a8, replacement, sealed_rows, "sealed-digest")


def test_candidate_swap_uses_the_pool_record(table, monkeypatch, tmp_path):
    _patch_evalplus(monkeypatch, tmp_path)
    template = next(row for row in table.to_pylist() if row["task_id"] == SWAPPED_TASK)
    candidate = humanevalplus_candidate(SWAP_SOURCE, template)
    assert candidate.task_id == SEALED_TASK
    assert candidate.source_task_id == SWAP_SOURCE
