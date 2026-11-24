# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replacement rows construct cleanly and their Dockerfiles carry no placeholder."""

import base64
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.evaluation.micro210_replacements import (
    _verifier_files,
    humanevalplus_candidate,
    ot_tblite_candidate,
    tb2_candidate,
)

PLACEHOLDER = "__BUILD_FROM_UPSTREAM__"


def _inline_file(path: str, payload: bytes) -> dict:
    return {
        "path": path,
        "source": {"kind": "inline_file", "content_base64": base64.b64encode(payload).decode()},
    }


@pytest.fixture
def template_row() -> dict:
    """A frozen row's shape: verifier resources carry the files candidates rewrite."""
    config = {"spec": {"doc": {"prompt": "original"}, "grader": {"reference": {"problem": {"prompt": "original"}}}}}
    return {
        "task_id": "template/row",
        "benchmark": "template",
        "source_task_id": "template-source",
        "task_spec": json.dumps(
            {
                "id": "template/row",
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
                "harness": "harbor_trial",
                "task_machine": None,
                "verifier_machine": None,
                "limits": {},
                "trace": "exact_tokens",
            }
        ),
        "repetitions": 3,
    }


def _tb2_package(dockerfile: str) -> dict[str, bytes]:
    """Minimal tb2-shaped package: the candidate builder reads these keys only."""
    return {
        "Dockerfile": dockerfile.encode(),
        "task.toml": b"[environment]\n",
        "instruction.md": b"do the task",
        "solution/solve.sh": b"#!/bin/bash\n",
        "tests/test.sh": b"#!/bin/bash\n",
    }


def _tblite_package(dockerfile: str) -> dict[str, bytes]:
    """Minimal tblite-shaped package; its Dockerfile lives under environment/."""
    return {
        "environment/Dockerfile": dockerfile.encode(),
        "task.toml": b"[environment]\n",
        "instruction.md": b"do the task",
        "solution/solve.sh": b"#!/bin/bash\n",
        "tests/test.sh": b"#!/bin/bash\n",
    }


def test_generated_dockerfiles_carry_no_placeholder(template_row, monkeypatch):
    """The build-from-upstream path — where the placeholder used to appear — emits a
    real Dockerfile: Pi stage first, the task's own base carried verbatim."""
    # The registry-image probe is a network call; force the build-from-upstream path.
    monkeypatch.setattr(
        "experiments.evaluation.micro210_replacements._tb2_registry_image_exists",
        lambda source_task_id: False,
    )

    tb2 = tb2_candidate(
        "adaptive-rejection-sampler",
        template_row,
        _tb2_package("FROM ghcr.io/laude-institute/t-bench/python-3-13:20250620\nWORKDIR /app\n"),
    )
    tblite = ot_tblite_candidate(
        "acl-permissions-inheritance",
        template_row,
        _tblite_package("FROM python:3.11-slim\nWORKDIR /app\n"),
    )

    for candidate in (tb2, tblite):
        files = _verifier_files(json.loads(candidate.task_spec))
        dockerfile = files["environment/Dockerfile"].decode()
        assert PLACEHOLDER not in dockerfile
        assert dockerfile.startswith("FROM node@"), "Pi runtime stage must come first"
        assert "long_loop_pi_runtime" in dockerfile
        assert "FROM python" in dockerfile or "FROM ghcr.io" in dockerfile, "task base must be carried"


def test_humanevalplus_candidate_rewrites_the_row(template_row, monkeypatch, tmp_path):
    """The humanevalplus family swaps in the pool record's problem, offline."""
    parquet = tmp_path / "evalplus.parquet"
    pq.write_table(
        pa.table(
            {
                "task_id": ["HumanEval/0"],
                "prompt": ["def add(a, b):"],
                "canonical_solution": ["    return a + b"],
                "entry_point": ["add"],
                "test": ["assert add(1, 2) == 3"],
            }
        ),
        parquet,
    )
    monkeypatch.setattr(
        "experiments.evaluation.micro210_replacements.hf_hub_download",
        lambda dataset, repo_type, revision, filename: str(parquet),
    )

    candidate = humanevalplus_candidate("HumanEval/0", template_row)

    spec = json.loads(candidate.task_spec)
    assert spec["id"] == "humanevalplus/HumanEval/0"
    config = json.loads(
        base64.b64decode(
            next(r for r in spec["resources"]["verifier"] if r["path"] == "config.json")["source"]["content_base64"]
        )
    )
    assert config["spec"]["doc"]["task_id"] == "HumanEval/0"
    assert config["spec"]["doc"]["entry_point"] == "add"
    assert config["spec"]["grader"]["reference"]["problem"]["prompt"] == "def add(a, b):"
