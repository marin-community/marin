# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

import train_semantic_projection_large as training_module  # noqa: E402
from glm_semantic_labels import SampleDocument, write_jsonl  # noqa: E402
from label_frozen_hierarchy_training import identity_digest  # noqa: E402
from rigging.filesystem import StoragePath  # noqa: E402


def documents() -> list[SampleDocument]:
    return [
        SampleDocument(
            sample_index=index,
            raw_sha256=f"{index:064x}",
            source=f"source-{index % 2}",
            source_category="standard",
            eval_rank=index,
            text=f"document {index}",
        )
        for index in range(3)
    ]


def write_training_root(root: StoragePath, rows: list[SampleDocument]) -> None:
    config = {
        "purpose": "semantic_projection_training",
        "excluded_evaluation_run_id": training_module.EXCLUDED_EVALUATION_RUN_ID,
        "training_identity_sha256": identity_digest(rows),
    }
    summary = {
        "documents": 3,
        "complete": True,
        "validation_repair_count": 0,
    }
    (root / "run-config.json").write_text(json.dumps(config))
    (root / "summary.json").write_text(json.dumps(summary))
    write_jsonl(root / "sample-private.jsonl.gz", (asdict(row) for row in rows))


def test_validated_training_documents_accepts_complete_pinned_sample(tmp_path: Path, monkeypatch) -> None:
    root = StoragePath(str(tmp_path))
    rows = documents()
    write_training_root(root, rows)
    monkeypatch.setattr(training_module, "EXPECTED_DOCUMENTS", 3)

    stored, config, metadata = training_module.validated_training_documents(root)

    assert stored == rows
    assert config["training_identity_sha256"] == identity_digest(rows)
    assert metadata["run_config_sha256"]
    assert metadata["summary_sha256"]


@pytest.mark.parametrize("fault", ["incomplete", "evaluation", "indices", "identity"])
def test_validated_training_documents_rejects_invalid_release_inputs(
    tmp_path: Path,
    monkeypatch,
    fault: str,
) -> None:
    root = StoragePath(str(tmp_path))
    rows = documents()
    if fault == "indices":
        rows[1] = replace(rows[1], sample_index=0)
    write_training_root(root, rows)
    if fault == "incomplete":
        summary = json.loads((root / "summary.json").read_text())
        summary["complete"] = False
        (root / "summary.json").write_text(json.dumps(summary))
    elif fault == "evaluation":
        config = json.loads((root / "run-config.json").read_text())
        config["excluded_evaluation_run_id"] = "wrong-evaluation"
        (root / "run-config.json").write_text(json.dumps(config))
    elif fault == "identity":
        config = json.loads((root / "run-config.json").read_text())
        config["training_identity_sha256"] = "0" * 64
        (root / "run-config.json").write_text(json.dumps(config))
    monkeypatch.setattr(training_module, "EXPECTED_DOCUMENTS", 3)

    with pytest.raises(ValueError):
        training_module.validated_training_documents(root)
