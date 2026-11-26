# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The OpenSWE conversion release: the fidelity gate's evidence binds the typed
exclusions, and the manifest refuses to lose rows."""

import json

import pytest
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.datasets.openswe import release


def _stage_gate(root, result: dict, rows: list[dict]):
    storage = StoragePath(str(root))
    (storage / release.GATE_RESULT_FILENAME).write_text(json.dumps(result) + "\n")
    (storage / release.GATE_ROWS_FILENAME).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def test_load_gate_evidence_parses_verdicts(tmp_path):
    rows = [
        {"instance_id": "a", "status": "qualified"},
        {"instance_id": "b", "status": "excluded", "exclusion_reason": "noop_mismatch", "exclusion_detail": "x"},
    ]
    _stage_gate(tmp_path, {"sampled": 2, "qualified": 1, "excluded": 1}, rows)
    evidence = release.load_gate_evidence(str(tmp_path))
    assert evidence.qualified == frozenset({"a"})
    assert set(evidence.excluded) == {"b"}
    assert evidence.sampled == 2


def test_load_gate_evidence_refuses_incomplete_reports(tmp_path):
    rows = [{"instance_id": "a", "status": "qualified"}]
    _stage_gate(tmp_path, {"sampled": 3, "qualified": 1, "excluded": 0}, rows)
    with pytest.raises(ValueError, match="sampled"):
        release.load_gate_evidence(str(tmp_path))

    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        release.load_gate_evidence(str(tmp_path / "empty"))


def test_write_conversion_artifacts_records_typed_exclusions(tmp_path):
    converted = [
        {"path": "a", "task_spec": json.dumps({"id": "a"}), "task_sha256": "0" * 64},
        {"path": "c", "task_spec": json.dumps({"id": "c"}), "task_sha256": "0" * 64},
    ]
    exclusions = [
        {"instance_id": "b", "reason": "dockerfile_rejected", "detail": "d", "evidence": {}},
        {"instance_id": "b", "reason": "dockerfile_rejected", "detail": "d", "evidence": {}},
    ]
    manifest = release.write_conversion_artifacts(
        converted,
        [],
        exclusions,
        {"input_rows": 4, "converted_rows": 2, "excluded_rows": 2, "gate_sampled": 1, "gate_qualified": 1},
        output_root=str(tmp_path),
    )
    assert manifest["rejections"] == {"dockerfile_rejected": 2}
    assert manifest["excluded_rows"] == 2
    assert (tmp_path / release.EXCLUSIONS_FILENAME).exists()
    assert (tmp_path / release.RECEIPTS_FILENAME).exists()
