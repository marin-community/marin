# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import sys

import pytest

from experiments.weight_merging import publish_checkpoint


def test_missing_responsible_use_stops_publication_before_checkpoint_read(tmp_path, monkeypatch):
    card = tmp_path / "README.md"
    card.write_text("---\nlicense: openmdw-1.1\n---\n\n" + publish_checkpoint.WARNING + "\n")
    license_file = tmp_path / "LICENSE"
    license_file.write_bytes(b"fixture license")
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_checkpoint",
            "--checkpoint",
            str(tmp_path / "unreadable-checkpoint"),
            "--repo-id",
            "fixture/unpublished",
            "--model-card",
            str(card),
            "--license",
            str(license_file),
            "--license-sha256",
            hashlib.sha256(license_file.read_bytes()).hexdigest(),
            "--report",
            str(report),
            "--workers",
            "1",
        ],
    )
    with pytest.raises(ValueError):
        publish_checkpoint.main()
    assert not report.exists()
