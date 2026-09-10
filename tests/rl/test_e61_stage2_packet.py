# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input provenance must fail before a prospective Snowball treatment is composed."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.post_training import e61_stage2
from experiments.post_training.e61_stage2 import stage2_packet
from experiments.post_training.e61_stage2_preparation import check_process


def test_wrong_input_receipt_rejected():
    with pytest.raises(ValueError, match="Original stage-2 input receipt differs"):
        stage2_packet({}, b"{}", "parser_only")


def test_receipt_objects_join_before_model_selection():
    skeleton = json.loads(e61_stage2.SKELETON.read_bytes())
    receipt = {"reads": copy.deepcopy(skeleton["input_objects"])}
    receipt["reads"][next(iter(receipt["reads"]))]["bytes"] += 1
    raw = json.dumps(receipt).encode()
    skeleton["input_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    with patch.object(Path, "read_bytes", return_value=json.dumps(skeleton).encode()):
        with pytest.raises(ValueError, match="Stage-2 input object differs"):
            stage2_packet({}, raw, "parser_only")


def test_failed_native_preparation_retains_bounded_initiating_error(monkeypatch):
    captures = []
    monkeypatch.setenv("E61_TEST_API_KEY", "fixture-secret-value")
    monkeypatch.setattr(
        "experiments.post_training.e61_stage2_preparation.persist_new", lambda uri, data: captures.append((uri, data))
    )
    failure = subprocess.CompletedProcess(
        [], 1, stdout="not retained", stderr="initiating fixture-secret-value\n" + "x" * 50000
    )
    with pytest.raises(RuntimeError, match="bounded initiating error retained"):
        check_process(failure, "memory://failure.json", "parser_only", "compose")
    assert len(captures) == 1
    retained = json.loads(captures[0][1])
    assert retained["stderr_first_16k"].startswith("initiating <redacted>")
    assert "fixture-secret-value" not in captures[0][1].decode()
    assert len(retained["stderr_first_16k"]) == 16384 and len(retained["stderr_last_4k"]) == 4096
