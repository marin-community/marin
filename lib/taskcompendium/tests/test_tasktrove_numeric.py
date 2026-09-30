# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic coverage for the unobserved numeric TaskTrove mode."""

import json

import pytest

from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.importers.tasktrove.numeric import import_task
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import AnswerType, VerifierKind
from taskcompendium.submission import PlainText

from .harbor_replay import run_replay_trial


def _synthetic_archive() -> TaskArchive:
    return TaskArchive(
        upstream_subset="synthetic-numeric",
        archive_path="numeric.tar.gz",
        release_uri="s3://example/tasktrove/clean/synthetic",
        release_revision="synthetic",
        archive_sha256="0" * 64,
        files={
            "task.toml": (
                b"""[metadata]
tasktrove_source = "synthetic-numeric"
tasktrove_path = "numeric.tar.gz"
family = "qa-short-answer"
converter = "synthetic_numeric"
template_id = "synthetic-numeric-v1"
mode = "numeric"
tags = ["synthetic", "numeric"]
"""
            ),
            "instruction.md": b"What is 5 / 2? Give one number.",
            "tests/verifier.toml": b'mode = "numeric"\nexpected = 2.5\ntolerance_abs = 0.01\ntolerance_rel = 0.02\n',
        },
    )


def test_synthetic_numeric_import_preserves_numeric_tolerances_and_tags():
    specification = import_task(_synthetic_archive())

    assert specification.answer_type is AnswerType.NUMBER
    assert specification.tags == ("synthetic", "numeric")
    assert specification.verifier.kind is VerifierKind.NUMERIC_ANSWER
    assert json.loads(specification.verifier.parameters_json) == {
        "expected": 2.5,
        "tolerance_abs": 0.01,
        "tolerance_rel": 0.02,
    }


@pytest.mark.asyncio
async def test_synthetic_numeric_direct_chat_grades_correct_incorrect_and_malformed(tmp_path):
    specification = import_task(_synthetic_archive())
    task = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tmp_path / "task",
    )
    trials = (
        ("correct", "2.52", 1.0),
        ("incorrect", "2.6", 0.0),
        ("malformed", "not-a-number", 0.0),
    )
    for name, answer, expected_reward in trials:
        result = await run_replay_trial(task, {"role": "assistant", "content": answer}, tmp_path / "trials", name)
        outcome = json.loads((tmp_path / f"trials/{name}/verifier/taskcompendium-result.json").read_text())
        assert result.exception_info is None, result.exception_info
        assert outcome == {"status": "graded", "reward": expected_reward, "error": None}


def test_synthetic_numeric_rejects_math_verifier_instead_of_casting_gold():
    archive = _synthetic_archive()
    archive.files["tests/verifier.toml"] = b'mode = "math"\nexpected = "5/2"\n'

    with pytest.raises(ValueError, match="numeric verifier"):
        import_task(archive)
