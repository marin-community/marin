# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned SkyRL IFEval normalization and native scorer bridge."""

import hashlib
import json

import pytest
from taskcompendium.grader import grader_config
from taskcompendium.models import Source, TaskSpec
from taskcompendium.native_grader import NativeCommandSpec
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow
from taskcompendium.runtime.resources import resource_bytes
from verifyit.execution import source_callable

from experiments.post_training.task_curation.datasets.skyrl import ifeval_native_binding as skyrl_ifeval


@pytest.fixture
def source():
    return Source(dataset="ifeval-test", revision="pinned", row="1", importer_revision="1")


def test_rlvr_binds_native_pinned_scorer_without_rewriting_unknown_functions(source):
    constraints = {"func_name": "unknown_constraint"}
    task = skyrl_ifeval.normalize_rlvr(
        RawRow(
            "rlvr",
            source,
            {
                "messages": [{"role": "user", "content": "Follow the instruction."}],
                "ground_truth": json.dumps(constraints),
            },
        )
    )
    assert isinstance(task, TaskSpec)
    assert task.verifier.kind == "native_command"
    command = NativeCommandSpec.model_validate_json(task.verifier.parameters_json)
    assert command.result_format == "score_json"
    assert grader_config(task)["constraints"] == constraints
    invocation = next(resource for resource in task.resources.verifier if resource.path == "invocation.json")
    descriptor = json.loads(resource_bytes(invocation))
    assert descriptor["function"] == "pinned_skyrl_ifeval:compute_score"
    assert descriptor["reward_key"] == "score"


def test_unmappable_nemotron_constraint_is_unsupported(source):
    result = skyrl_ifeval.normalize_nemotron(
        RawRow(
            "unknown",
            source,
            {
                "input": [{"role": "user", "content": "Follow the instruction."}],
                "args": {"instruction_id_list": ["unknown"], "instruction_kwargs": [{}]},
            },
        )
    )
    assert isinstance(result, ImportRejection)
    assert result.kind == ImportFailureKind.UNSUPPORTED


def test_runner_hash_checks_the_exact_scorer_before_scoring(tmp_path):
    scorer = tmp_path / "utils.py"
    scorer.write_text(
        "def compute_score(answer, constraints, verifyit_enabled=False):\n"
        "    return {'score': 0.25 if answer == 'candidate' and constraints['func_name'] == "
        "'unknown_constraint' and not verifyit_enabled else 0.0}\n"
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"constraints": {"func_name": "unknown_constraint"}}))
    descriptor = tmp_path / "invocation.json"
    descriptor.write_text(
        json.dumps(
            {
                "function": "pinned_skyrl_ifeval:compute_score",
                "source_path": str(scorer),
                "source_sha256": hashlib.sha256(scorer.read_bytes()).hexdigest(),
                "args": ["answer", "contract.constraints"],
                "reward_key": "score",
            }
        )
    )
    answer = tmp_path / "answer.txt"
    answer.write_text("candidate")
    result = tmp_path / "reward.json"
    source_callable.main(descriptor, config, answer, result)
    assert json.loads(result.read_text()) == {"reward": 0.25, "detail": {}}
    scorer.write_text("raise RuntimeError('tampered source executed')\n")
    with pytest.raises(ValueError, match="scorer has changed"):
        source_callable.main(descriptor, config, answer, result)
