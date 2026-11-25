# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A policy file reads into the schema, with every name resolved and unknown names rejected."""

import pytest
from evalrecords.records import PromptFormat, TaskImageBuild, TaskMachineSettings

from experiments.evaluation.models import models
from experiments.evaluation.policy.load import load_policy
from experiments.evaluation.policy.schema import EvalPolicy, PolicyCandidate, PolicyEvaluation, PolicyOverride

POLICY_FILE = """
version: test
default_context: 73728
candidates:
  qwen3.6-35b-a3b: {max_context: 262144}
evaluations:
  - boolq
  - {benchmark: gsm8k, label: gsm8k-0shot, num_fewshot: 0, prompt_format: chat, judge: together}
  - {benchmark: swebench-verified, protocol: mini-swe-agent, concurrency: 16, task_machine: {image_build: declarative}}
overrides:
  - {candidate: qwen3.6-35b-a3b, evaluation: boolq, context: 32768, reason: test}
"""


def test_a_policy_file_reads_into_the_schema(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(POLICY_FILE)
    qwen = PolicyCandidate(model=models()["qwen3.6-35b-a3b"], max_context=262_144)
    boolq = PolicyEvaluation("boolq")
    assert load_policy(path) == EvalPolicy(
        version="test",
        default_context=73_728,
        candidates=(qwen,),
        in_distribution=(
            boolq,
            PolicyEvaluation(
                "gsm8k", label="gsm8k-0shot", num_fewshot=0, prompt_format=PromptFormat.CHAT, judge="together"
            ),
            PolicyEvaluation(
                "swebench-verified",
                protocol="mini-swe-agent",
                concurrency=16,
                task_machine=TaskMachineSettings(image_build=TaskImageBuild.DECLARATIVE),
            ),
        ),
        overrides=(PolicyOverride(candidate=qwen, evaluation=boolq, context=32_768, reason="test"),),
    )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param("version: test", "version: test\ncontext: 1", id="unknown-key"),
        pytest.param("qwen3.6-35b-a3b: {max", "no-such-model: {max", id="unknown-model"),
        pytest.param("  - boolq", "  - no-such-benchmark", id="unknown-benchmark"),
        pytest.param("judge: together", "judge: nobody", id="unknown-judge"),
        pytest.param("evaluation: boolq", "evaluation: piqa", id="override-names-no-evaluation"),
    ],
)
def test_a_policy_file_with_an_unknown_name_is_rejected(tmp_path, old, new):
    path = tmp_path / "policy.yaml"
    path.write_text(POLICY_FILE.replace(old, new, 1))
    with pytest.raises(ValueError):
        load_policy(path)
