# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thoughtcrime tasks run through the canonical task loader and the rollout worker."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf
from rolloutengine.validation import validate_spec
from shellbox.backends.local.machine import check_local_environment
from shellbox.machine import Backend, ExecutionRequirements, Machine, MachineSpec
from skyrl_gym.rollout_specs import is_shell_harness
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import SourceTaskDataset
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.model_clients import REASONING_VIOLATION_TOKENS_KEY, REASONING_VIOLATIONS_KEY
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TokenProvenance, TrajectoryID
from taskcompendium.convert.pi_trajectory import prefix_row_task
from taskcompendium.grading_result import Outcome
from taskcompendium.models import Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast
from verifyit.reasoning_markup import ReasoningViolation

REPOSITORY = "open-athena/Marin-Thoughtcrime"
BASH = {
    "name": "bash",
    "description": "Execute a bash command.",
    "parameters": {"type": "object", "required": ["command"], "properties": {"command": {"type": "string"}}},
}
ROW = {
    "digest": "0" * 64,
    "messages": json.dumps(
        [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "List the workspace."},
            {
                "role": "assistant",
                "content": "<|start_think|>Look first.<|end_think|>",
                "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "a.txt\n"},
        ]
    ),
    "tools": json.dumps([BASH]),
    "recorded_violation": "stray_end",
    "prompt_tokens": 1300,
    "benchmark": "bixbench-cli-marin-corrections",
    "task": "bix-1-q1",
    "model": "Grug-67B-A2B-Datakit-SFT-262K-2026.09.21",
    "run": "20261017-022041-bixbench-pi",
    "trial": "bix-1-q1__abc",
    "position": 4,
    "reply": "{}",
}
WELL_FORMED = "<|start_think|>Read it.<|end_think|>"
STRAY_END = "<|start_think|>Read it.<|end_think|>Reading.<|end_think|>"


class LocalAdmissionFactory:
    backend = Backend.LOCAL

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        check_local_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        raise AssertionError(f"Admission must not acquire a machine: {spec}")


class WorkerTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt, **options):
        return [1, 2]


def prefix_task():
    source = Source(dataset=REPOSITORY, revision="f6ab8adc", row="rows.parquet:0", importer_revision="1")
    return prefix_row_task(RawRow("thoughtcrime-0", source, ROW), ConversionContext({}, None))


def lowered_prefix_task(tmp_path: Path):
    """Lower the task the way the trainer loads a selection's ``task_spec`` Parquet."""
    path = tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_spec": prefix_task().model_dump_json()}]), path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message.content }} {% endfor %}",
    )
    dataset = SourceTaskDataset(
        [str(path)],
        tokenizer,
        4096,
        num_workers=1,
        environment_configs=OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True),
    )
    (row,) = dataset.collate_fn(list(dataset))
    return row["input"]


def test_prefix_tasks_lower_to_a_machine_free_shell_harness_that_advertises_pi_tools(tmp_path: Path):
    task_in = lowered_prefix_task(tmp_path)
    lowered = task_in.spec
    assert is_shell_harness(lowered.harness)
    assert lowered.task_machine is None and lowered.limits.max_turns == 1
    assert task_in.metadata.data_source == REPOSITORY
    validate_spec(lowered, {"local": LocalAdmissionFactory()})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply,violation,status,reward",
    [
        (WELL_FORMED, None, Outcome.GRADED, 1.0),
        # A bare function call has no reasoning markup, so it is well formed.
        ("", None, Outcome.GRADED, 1.0),
        (STRAY_END, ReasoningViolation.STRAY_END, Outcome.SUBMISSION_FAILURE, 0.0),
    ],
)
async def test_worker_grades_prefix_replies_on_reasoning_markup(tmp_path: Path, reply, violation, status, reward):
    task_in = lowered_prefix_task(tmp_path)
    config = OmegaConf.create(
        {
            "backend": "vllm",
            "max_turns": 1,
            "max_input_length": 4096,
            "apply_overlong_filtering": False,
            "sampling_params": {
                "max_generate_length": 64,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0,
                "logprobs": True,
            },
        }
    )
    request = TaskRequestBatch(rows=(task_in,), trajectory_ids=(TrajectoryID(task_in.spec.task.id, 0),))
    observed = []

    class Client:
        async def generate(self, model_request):
            observed.append(model_request)
            return {
                "responses": [reply],
                "assistant_messages": [
                    {
                        "role": "assistant",
                        "content": reply,
                        "tool_calls": [
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {"name": "bash", "arguments": '{"command":"cat a.txt"}'},
                            }
                        ],
                    }
                ],
                "prompt_ids": [[1, 2]],
                "response_ids": [[11, 12, 13]],
                "response_logprobs": [[-0.1, -0.2, -0.3]],
                "stop_reasons": ["tool_calls"],
                "token_provenance": TokenProvenance.ENGINE,
                REASONING_VIOLATIONS_KEY: [violation],
                REASONING_VIOLATION_TOKENS_KEY: [None if violation is None else 2],
            }

    worker = TaskRolloutWorker(config, WholeTaskProjection(config, WorkerTokenizer()), Client(), {}, shutdown_timeout=30)
    try:
        (rollout,) = await worker.generate(request)
        batch = await worker.training_batch(request, [rollout])
    finally:
        await worker.shutdown()

    # One model turn: the call to a final tool ends the episode without a machine.
    assert len(observed) == 1
    (chat_options,) = observed[0]["chat_completion_params"]
    assert [tool["function"]["name"] for tool in chat_options["tools"]] == ["bash"]
    assert observed[0]["prompts"][0][-1] == {"role": "tool", "tool_call_id": "call-1", "content": "a.txt\n"}
    assert (rollout.grade.status, rollout.grade.reward) == (status, reward)
    assert batch["rewards"] == [pytest.approx(reward)]
    assert batch["data_sources"] == [REPOSITORY]
