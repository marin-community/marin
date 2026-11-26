# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for per-task paired training requests."""

from dataclasses import replace

import pytest

from skyrlrecipe.declared import PromptAcquisition
from skyrl_train.config.driver_validation import validate_driver_config, validate_generation_driver_config
from skyrl_train.config.rollout_validation import EntrypointOperation
from skyrl_train.config.utils import get_default_config
from skyrl_train.contextual_sampling import PromptThompsonOrder, PublicPrompt
from skyrl_train.rollouts.context import RolloutRequestSpec
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request
from tests.cpu.task_specs import lowered_task, task_input, text_task


def paired_config():
    cfg = get_default_config()
    cfg.data.sampling.kind = PromptAcquisition.WITHIN_SOURCE_PASS_COST_AWARE.value
    cfg.data.sampling.paired_rollout_seed = 137
    cfg.generator.max_turns = 1
    cfg.generator.n_samples_per_prompt = 4
    cfg.generator.sampling_params.logprobs = None
    cfg.trainer.rollout_buffer.max_staleness_steps = 0
    return cfg


def prompt():
    task = task_input(lowered_task(text_task("stable-task-id"), max_turns=1))
    return {"uid": task.metadata.uid, "input": task}


def source_pass(row):
    return PromptThompsonOrder(
        [PublicPrompt(uid=row["uid"], source="string", text="Question")],
        reference={"string": 1.0},
        held_out_digests=frozenset(),
        acquisition=PromptAcquisition.WITHIN_SOURCE_PASS_COST_AWARE,
        seed=13,
        window_size=1,
        decay=0.95,
        epsilon=1.0,
        reference_tokens=128,
    )


def test_pair_seeds_match_stable_task_ids_and_separate_resumed_repetitions():
    a = prompt()
    b = {
        "uid": "other-filtered-row",
        "input": replace(a["input"], metadata=replace(a["input"].metadata, uid="other-filtered-row")),
    }
    requests = RolloutRequestSpec.from_config(paired_config())
    order_a, order_b = source_pass(a), source_pass(b)
    order_a.next_index()
    order_b.next_index()
    first = requests.request(a, 1, source_visit=order_a.source_pass_visit(a["uid"]))
    matched = requests.request(b, 99, source_visit=order_b.source_pass_visit(b["uid"]))
    assert first.request_seeds == matched.request_seeds
    assert len(set(first.request_seeds)) == 4

    restored = source_pass(a)
    restored.load_state_dict(order_a.state_dict())
    restored.next_index()
    order_a.next_index()
    continued = requests.request(a, 2, source_visit=order_a.source_pass_visit(a["uid"]))
    resumed = requests.request(a, 2, source_visit=restored.source_pass_visit(a["uid"]))
    assert continued.request_seeds == resumed.request_seeds
    assert len(set(first.request_seeds + resumed.request_seeds)) == 8
    assert all(0 <= seed < 1 << 63 for seed in resumed.request_seeds)


class ChatTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, skip_special_tokens):
        return "answer"

    def encode(self, text, add_special_tokens):
        return [12]


class CapturedChat:
    model_name = "policy"
    tokenizer = ChatTokenizer()

    def __init__(self):
        self.requests = {}

    async def tokenize(self, request):
        return {"tokens": [1, 2]}

    async def chat_completion(self, request):
        body = request["json"]
        self.requests[body["session_id"]] = body
        return {"choices": [{"message": {"role": "assistant", "content": "answer"},
                             "token_ids": [12], "finish_reason": "stop"}]}


@pytest.mark.asyncio
async def test_native_task_chat_receives_each_request_seed_and_preserves_evaluation():
    cfg = paired_config()
    request_spec = RolloutRequestSpec.from_config(cfg)
    row = prompt()
    order = source_pass(row)
    order.next_index()
    paired = request_spec.request(row, 1, source_visit=order.source_pass_visit(row["uid"]))
    evaluation, _ = prepare_trajectory_request([row], 1, {"seed": 19}, "eval", 0)
    cfg.data.sampling.paired_rollout_seed = None
    default = RolloutRequestSpec.from_config(cfg).request(row, 1)
    chat = CapturedChat()
    worker = TaskRolloutWorker(
        cfg.generator,
        WholeTaskProjection(cfg.generator, chat.tokenizer),
        DirectModelClient(chat),
        {},
        shutdown_timeout=30,
    )
    try:
        result = await worker.generate(paired)
        assert len(result) == 4 and all(rollout.response_token_ids == (12,) for rollout in result)
        assert {
            session: body["seed"] for session, body in chat.requests.items()
        } == {
            identity.to_string(): seed for identity, seed in zip(
                paired.trajectory_ids, paired.request_seeds, strict=True
            )
        }
        await worker.generate(evaluation)
        assert chat.requests[evaluation.trajectory_ids[0].to_string()]["seed"] == 19
        await worker.generate(default)
        assert all("seed" not in body for body in chat.requests.values())
    finally:
        await worker.shutdown()


def test_paired_rollout_seed_is_rejected_outside_its_native_training_mode():
    cfg = paired_config()
    with pytest.raises(ValueError, match="synchronous one-turn source passes"):
        validate_generation_driver_config(cfg, check_runtime_resources=False)
    cfg.trainer.rollout_buffer.max_staleness_steps = 2
    with pytest.raises(ValueError, match="synchronous one-turn source passes"):
        validate_driver_config(cfg, EntrypointOperation.TRAIN, check_runtime_resources=False)
