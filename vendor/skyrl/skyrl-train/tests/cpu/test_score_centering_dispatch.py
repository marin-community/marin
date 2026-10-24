"""Capture costs travel with real task execution and projected training data."""

import numpy as np
import pytest
from skyrl_train.config.utils import get_default_config
from skyrl_train.rollouts.buffer import RolloutLease, RolloutTask
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.types import TaskRequestBatch
from taskcompendium.models import ConversationInput, TextMessage
from taskcompendium.grader import verifyit_package
from verifyit.spec import NumericSpec
from tests.cpu.task_specs import lowered_task, task_input, text_task


@pytest.mark.asyncio
@pytest.mark.parametrize("topk,probe_tokens", [(0, 0), (128, 0), (128, 32)])
async def test_training_task_retains_heads_and_capture_costs(topk, probe_tokens):
    length = 40 if probe_tokens else 4
    response = list(range(11, 11 + length))
    ids = np.tile(np.arange(128, dtype=np.int32), (length, 1))
    scores = np.full((length, 128), -np.log(128), dtype=np.float32)

    class Tokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, **kwargs):
            return [1]

    class Engine:
        tokenizer = Tokenizer()

        async def generate(self, request):
            return {
                "prompt_ids": request["prompt_token_ids"],
                "responses": ["12"],
                "response_ids": [response],
                "response_logprobs": [scores[:, 0].tolist()],
                "stop_reasons": ["stop"],
                **({"student_topk_indices": [ids], "behavior_topk_logprobs": [scores]} if topk else {}),
            }

    class Writer:
        async def write_rollout(self, lease, group):
            self.group = group

    config = get_default_config().generator
    config.sampling_params.logprobs = topk
    task = text_task("prompt").model_copy(
        update={"context": ConversationInput(events=(TextMessage(role="assistant", content="prefix"),))}
    )
    if probe_tokens:
        task = task.model_copy(
            update={"grader": verifyit_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0)).grader}
        )
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(task, "assistant_prefix", max_turns=1)),),
        sampling_params={"logprobs": topk},
        score_centering_probe_tokens=probe_tokens,
    )
    worker = TaskRolloutWorker(
        config, WholeTaskProjection(config, Tokenizer()), DirectModelClient(Engine()), {}, shutdown_timeout=5
    )
    writer = Writer()
    try:
        assert (
            await worker.run_task(RolloutTask(RolloutLease("lease", 7, 8), {"uid": "prompt"}, request), writer)
            == length
        )
    finally:
        await worker.shutdown()
    batch = writer.group.trajectory_batch
    assert writer.group.policy_step == 7
    assert batch["response_ids"] == [response]
    metrics = batch.get("rollout_metrics", {})
    if topk:
        if probe_tokens:
            positions = batch["score_probe_positions"][0]
            assert len(positions) == len(np.unique(positions)) == 32
            assert positions[0] == 0 and positions[-1] == length - 1
            np.testing.assert_array_equal(batch["score_probe_indices"][0], ids[positions])
            np.testing.assert_array_equal(batch["score_probe_logprobs"][0], scores[positions])
            assert batch.get("student_topk_indices") is None
        else:
            np.testing.assert_array_equal(batch["student_topk_indices"][0], ids)
            np.testing.assert_array_equal(batch["behavior_topk_logprobs"][0], scores)
        assert metrics["score_centering_capture/requests"] == 1
        assert metrics["score_centering_capture/captured_tokens"] == length
        assert metrics["score_centering_capture/captured_head_bytes"] == length * 1024
        assert metrics["score_centering_capture/exported_head_bytes"] == (32 * 1028 if probe_tokens else length * 1024)
        assert metrics["score_centering_capture/collection_seconds_sum"] >= 0
        assert metrics["score_centering_capture/assembly_seconds_sum"] >= 0
    else:
        assert batch.get("student_topk_indices") is None
        assert not any(name.startswith("score_centering_capture/") for name in metrics)
