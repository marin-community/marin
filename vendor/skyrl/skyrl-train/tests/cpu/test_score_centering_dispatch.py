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
from tests.cpu.task_specs import lowered_task, task_input, text_task


@pytest.mark.asyncio
@pytest.mark.parametrize("topk", [0, 128])
async def test_training_task_retains_heads_and_capture_costs(topk):
    ids = np.tile(np.arange(128, dtype=np.int32), (4, 1))
    scores = np.full((4, 128), -np.log(128), dtype=np.float32)

    class Tokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, **kwargs):
            return [1]

    class Engine:
        tokenizer = Tokenizer()

        async def generate(self, request):
            return {
                "prompt_ids": request["prompt_token_ids"],
                "responses": ["answer"],
                "response_ids": [[11, 12, 13, 14]],
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
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(task, "assistant_prefix", max_turns=1)),),
        sampling_params={"logprobs": topk},
    )
    worker = TaskRolloutWorker(
        config, WholeTaskProjection(config, Tokenizer()), DirectModelClient(Engine()), {}, shutdown_timeout=5
    )
    writer = Writer()
    try:
        assert await worker.run_task(RolloutTask(RolloutLease("lease", 7, 8), {"uid": "prompt"}, request), writer) == 4
    finally:
        await worker.shutdown()
    batch = writer.group.trajectory_batch
    assert writer.group.policy_step == 7
    assert batch["response_ids"] == [[11, 12, 13, 14]]
    metrics = batch.get("rollout_metrics", {})
    if topk:
        np.testing.assert_array_equal(batch["student_topk_indices"][0], ids)
        np.testing.assert_array_equal(batch["behavior_topk_logprobs"][0], scores)
        assert metrics["score_centering_capture/requests"] == 1
        assert metrics["score_centering_capture/captured_tokens"] == 4
        assert metrics["score_centering_capture/captured_head_bytes"] == 4096
        assert metrics["score_centering_capture/exported_head_bytes"] == 4096
        assert metrics["score_centering_capture/collection_seconds_sum"] >= 0
        assert metrics["score_centering_capture/assembly_seconds_sum"] >= 0
    else:
        assert batch.get("student_topk_indices") is None
        assert not any(name.startswith("score_centering_capture/") for name in metrics)
