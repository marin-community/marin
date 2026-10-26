from types import SimpleNamespace

import pytest
import ray
import torch

from skyrl_train.config.numerics import CheckFailure
from skyrl_train.config.utils import get_default_config
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.exactness_check import ExactnessCheck
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.inference_engines.base import UNKNOWN_ENGINE_DP_RANK
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingOutputBatch
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.trajectory_runners.types import ROLLOUT_ENGINE_DP_RANKS_KEY
from skyrl_train.weight_sync.base import WeightSyncReason
from tests.cpu.util import example_dummy_config

# The two live engines' responses to the check prompt and their log-probabilities, by data-parallel rank.
_CHECK_RESPONSES = {1: ([7, 8, 9], [-0.5, -1.25, -2.0]), 0: ([5, 6], [-0.75, -3.5])}
_CHECK_LOGPROBS = {
    token: value for ids, values in _CHECK_RESPONSES.values() for token, value in zip(ids, values, strict=True)
}


class _TwoEngineClient:
    """Two live engines, data-parallel ranks 1 and 0, that answer the check prompt with ``_CHECK_RESPONSES``; an engine
    dies during the check requests numbered in ``failing`` (from 1)."""

    def __init__(self, failing):
        self.requests = 0
        self.failing = failing

    async def generate_on_live_engines(self, prompt_token_ids, sampling_params):
        self.requests += 1
        if self.requests in self.failing:
            raise ray.exceptions.ActorDiedError()
        return [
            (rank, {"response_ids": [ids], "response_logprobs": [values]})
            for rank, (ids, values) in _CHECK_RESPONSES.items()
        ]


def _policy_logprobs(batch):
    """Each response token's engine log-probability (0 for a token no engine sampled), except the second token of the
    fourth row, one float32 ulp toward zero."""
    responses = batch["sequences"][:, -batch.metadata["response_length"] :].tolist()
    output = torch.tensor([[_CHECK_LOGPROBS.get(token, 0.0) for token in row] for row in responses])
    if len(output) == 4:
        output[3, 1] = torch.nextafter(output[3, 1], torch.tensor(0.0))
    return output


@pytest.mark.asyncio
async def test_exactness_check_counts_differing_bits_of_the_responses_taken_after_qualifying_syncs():
    tokenizer = SimpleNamespace(pad_token_id=0, encode=lambda text, add_special_tokens: [1, 2, 3])
    sampling_params = get_default_config().generator.sampling_params
    check = ExactnessCheck(2, CheckFailure.LOG, sampling_params, tokenizer)
    client = _TwoEngineClient(failing={2})

    reported = []
    for reason in (
        WeightSyncReason.INITIAL,
        WeightSyncReason.TRAINING_STEP,
        WeightSyncReason.EVALUATION,
        WeightSyncReason.TRAINING_STEP,
        WeightSyncReason.AFTER_CHECKPOINT_SAVE,
    ):
        metrics = await check.after_weight_sync(reason, client)
        metrics |= check.score(_policy_logprobs, dp_size=4, num_experts=None)
        reported.append(
            {
                name.removeprefix("exactness_check/"): value
                for name, value in metrics.items()
                if not name.endswith("_seconds")
            }
        )

    # The check follows the first sync, every second training sync and the re-sync after a checkpoint save; the
    # generation an engine died during is counted and scores nothing. Its rows repeat to the policy's four
    # data-parallel ranks, each counted at the rank of the engine that served it; the one value a float32 ulp off is
    # one mismatch, at rank 0.
    scored = {
        "engines": 2.0,
        "tokens": 10.0,
        "mismatched_tokens": 1.0,
        "log_ratio_abs_max": 2.0**-22,
        "dp_rank_1/tokens": 6.0,
        "dp_rank_1/mismatched_tokens": 0.0,
        "dp_rank_0/tokens": 4.0,
        "dp_rank_0/mismatched_tokens": 1.0,
    }
    assert reported == [scored, {}, {}, {"generate_failures": 1.0}, scored]

    # A mode that promises bit-identical log-probabilities stops the run on that one bit under on_failure=stop.
    stopping = ExactnessCheck(2, CheckFailure.STOP, sampling_params, tokenizer)
    await stopping.after_weight_sync(WeightSyncReason.INITIAL, client)
    with pytest.raises(RuntimeError):
        stopping.score(_policy_logprobs, dp_size=4, num_experts=None)


class _RecordingPolicyGroup:
    """A one-rank policy that records each forward batch and scores every response token 0."""

    actor_infos = [SimpleNamespace(rank=MeshRank(dp=0, sp=0, tp=0, pp=0, world_size=1, dp_size=1, pp_size=1))]

    def __init__(self):
        self.forward_batches = []

    def async_run_ray_method(self, dispatch_type, method_name, **kwargs):
        if method_name == "empty_cache":
            return []
        assert method_name == "forward", method_name
        data = kwargs["data"]
        self.forward_batches.append(data)
        return [ray.put(TrainingOutputBatch({"output": torch.zeros(len(data), data.metadata["response_length"])}))]


def _trajectory_rows(prompts, responses, **channels):
    return {
        "prompt_token_ids": prompts,
        "response_ids": responses,
        "rewards": [[0.0] * len(response) for response in responses],
        "loss_masks": [[1] * len(response) for response in responses],
        "rollout_logprobs": None,
        **channels,
    }


def test_the_pre_update_policy_forward_receives_each_rows_serving_engine_rank_and_loss_mask(ray_init):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = example_dummy_config()
    trainer.group_advantage_invariant = GroupAdvantageInvariant.no_group_advantage(physical_group_size=1)
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.pad_batch = lambda batch: batch
    trainer.policy_model = _RecordingPolicyGroup()
    trainer.ref_model = None
    trainer.critic_model = None
    trainer.colocate_all = False
    trainer.global_step = 1
    # A batch whose model turns report no serving rank fills its rows with UNKNOWN_ENGINE_DP_RANK.
    trajectory_batch = concatenate_trajectory_batches(
        [
            _trajectory_rows([[1, 2], [3]], [[4, 5, 6], [7]], **{ROLLOUT_ENGINE_DP_RANKS_KEY: [3, 0]}),
            _trajectory_rows([[8]], [[9, 10]]),
        ],
        tis_lcs_alert_threshold=0.0,
    )

    training_input = trainer.convert_to_training_input(trajectory_batch, ["a", "b", "c"])
    trainer.fwd_logprobs_values_reward(training_input)

    [forward] = trainer.policy_model.forward_batches
    assert forward[ROLLOUT_ENGINE_DP_RANKS_KEY].tolist() == [3, 0, UNKNOWN_ENGINE_DP_RANK]
    # The exact trainer needs the rank of each row with loss tokens.
    assert torch.equal(forward["loss_mask"], training_input["loss_mask"])
