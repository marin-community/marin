from types import SimpleNamespace

import pytest
import ray
import torch

from skyrl_train.config.utils import get_default_config
from skyrl_train.exactness_check import ExactnessCheck
from skyrl_train.weight_sync.base import WeightSyncReason

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
    check = ExactnessCheck(2, get_default_config().generator.sampling_params, tokenizer)
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
