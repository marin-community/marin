import pytest
import ray.exceptions

from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.base import InferenceEngineInput, InferenceEngineOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import hash_with_sha256
from skyrl_train.trajectory_runners.types import ENGINE_DP_RANKS_KEY


class _Engine:
    """An engine that answers every prompt with the one token ``token``, or fails as a dead Ray actor."""

    def __init__(self, token, dead=False):
        self.token = token
        self.dead = dead

    async def generate(self, request):
        if self.dead:
            raise ray.exceptions.RayActorError()
        rows = len(request["prompt_token_ids"])
        return InferenceEngineOutput(
            responses=["answer"] * rows,
            response_ids=[[self.token]] * rows,
            stop_reasons=["stop"] * rows,
            response_logprobs=[[-0.1]] * rows,
        )


@pytest.mark.asyncio
async def test_generate_on_live_engines_skips_a_dead_engine_and_returns_each_engines_rank_in_its_group():
    cfg = get_default_config()
    cfg.generator.enable_http_endpoint = False
    cfg.generator.inference_engine_data_parallel_size = 2
    # Two engine groups of two data-parallel ranks each. A request routed to the first engine finds it dead.
    engines = [_Engine(10, dead=True), _Engine(11), _Engine(12), _Engine(13)]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=cfg)
    session = next(s for s in (f"trial-{i}" for i in range(20)) if hash_with_sha256(s) % len(engines) == 0)
    served = await client.generate(
        InferenceEngineInput(prompt_token_ids=[[1, 2, 3]], sampling_params={"max_tokens": 5}, session_ids=[session])
    )
    # The retried request carries the rank of the live engine that answered it.
    assert served[ENGINE_DP_RANKS_KEY] == [{11: 1, 12: 0, 13: 1}[served["response_ids"][0][0]]]

    outputs = await client.generate_on_live_engines([1, 2, 3], {"max_tokens": 5})

    assert [(rank, output["response_ids"]) for rank, output in outputs] == [(1, [[11]]), (0, [[12]]), (1, [[13]])]

    # One group of four ranks, so every live engine's rank differs from the dead one's: a batched request routed to the
    # dead engine is retried on a live one, and both rows carry the rank of the engine that answered them.
    cfg.generator.inference_engine_data_parallel_size = 4
    engines = [_Engine(20, dead=True), _Engine(21), _Engine(22), _Engine(23)]
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=cfg)
    sessions = [s for s in (f"trial-{i}" for i in range(40)) if hash_with_sha256(s) % len(engines) == 0][:2]
    served = await client.generate(
        InferenceEngineInput(prompt_token_ids=[[1], [2]], sampling_params={"max_tokens": 5}, session_ids=sessions)
    )
    assert served[ENGINE_DP_RANKS_KEY] == [row[0] - 20 for row in served["response_ids"]]
