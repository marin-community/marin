# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare Snowball serving backends against representative goldens.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

The 64 content-addressed prompts span short through 32K-context workloads and
are evaluated in the same production-shaped buckets as the checkpoint test.
Levanter projects only each prompt's last hidden state. vLLM receives the exact
token IDs through its OpenAI completions endpoint, with one concurrent request
pinned to each data-parallel rank.

Run the complete standing-cluster gate from the repository root only after
interactive H100 validation::

    uv run pytest tests/cluster/vllm/test_snowball_backend_parity.py \
      -m cluster -o addopts= --import-mode=importlib -vv -s
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client import IrisClient
from iris.cluster.setup_scripts import default_setup_script
from iris.rpc import job_pb2

from tests.cluster.vllm.backend_parity import (
    NextTokenParity,
    parity_from_logprob_map,
    parity_from_logprob_row,
)
from tests.cluster.vllm.snowball import (
    BATCH_SIZE,
    MAX_PROBABILITY_ERROR,
    SNOWBALL,
    RepresentativeCase,
    RepresentativeGolden,
    pad_prompt_batch,
    read_prompt_fixture,
    read_representative_goldens,
)

logger = logging.getLogger(__name__)

PENDING_TIMEOUT = 30 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0
HTTP_CONNECT_TIMEOUT = 30.0
HTTP_READ_TIMEOUT = 5 * 60.0
GPU_COUNT = 8
MAX_MODEL_LEN = 32768
MAX_NUM_BATCHED_TOKENS = 512
RETURNED_LOGPROBS = 50
MOE_IMPLEMENTATION = "sonic"
ATTENTION_IMPLEMENTATION = "gpu_fa4_cute"

pytestmark = [pytest.mark.cluster, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 300)]


def _log_parities(backend: str, parities: list[NextTokenParity]) -> None:
    logger.info(
        "%s parity vs representative Grug goldens:\n%s",
        backend,
        "\n".join(
            f"  case={parity.case_id} rank={parity.backend_rank} greedy={parity.greedy_token_id} "
            f"greedy_gap={parity.golden_probability_gap_to_greedy:.6f} "
            f"max_prob_err={parity.max_probability_error:.6f} l1={parity.top_probability_l1_error:.6f}"
            for parity in sorted(parities, key=lambda item: (item.case_id, item.backend_rank))
        ),
    )


def score_levanter_against_goldens(goldens: tuple[RepresentativeGolden, ...]) -> None:
    """Load the June export once and score every prompt through Levanter."""
    import haliax as hax  # noqa: PLC0415 -- entrypoint runs in the remote job's interpreter
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from haliax import Axis  # noqa: PLC0415
    from jax.sharding import PartitionSpec as P  # noqa: PLC0415
    from levanter.models.snowball import SnowballLMHeadModel  # noqa: PLC0415
    from marin.inference.backend import ModelSpec  # noqa: PLC0415
    from marin.inference.config import LevanterEngineConfig  # noqa: PLC0415
    from marin.inference.levanter_backend import LevanterBackend  # noqa: PLC0415
    from marin.inference.model_preparation import read_attention_heads, select_tensor_parallel_size  # noqa: PLC0415

    prompt_fixture = read_prompt_fixture(goldens)
    num_chips = jax.device_count()
    assert num_chips == GPU_COUNT, f"expected {GPU_COUNT} H100s, found {num_chips} devices"
    num_heads, num_kv_heads = read_attention_heads(SNOWBALL.export_uri)
    tensor_parallel_size = select_tensor_parallel_size(num_heads, num_chips, num_kv_heads)
    assert tensor_parallel_size == 1, (num_heads, num_kv_heads, num_chips)

    spec = ModelSpec(
        model=SNOWBALL.model_name,
        model_path=SNOWBALL.export_uri,
        num_chips=num_chips,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        chat_template_content=None,
    )
    config_overrides = {
        "moe_implementation": MOE_IMPLEMENTATION,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
    }
    with LevanterBackend(LevanterEngineConfig()).load_model(spec, config_overrides=config_overrides) as loaded:
        model = loaded.model
        assert isinstance(model, SnowballLMHeadModel)
        assert model.transformer.token_embed.dtype == jnp.bfloat16
        assert loaded.trainer.data_axis_size == BATCH_SIZE
        assert loaded.tokenizer.eos_token_id is not None
        logger.info("Levanter compilation cache: %s", jax.config.jax_compilation_cache_dir)

        @hax.named_jit(axis_resources=loaded.trainer.compute_axis_mapping)
        def next_token_logprobs(m: SnowballLMHeadModel, ids, last_positions) -> jax.Array:
            hidden = m.activations(ids).rearrange(("batch", "position", "embed")).array
            last_hidden = hidden.at[jnp.arange(hidden.shape[0]), last_positions.array].get(out_sharding=P("data"))
            logits = jnp.einsum(
                "bh,hv->bv",
                last_hidden,
                m.transformer.output_proj,
                out_sharding=P("data"),
            )
            assert logits.dtype == jnp.bfloat16
            return jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)

        parities: list[NextTokenParity] = []
        Batch = Axis("batch", BATCH_SIZE)
        for batch_index, batch in enumerate(prompt_fixture.batches):
            assert len(batch.cases) == BATCH_SIZE
            logger.info(
                "Levanter batch %d/%d: max_tokens=%d cases=%s",
                batch_index + 1,
                len(prompt_fixture.batches),
                batch.max_tokens,
                [case.id for case in batch.cases],
            )
            token_ids, last_token_indices = pad_prompt_batch(batch, loaded.tokenizer.eos_token_id)

            Pos = Axis("position", batch.max_tokens)
            input_ids = hax.named(jnp.asarray(token_ids), (Batch, Pos))
            last_positions = hax.named(jnp.asarray(last_token_indices), (Batch,))
            logprobs = np.asarray(jax.device_get(next_token_logprobs(model, input_ids, last_positions)))
            for row, case in enumerate(batch.cases):
                parities.append(
                    parity_from_logprob_row(
                        case.id,
                        case.top_logprobs,
                        logprobs[row],
                        backend_rank=row,
                    )
                )

    assert len(parities) == len(prompt_fixture.cases)
    _log_parities("levanter-gpu", parities)
    for parity in parities:
        parity.assert_matches(max_probability_error=MAX_PROBABILITY_ERROR)


def score_vllm_against_goldens(
    goldens: tuple[RepresentativeGolden, ...],
    attention_backend: str,
) -> None:
    """Serve the June export with vLLM and score rank-pinned prompts."""
    import requests  # noqa: PLC0415
    from marin.inference.backend import OPENAI_API_SUFFIX, ModelSpec  # noqa: PLC0415
    from marin.inference.config import (  # noqa: PLC0415
        VllmEngineConfig,
        VllmLauncherType,
        VllmSource,
    )
    from marin.inference.vllm_backend import VllmBackend  # noqa: PLC0415

    prompt_fixture = read_prompt_fixture(goldens)

    spec = ModelSpec(
        model=SNOWBALL.model_name,
        model_path=SNOWBALL.export_uri,
        num_chips=GPU_COUNT,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        chat_template_content=None,
    )
    backend = VllmBackend(
        VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            source=VllmSource.MARIN_FORK,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            extra_args=(
                "--data-parallel-size",
                str(GPU_COUNT),
                "--enable-expert-parallel",
                "--model-loader-extra-config",
                '{"distributed":true}',
                "--max-num-seqs",
                "1",
                "--max-logprobs",
                str(RETURNED_LOGPROBS),
                "--attention-backend",
                attention_backend,
            ),
        ),
    )
    parities: list[NextTokenParity] = []
    with backend.serve(spec) as served:
        completions_url = f"{served.base_url}{OPENAI_API_SUFFIX}/completions"

        def request_case(case: RepresentativeCase, rank: int, request_id: str) -> NextTokenParity:
            context = f"case={case.id} rank={rank} request={request_id}"
            assert len(case.prompt_token_ids) + 1 <= MAX_MODEL_LEN, context
            try:
                response = requests.post(
                    completions_url,
                    headers={
                        "X-data-parallel-rank": str(rank),
                        "X-Request-Id": request_id,
                    },
                    json={
                        "model": served.model_id,
                        "prompt": list(case.prompt_token_ids),
                        "add_special_tokens": False,
                        "temperature": 0.0,
                        "max_tokens": 1,
                        "logprobs": RETURNED_LOGPROBS,
                        "return_tokens_as_token_ids": True,
                        "return_token_ids": True,
                    },
                    timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
                )
                response.raise_for_status()
                (choice,) = response.json()["choices"]
                assert choice["prompt_token_ids"] == list(case.prompt_token_ids)
                (greedy_token_id,) = choice["token_ids"]
                (returned_top_logprobs,) = choice["logprobs"]["top_logprobs"]
                actual_logprobs = {
                    int(token.removeprefix("token_id:")): float(logprob)
                    for token, logprob in returned_top_logprobs.items()
                }
                return parity_from_logprob_map(
                    case.id,
                    case.top_logprobs,
                    int(greedy_token_id),
                    actual_logprobs,
                    backend_rank=rank,
                )
            except Exception as error:
                error.add_note(context)
                raise

        def request_wave(
            executor: ThreadPoolExecutor,
            cases: tuple[RepresentativeCase, ...],
            request_prefix: str,
        ) -> list[NextTokenParity]:
            assert len(cases) == GPU_COUNT
            futures = [
                executor.submit(request_case, case, rank, f"{request_prefix}-{case.id}-rank-{rank}")
                for rank, case in enumerate(cases)
            ]
            return [future.result() for future in as_completed(futures)]

        with ThreadPoolExecutor(max_workers=GPU_COUNT) as executor:
            for wave, batch in enumerate(prompt_fixture.batches):
                logger.info(
                    "vLLM wave %d/%d: max_tokens=%d cases=%s",
                    wave + 1,
                    len(prompt_fixture.batches),
                    batch.max_tokens,
                    [case.id for case in batch.cases],
                )
                parities.extend(request_wave(executor, batch.cases, f"wave-{wave}"))

            sentinel = next(case for case in prompt_fixture.cases if case.id == "knowledge-longbench-02")
            assert len(sentinel.prompt_token_ids) > 2048
            logger.info("vLLM rank sentinel: case=%s tokens=%d", sentinel.id, len(sentinel.prompt_token_ids))
            parities.extend(request_wave(executor, (sentinel,) * GPU_COUNT, "rank-sentinel"))

    assert len(parities) == len(prompt_fixture.cases) + GPU_COUNT
    _log_parities("vllm-gpu", parities)
    for parity in parities:
        parity.assert_matches(max_probability_error=MAX_PROBABILITY_ERROR)


def _levanter_job(goldens: tuple[RepresentativeGolden, ...]) -> JobRequest:
    return JobRequest(
        name=f"snowball-parity-levanter-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(score_levanter_against_goldens, args=[goldens]),
        resources=ResourceConfig.with_gpu("H100", count=GPU_COUNT, cpu=64, ram="256g", disk="128g"),
        environment=create_environment(
            extras=["gpu"],
            sync_packages=["marin-levanter", "marin-core"],
            env_vars={
                "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
            },
        ),
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
    )


def _vllm_job(goldens: tuple[RepresentativeGolden, ...], attention_backend: str) -> JobRequest:
    return JobRequest(
        name=f"snowball-parity-vllm-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(score_vllm_against_goldens, args=[goldens, attention_backend]),
        resources=ResourceConfig.with_gpu("H100", count=GPU_COUNT, cpu=64, ram="512g", disk="128g"),
        environment=create_environment(
            setup_scripts=[default_setup_script(packages=["marin-core"])],
            env_vars={
                "VLLM_BATCH_INVARIANT": "1",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
            },
        ),
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
    )


@pytest.mark.parametrize("backend", ["levanter-gpu", "vllm-gpu"])
def test_snowball_export_matches_representative_goldens(
    marin_gpu_client: IrisClient,
    backend: str,
    vllm_attention_backend: str,
    run_test_job,
) -> None:
    goldens = read_representative_goldens()
    request = _levanter_job(goldens) if backend == "levanter-gpu" else _vllm_job(goldens, vllm_attention_backend)
    run_test_job(
        marin_gpu_client,
        request,
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
