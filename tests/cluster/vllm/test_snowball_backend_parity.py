# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare Snowball serving backends against representative goldens.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

The 64 content-addressed prompts span short through 32K-context workloads and
are evaluated in the same production-shaped buckets as the checkpoint test.
vLLM receives the exact token IDs through its OpenAI completions endpoint,
with one concurrent request pinned to each data-parallel rank. Export-loaded
Levanter now uses the shared report/contract gate in
``test_snowball_exported_levanter_inference.py``.

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
)
from tests.cluster.vllm.snowball import (
    MAX_PROBABILITY_ERROR,
    SNOWBALL,
    SNOWBALL_GPU,
    VLLM_MAX_MODEL_LEN,
    VLLM_MAX_NUM_BATCHED_TOKENS,
    RepresentativeCase,
    RepresentativeGolden,
    read_prompt_fixture,
    read_representative_goldens,
)

logger = logging.getLogger(__name__)

PENDING_TIMEOUT = 30 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0
HTTP_CONNECT_TIMEOUT = 30.0
HTTP_READ_TIMEOUT = 5 * 60.0
GPU_COUNT = 8
RETURNED_LOGPROBS = 50

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
        model_path=SNOWBALL_GPU.export_uri,
        num_chips=GPU_COUNT,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=VLLM_MAX_MODEL_LEN,
        chat_template_content=None,
    )
    backend = VllmBackend(
        VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            source=VllmSource.MARIN_FORK,
            max_num_batched_tokens=VLLM_MAX_NUM_BATCHED_TOKENS,
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
            assert len(case.prompt_token_ids) + 1 <= VLLM_MAX_MODEL_LEN, context
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


def test_snowball_vllm_gpu_export_matches_representative_goldens(
    marin_gpu_client: IrisClient,
    vllm_attention_backend: str,
    run_test_job,
) -> None:
    goldens = read_representative_goldens()
    run_test_job(
        marin_gpu_client,
        _vllm_job(goldens, vllm_attention_backend),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
