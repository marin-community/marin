# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic backend parity: score each marin-serve backend against the grug golden set.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

Parametrized over the two marin-serve backends that can serve the 67B ``grug_moe`` HF export:

- ``levanter-gpu`` -- ``LevanterBackend.load_model`` + a single forward (Snowball has no paged decode
  yet, so its full ``serve()`` generation path is separate work);
- ``vllm-gpu`` -- ``VllmBackend.serve()`` + the OpenAI ``/completions`` logprobs API. The 67B needs
  Marin's vLLM fork, installed onto the job venv so the default ``WorkspaceVllm`` launcher serves it.

Both load through ``marin.inference.serving_backend`` and are scored the same way -- for every golden
prompt the backend's next-token distribution is compared to the grug reference's frozen top-25: the
greedy token must match exactly, and the worst single-token probability error must stay within a
per-backend bound. See ``backend_parity`` for why the bound is looser for vLLM: the goldens are the
levanter reference, Snowball is a levanter reimplementation of it (only bf16 noise separates them),
and vLLM is a different framework serving the same weights (it diverges more on higher-entropy
prompts, though the greedy token still matches exactly everywhere).

Run from the repository root:
    uv run pytest tests/vllm/e2e/test_snowball_backend_parity.py -o addopts= -vv -s
"""

import logging
import os
import tempfile
import tomllib
import uuid
from pathlib import Path

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client import IrisClient
from iris.cluster.setup_scripts import default_setup_script
from iris.rpc import job_pb2

from .backend_parity import (
    LEVANTER_MAX_PROBABILITY_ERROR,
    VLLM_MAX_PROBABILITY_ERROR,
    NextTokenParity,
    parity_from_logprob_map,
    parity_from_logprob_row,
    read_golden_set,
)
from .june_67b_a2b import JUNE_67B_A2B, InferenceGolden
from .remote_job import run_remote_test_job

logger = logging.getLogger(__name__)

PENDING_TIMEOUT = 30 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0
GPU_COUNT = 8
RETURNED_LOGPROBS = 50  # >= the golden's top-25, so every golden token is always present.
MOE_IMPLEMENTATION = "sonic"  # the golden's backend; the memory-efficient MoE kernel for 256 experts.
# Right-pad every prompt to one length so the Snowball forward compiles once, not once per prompt
# length; the logits at a prompt's true last position are read back with a dynamic gather.
PAD_LEN = 16
PAD_TOKEN_ID = 0
LEVANTER_CACHE = "s3://marin-us-east-02a/tmp/ttl=30d/compilation-cache/snowball-67b-a2b-step-42150-parity-v1"

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


# TODO(#7135): drop this overlay for marin-core[vllm-gpu] once #7134 provides the managed baseline.
def _vllm_setup_script() -> str:
    source = tomllib.loads((Path(__file__).parents[3] / "pyproject.toml").read_text())["tool"]["uv"]["sources"]["vllm"]
    return (
        'VLLM_USE_PRECOMPILED=1 uv pip install --no-config --python "$IRIS_VENV/bin/python" '
        f'--torch-backend=cu130 "vllm @ git+{source["git"]}@{source["rev"]}" "runai-model-streamer[s3]==0.16.0"'
    )


def _log_parities(backend: str, parities: list[NextTokenParity]) -> None:
    logger.info(
        "%s parity vs grug goldens:\n%s",
        backend,
        "\n".join(
            f"  {p.prompt!r:60s} greedy={p.greedy_token_id} (golden {p.golden_greedy_token_id}) "
            f"max_prob_err={p.max_probability_error:.5f} l1={p.top_probability_l1_error:.5f}"
            for p in parities
        ),
    )


def score_levanter_against_goldens(goldens: list[InferenceGolden]) -> None:
    """Remote entrypoint: load the 67B via ``LevanterBackend`` and score every golden's next token."""
    import haliax as hax  # noqa: PLC0415 -- entrypoint runs in the remote job's interpreter
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from haliax import Axis  # noqa: PLC0415
    from levanter.models.snowball import SnowballLMHeadModel  # noqa: PLC0415
    from marin.inference.quick_serve import read_attention_heads, select_tensor_parallel_size  # noqa: PLC0415
    from marin.inference.serving_backend import LevanterBackend, ModelSpec  # noqa: PLC0415

    num_chips = jax.device_count()
    num_heads, num_kv_heads = read_attention_heads(JUNE_67B_A2B.export_uri)
    tensor_parallel_size = select_tensor_parallel_size(num_heads, num_chips, num_kv_heads)

    spec = ModelSpec(
        model="snowball-67b-a2b",
        model_path=JUNE_67B_A2B.export_uri,
        num_chips=num_chips,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=None,
        chat_template_content=None,
    )
    with LevanterBackend().load_model(spec, config_overrides={"moe_implementation": MOE_IMPLEMENTATION}) as loaded:
        model = loaded.model
        assert isinstance(model, SnowballLMHeadModel)
        assert model.transformer.token_embed.dtype == jnp.bfloat16
        tokenizer = loaded.tokenizer

        @hax.named_jit
        def next_token_logprobs(m: SnowballLMHeadModel, ids, last_pos) -> jax.Array:
            logits = m(ids)  # {batch, position, vocab}
            assert logits.dtype == jnp.bfloat16
            ordered = logits.rearrange(("batch", "position", ...)).array  # [batch, PAD_LEN, vocab]
            last = jnp.take_along_axis(ordered, last_pos[:, None, None], axis=1)[:, 0]  # [batch, vocab]
            return jax.nn.log_softmax(last.astype(jnp.float32), axis=-1)

        parities: list[NextTokenParity] = []
        for golden in goldens:
            prompt_token_ids = tokenizer.encode(golden.prompt, add_special_tokens=False)
            assert prompt_token_ids == golden.prompt_token_ids
            assert len(prompt_token_ids) <= PAD_LEN, (golden.prompt, len(prompt_token_ids))
            padded = prompt_token_ids + [PAD_TOKEN_ID] * (PAD_LEN - len(prompt_token_ids))
            # One prompt per device: the batch axis spans "data", matching how the golden was scored.
            Batch = Axis("batch", num_chips)
            Pos = Axis("position", PAD_LEN)
            input_ids = hax.named(jnp.asarray([padded] * num_chips, dtype=jnp.int32), (Batch, Pos))
            last_pos = jnp.full((num_chips,), len(prompt_token_ids) - 1, dtype=jnp.int32)
            logprobs = np.asarray(jax.device_get(next_token_logprobs(model, input_ids, last_pos)))  # [batch, vocab]
            parities.extend(parity_from_logprob_row(golden, row) for row in logprobs)

    _log_parities("levanter-gpu", parities)
    for parity in parities:
        parity.assert_matches(max_probability_error=LEVANTER_MAX_PROBABILITY_ERROR)


def score_vllm_against_goldens(goldens: list[InferenceGolden], attention_backend: str) -> None:
    """Remote entrypoint: serve the 67B via ``VllmBackend`` and score every golden's next token."""
    import requests  # noqa: PLC0415
    from marin.inference.serving_backend import OPENAI_API_SUFFIX, ModelSpec, VllmBackend  # noqa: PLC0415

    # Run:ai ignores Iris's FSSPEC_S3 setting and CoreWeave rejects path-style S3 requests.
    with tempfile.TemporaryDirectory(prefix="snowball-parity-vllm-") as temp_dir:
        aws_config = Path(temp_dir) / "aws-config"
        aws_config.write_text("[default]\ns3 =\n    addressing_style = virtual\n")
        os.environ["AWS_CONFIG_FILE"] = str(aws_config)

        spec = ModelSpec(
            model=JUNE_67B_A2B.vllm_model_name,
            model_path=JUNE_67B_A2B.export_uri,
            num_chips=GPU_COUNT,
            tensor_parallel_size=1,  # the 67B shards its experts with data + expert parallelism, not TP.
            dtype="bfloat16",
            max_model_len=128,
            chat_template_content=None,
        )
        # Default WorkspaceVllm launcher serves the fork installed on the job venv (see _vllm_setup_script).
        backend = VllmBackend(
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
        )
        parities: list[NextTokenParity] = []
        with backend.serve(spec) as served:
            completions_url = f"{served.base_url}{OPENAI_API_SUFFIX}/completions"
            for golden in goldens:
                for rank in range(GPU_COUNT):
                    response = requests.post(
                        completions_url,
                        # Pin every request: vLLM's DP load balancer would otherwise choose the rank.
                        headers={"X-data-parallel-rank": str(rank)},
                        json={
                            "model": served.model_id,
                            "prompt": golden.prompt,
                            "add_special_tokens": False,
                            "temperature": 0.0,
                            "max_tokens": 1,
                            "logprobs": RETURNED_LOGPROBS,
                            "return_tokens_as_token_ids": True,
                            "return_token_ids": True,
                        },
                        timeout=300,
                    )
                    response.raise_for_status()
                    choice = response.json()["choices"][0]
                    assert choice["prompt_token_ids"] == golden.prompt_token_ids
                    greedy_token_id = int(choice["token_ids"][0])
                    actual_logprobs = {
                        int(token.removeprefix("token_id:")): logprob
                        for token, logprob in choice["logprobs"]["top_logprobs"][0].items()
                    }
                    parities.append(parity_from_logprob_map(golden, greedy_token_id, actual_logprobs))

    _log_parities("vllm-gpu", parities)
    for parity in parities:
        parity.assert_matches(max_probability_error=VLLM_MAX_PROBABILITY_ERROR)


def _levanter_job(goldens: list[InferenceGolden]) -> JobRequest:
    return JobRequest(
        name=f"snowball-parity-levanter-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(score_levanter_against_goldens, args=[goldens]),
        resources=ResourceConfig.with_gpu("H100", count=GPU_COUNT, cpu=32, ram="256g", disk="128g"),
        environment=create_environment(
            extras=["gpu"],
            sync_packages=["marin-levanter", "marin-core"],
            env_vars={
                "JAX_COMPILATION_CACHE_DIR": LEVANTER_CACHE,
                "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
            },
        ),
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
    )


def _vllm_job(goldens: list[InferenceGolden], attention_backend: str) -> JobRequest:
    return JobRequest(
        name=f"snowball-parity-vllm-{uuid.uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(score_vllm_against_goldens, args=[goldens, attention_backend]),
        resources=ResourceConfig.with_gpu("H100", count=GPU_COUNT, cpu=64, ram="512g", disk="64g"),
        environment=create_environment(
            setup_scripts=[default_setup_script(packages=["marin-core"]), _vllm_setup_script()],
            env_vars={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        ),
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
    )


@pytest.mark.parametrize("backend", ["levanter-gpu", "vllm-gpu"])
def test_snowball_backend_matches_grug_goldens(
    marin_gpu_client: IrisClient,
    backend: str,
    vllm_attention_backend: str,
) -> None:
    goldens = read_golden_set()
    request = _levanter_job(goldens) if backend == "levanter-gpu" else _vllm_job(goldens, vllm_attention_backend)
    run_remote_test_job(
        marin_gpu_client,
        request,
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
