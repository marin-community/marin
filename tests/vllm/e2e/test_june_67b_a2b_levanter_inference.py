# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify June 67B A2B Levanter inference against its frozen golden.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

Run from the repository root:
    uv run pytest tests/vllm/e2e/test_june_67b_a2b_levanter_inference.py -o addopts= -vv -s
"""

import dataclasses
import logging
import time
import uuid

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from haliax.partitioning import set_mesh
from iris.client import IrisClient
from iris.rpc import job_pb2
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer
from levanter.utils.jax_utils import parameter_count

from .june_67b_a2b import (
    JUNE_67B_A2B,
    InferenceGolden,
    VendoredTransformer,
    apply_pending_qb_betas,
    decode_vendored_config,
    load_checkpoint,
    read_executor_info,
    read_inference_golden,
)
from .remote_job import run_remote_test_job

logger = logging.getLogger(__name__)

PENDING_TIMEOUT = 5 * 60.0
RUNTIME_TIMEOUT = 10 * 60.0
TOP_K = 25
JAX_COMPILATION_CACHE_DIR = (
    "s3://marin-us-east-02a/tmp/ttl=30d/compilation-cache/june-tpu-67b-a2b-step-18000-sonic-deterministic-v1"
)

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


@eqx.filter_jit
def top_k_next_token_logprobs(
    model: VendoredTransformer,
    pending_qb_betas: jax.Array,
    token_ids: jax.Array,
    policy: jmp.Policy,
) -> tuple[jax.Array, jax.Array]:
    model = apply_pending_qb_betas(model, pending_qb_betas)
    model = policy.cast_to_compute(model)
    logits = model.logits(token_ids)
    assert logits.dtype == jnp.bfloat16
    logprobs = jax.nn.log_softmax(logits[:, -1].astype(jnp.float32))
    return jax.lax.top_k(logprobs, TOP_K)


def assert_checkpoint_inference_matches_golden(expected_inference: InferenceGolden) -> None:
    executor_info = read_executor_info()
    model_config = decode_vendored_config(executor_info)
    assert executor_info["config"]["mp"] == expected_inference.mp
    assert executor_info["config"]["data"]["tokenizer"] == expected_inference.tokenizer
    expected_top = expected_inference.top_logprobs
    assert len(expected_top) == TOP_K
    # With expert parallelism disabled, the source default falls back to nondeterministic scatter-add on GPU.
    inference_model_config = dataclasses.replace(model_config, moe_implementation=expected_inference.moe_implementation)

    mesh = compact_grug_mesh()
    with set_mesh(mesh):
        checkpoint_started = time.perf_counter()
        params, pending_qb_betas = load_checkpoint(inference_model_config, mesh)
        checkpoint_elapsed = time.perf_counter() - checkpoint_started
        gib = 1024**3
        checkpoint_logical_gib = parameter_count(params) * params.token_embed.dtype.itemsize / gib

        tokenizer = load_tokenizer(expected_inference.tokenizer)
        prompt_token_ids = tokenizer.encode(expected_inference.prompt, add_special_tokens=False)
        assert prompt_token_ids == expected_inference.prompt_token_ids
        # The batch axis spans all eight GPUs, so inference requires one prompt per device.
        token_ids = jnp.asarray([prompt_token_ids] * jax.device_count(), dtype=jnp.int32)
        policy = jmp.get_policy(expected_inference.mp)

        inference_started = time.perf_counter()
        top_logprobs, top_token_ids = top_k_next_token_logprobs(params, pending_qb_betas, token_ids, policy)
        jax.block_until_ready(top_logprobs)
        inference_elapsed = time.perf_counter() - inference_started

    top_token_ids = np.asarray(jax.device_get(top_token_ids))
    top_logprobs = np.asarray(jax.device_get(top_logprobs))

    expected_token_ids = np.asarray([entry.token_id for entry in expected_top])
    expected_logprobs = np.asarray([entry.logprob for entry in expected_top])
    np.testing.assert_array_equal(top_token_ids, np.broadcast_to(expected_token_ids, top_token_ids.shape))
    np.testing.assert_allclose(top_logprobs, np.broadcast_to(expected_logprobs, top_logprobs.shape), rtol=0, atol=1e-5)
    assert [tokenizer.decode([int(token_id)]) for token_id in top_token_ids[0]] == [entry.text for entry in expected_top]
    logger.info(
        "Checkpoint inference timing: %s",
        {
            "checkpoint_load_seconds": checkpoint_elapsed,
            "checkpoint_logical_gib": checkpoint_logical_gib,
            "compile_and_inference_seconds": inference_elapsed,
            "logical_gib_per_second": checkpoint_logical_gib / checkpoint_elapsed,
        },
    )


def test_h100_node_matches_levanter_inference_golden(marin_gpu_client: IrisClient) -> None:
    expected_inference = read_inference_golden(JUNE_67B_A2B.inference_golden_path)
    run_remote_test_job(
        marin_gpu_client,
        JobRequest(
            name=f"june-67b-checkpoint-inference-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_checkpoint_inference_matches_golden,
                args=[expected_inference],
            ),
            resources=ResourceConfig.with_gpu("H100", count=8, cpu=64, ram="256g", disk="64g"),
            environment=create_environment(
                extras=["gpu"],
                sync_packages=["marin-levanter"],
                env_vars={
                    "JAX_COMPILATION_CACHE_DIR": JAX_COMPILATION_CACHE_DIR,
                    # XLA's auxiliary caches require local paths; keep only JAX's LOTA-backed cache.
                    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                    # Keep BF16 kernel selection reproducible across independently compiled H100 nodes.
                    "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
                },
            ),
            # These e2es are manually triggered and highly interactive, so they use production priority.
            # Routine or automated workloads should not copy this priority.
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
