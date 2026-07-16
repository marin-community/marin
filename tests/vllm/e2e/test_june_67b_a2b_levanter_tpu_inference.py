# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify June 67B A2B Levanter inference on one v6e-8.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

Run from the repository root:
    uv run pytest tests/vllm/e2e/test_june_67b_a2b_levanter_tpu_inference.py -o addopts= -vv -s
"""

import uuid

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from haliax.partitioning import set_mesh
from iris.client import IrisClient
from iris.rpc import job_pb2
from jax.sharding import PartitionSpec as P
from levanter.grug.grug_moe import resolve_moe_implementation
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer

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

PENDING_TIMEOUT = 5 * 60.0
RUNTIME_TIMEOUT = 10 * 60.0
# Splash requires a 128-token input; score the final prompt token before its zero suffix.
SEQUENCE_LENGTH = 128
SCORED_POSITION = 3
TOP_K = 25

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


@eqx.filter_jit
def top_k_tpu_logprobs(
    model: VendoredTransformer,
    pending_qb_betas: jax.Array,
    token_ids: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    model = apply_pending_qb_betas(model, pending_qb_betas)
    model = jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        model,
    )
    hidden, router_metrics = model(token_ids)
    logits = jnp.einsum(
        "bsh,hv->bsv",
        hidden[:, SCORED_POSITION : SCORED_POSITION + 1],
        model.output_proj,
        out_sharding=P(("replica_dcn", "data", "expert")),
    )[:, 0]
    logprobs = jax.nn.log_softmax(logits.astype(jnp.float32))
    top_logprobs, top_token_ids = jax.lax.top_k(logprobs, TOP_K)
    return top_logprobs, top_token_ids, router_metrics["capacity_overflow_per_layer"]


def assert_tpu_checkpoint_inference_matches_golden(expected: InferenceGolden) -> None:
    assert jax.default_backend() == "tpu"
    assert jax.device_count() == 8

    executor_info = read_executor_info(JUNE_67B_A2B.tpu_executor_info_path)
    model_config = decode_vendored_config(executor_info)
    # The golden records resolved model configuration, not the EP1 dispatch backend.
    assert resolve_moe_implementation(model_config.moe_implementation) == expected.moe_implementation
    assert executor_info["config"]["data"]["tokenizer"] == expected.tokenizer
    tokenizer = load_tokenizer(expected.tokenizer)
    prompt_token_ids = tokenizer.encode(expected.prompt, add_special_tokens=False)
    assert prompt_token_ids == expected.prompt_token_ids

    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=1)
    with set_mesh(mesh):
        params, pending_qb_betas = load_checkpoint(
            model_config,
            mesh,
            checkpoint_path=JUNE_67B_A2B.tpu_checkpoint_path,
            parameter_dtype=jnp.bfloat16,
        )
        parameter_dtypes = {leaf.dtype for leaf in jax.tree.leaves(params) if eqx.is_inexact_array(leaf)}
        assert parameter_dtypes == {jnp.dtype(jnp.bfloat16)}

        token_ids = jnp.zeros((jax.device_count(), SEQUENCE_LENGTH), dtype=jnp.int32)
        token_ids = token_ids.at[:, : len(prompt_token_ids)].set(jnp.asarray(prompt_token_ids, dtype=jnp.int32))
        top_logprobs, top_token_ids, capacity_overflow = top_k_tpu_logprobs(
            params,
            pending_qb_betas,
            token_ids,
        )

    top_logprobs = np.asarray(top_logprobs)
    top_token_ids = np.asarray(top_token_ids)
    capacity_overflow = np.asarray(capacity_overflow)
    expected_logprobs = np.asarray([entry.logprob for entry in expected.top_logprobs])
    expected_token_ids = np.asarray([entry.token_id for entry in expected.top_logprobs])

    np.testing.assert_array_equal(top_logprobs, np.broadcast_to(expected_logprobs, top_logprobs.shape))
    np.testing.assert_array_equal(top_token_ids, np.broadcast_to(expected_token_ids, top_token_ids.shape))
    np.testing.assert_array_equal(capacity_overflow, np.zeros_like(capacity_overflow))
    assert [tokenizer.decode([int(token_id)]) for token_id in top_token_ids[0]] == [
        entry.text for entry in expected.top_logprobs
    ]


def test_v6e_8_matches_levanter_tpu_inference_golden(marin_tpu_client: IrisClient) -> None:
    expected = read_inference_golden(JUNE_67B_A2B.tpu_inference_golden_path)
    run_remote_test_job(
        marin_tpu_client,
        JobRequest(
            name=f"june-67b-tpu-inference-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(assert_tpu_checkpoint_inference_matches_golden, args=[expected]),
            resources=ResourceConfig.with_tpu(
                "v6e-8",
                cpu=160,
                ram="640g",
                disk="64g",
                zone="us-east5-b",
            ),
            environment=create_environment(extras=["tpu"], sync_packages=["marin-levanter"]),
            # These e2es are manually triggered and highly interactive, so they use production priority.
            # Routine or automated workloads should not copy this priority.
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
