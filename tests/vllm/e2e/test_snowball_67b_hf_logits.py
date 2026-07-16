# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the Snowball Levanter model against the June 67B A2B frozen golden.

This is the full-model parity gate for #7219: load the exact HF BF16 ``grug_moe`` export into the
first-class Levanter ``SnowballLMHeadModel`` and assert its next-token top-25 log-probs match the
committed golden. Unlike ``test_june_67b_a2b_levanter_inference.py`` (which restores the native
TensorStore checkpoint into the vendored training Transformer), this exercises the HF *load* path
that ``marin-serve`` uses, on the exported artifact.

The golden was produced on 8xH100 with ``moe_implementation="sonic"`` and
``--xla_gpu_deterministic_ops=true``; we run on the same platform + pins so the atol=1e-5 tolerance
is a same-graph comparison. The export already has the pending QB betas baked into ``router.bias``,
so Snowball loads them as-is (no re-application).

Run from the repository root:
    uv run pytest tests/vllm/e2e/test_snowball_67b_hf_logits.py -o addopts= -vv -s
"""

import dataclasses
import logging
import time
import uuid

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from haliax import Axis
from haliax.partitioning import set_mesh
from iris.client import IrisClient
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballLMHeadModel
from levanter.tokenizers import load_tokenizer
from levanter.utils.jax_utils import parameter_count

from .june_67b_a2b import JUNE_67B_A2B, InferenceGolden, read_inference_golden
from .remote_job import run_remote_test_job

logger = logging.getLogger(__name__)

PENDING_TIMEOUT = 5 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0
TOP_K = 25
JAX_COMPILATION_CACHE_DIR = (
    "s3://marin-us-east-02a/tmp/ttl=30d/compilation-cache/snowball-67b-a2b-step-42150-sonic-deterministic-v1"
)

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


def assert_snowball_hf_export_matches_golden(golden: InferenceGolden) -> None:
    expected_top = golden.top_logprobs
    assert len(expected_top) == TOP_K

    mesh = compact_grug_mesh()
    with set_mesh(mesh):
        # Discover + load the HF BF16 export into Snowball. moe_implementation must match the
        # golden's backend ("sonic") for an exact-tolerance comparison on GPU.
        converter = HFCheckpointConverter.from_hf(JUNE_67B_A2B.export_uri)
        config = dataclasses.replace(converter.default_config, moe_implementation=golden.moe_implementation)

        load_started = time.perf_counter()
        model = converter.load_pretrained(
            SnowballLMHeadModel,
            ref=JUNE_67B_A2B.export_uri,
            config=config,
            dtype=jnp.bfloat16,
        )
        jax.block_until_ready(model)
        load_elapsed = time.perf_counter() - load_started
        gib = 1024**3
        logical_gib = parameter_count(model) * model.transformer.token_embed.dtype.itemsize / gib

        tokenizer = load_tokenizer(golden.tokenizer)
        prompt_token_ids = tokenizer.encode(golden.prompt, add_special_tokens=False)
        assert prompt_token_ids == golden.prompt_token_ids

        # The batch axis spans all eight GPUs, so run one prompt per device (matches the golden run).
        Batch = Axis("batch", jax.device_count())
        Pos = Axis("position", len(prompt_token_ids))
        input_ids = hax.named(
            jnp.asarray([prompt_token_ids] * jax.device_count(), dtype=jnp.int32),
            (Batch, Pos),
        )

        @hax.named_jit
        def top_k_next_token(m: SnowballLMHeadModel, ids) -> tuple[jax.Array, jax.Array]:
            logits = m(ids)  # {batch, position, vocab}
            assert logits.dtype == jnp.bfloat16
            last = logits["position", -1].array.astype(jnp.float32)  # [batch, vocab]
            logprobs = jax.nn.log_softmax(last, axis=-1)
            return jax.lax.top_k(logprobs, TOP_K)

        infer_started = time.perf_counter()
        top_logprobs, top_token_ids = top_k_next_token(model, input_ids)
        jax.block_until_ready(top_logprobs)
        infer_elapsed = time.perf_counter() - infer_started

    top_token_ids = np.asarray(jax.device_get(top_token_ids))
    top_logprobs = np.asarray(jax.device_get(top_logprobs))

    expected_token_ids = np.asarray([entry.token_id for entry in expected_top])
    expected_logprobs = np.asarray([entry.logprob for entry in expected_top])
    np.testing.assert_array_equal(top_token_ids, np.broadcast_to(expected_token_ids, top_token_ids.shape))
    np.testing.assert_allclose(top_logprobs, np.broadcast_to(expected_logprobs, top_logprobs.shape), rtol=0, atol=1e-5)
    assert [tokenizer.decode([int(tid)]) for tid in top_token_ids[0]] == [entry.text for entry in expected_top]
    logger.info(
        "Snowball HF-export inference: %s",
        {
            "hf_load_seconds": load_elapsed,
            "logical_gib": logical_gib,
            "compile_and_inference_seconds": infer_elapsed,
        },
    )


def test_snowball_h100_hf_export_matches_golden(marin_gpu_client: IrisClient) -> None:
    golden = read_inference_golden(JUNE_67B_A2B.inference_golden_path)
    run_remote_test_job(
        marin_gpu_client,
        JobRequest(
            name=f"snowball-67b-hf-logits-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_snowball_hf_export_matches_golden,
                args=[golden],
            ),
            resources=ResourceConfig.with_gpu("H100", count=8, cpu=160, ram="640g", disk="128g"),
            environment=create_environment(
                extras=["gpu"],
                sync_packages=["marin-levanter"],
                env_vars={
                    "JAX_COMPILATION_CACHE_DIR": JAX_COMPILATION_CACHE_DIR,
                    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                    "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
                },
            ),
        ),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
