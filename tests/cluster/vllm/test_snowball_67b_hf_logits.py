# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the Snowball Levanter model against the June 67B A2B frozen golden.

This is the full-model parity gate: load the exact HF BF16 ``grug_moe`` export into the first-class
Levanter ``SnowballLMHeadModel`` and assert its next-token distribution matches the committed golden.
Unlike ``test_june_67b_a2b_levanter_inference.py`` (which restores the native TensorStore checkpoint
into the vendored training Transformer), this exercises the HF *load* path
that ``marin-serve`` uses, on the exported artifact.

The golden was produced on 8xH100 with ``moe_implementation="sonic"`` and
``--xla_gpu_deterministic_ops=true``. Snowball is a *reimplementation* of that graph, not the same
graph, so bf16 reduction-order noise reorders the golden's many exact-tied tail tokens (its
logprobs sit on a 1/32 grid). We therefore assert parity the way the vLLM export test does: the
greedy token matches exactly, and the probability error on the golden's token set stays within
``MAX_PROBABILITY_ERROR`` — a tie-insensitive bar. The export already has the pending QB betas baked
into ``router.bias``, so Snowball loads them as-is (no re-application).

Marked ``cluster`` (submits an H100 job to the standing CoreWeave cluster) so it is deselected by
default; the ``marin-cluster-smoke`` workflow runs it. Launch it on demand with::

    uv run pytest tests/cluster/vllm/test_snowball_67b_hf_logits.py \
      -m cluster -o addopts= --import-mode=importlib -vv -s
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

from tests.cluster.vllm.backend_parity import LEVANTER_MAX_PROBABILITY_ERROR, parity_from_logprob_row
from tests.cluster.vllm.june_67b_a2b import JUNE_67B_A2B, InferenceGolden, read_inference_golden

logger = logging.getLogger(__name__)

PENDING_TIMEOUT = 5 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0
TOP_K = 25
JAX_COMPILATION_CACHE_DIR = (
    "s3://marin-us-east-02a/tmp/ttl=30d/compilation-cache/snowball-67b-a2b-step-42150-sonic-deterministic-v1"
)

pytestmark = [pytest.mark.cluster, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


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
        def next_token_logprobs(m: SnowballLMHeadModel, ids) -> jax.Array:
            logits = m(ids)  # {batch, position, vocab}
            assert logits.dtype == jnp.bfloat16
            last = logits["position", -1].array.astype(jnp.float32)  # [batch, vocab]
            return jax.nn.log_softmax(last, axis=-1)

        infer_started = time.perf_counter()
        logprobs = next_token_logprobs(model, input_ids)  # [batch, vocab]
        jax.block_until_ready(logprobs)
        infer_elapsed = time.perf_counter() - infer_started

    logprobs = np.asarray(jax.device_get(logprobs))

    # Greedy-token match + rank-independent probability parity on the golden's token set (insensitive
    # to the bf16 tie reordering, meaningful against a real regression), one row per device.
    parities = [parity_from_logprob_row(golden, row) for row in logprobs]
    logger.info(
        "Snowball HF-export inference: %s",
        {
            "hf_load_seconds": load_elapsed,
            "logical_gib": logical_gib,
            "compile_and_inference_seconds": infer_elapsed,
            "greedy_token": tokenizer.decode([parities[0].greedy_token_id]),
            "max_probability_error_vs_golden": max(p.max_probability_error for p in parities),
        },
    )
    # Rank 0 sits 3.98 nats clear of rank 1, so the greedy token must match on every device.
    for parity in parities:
        parity.assert_matches(max_probability_error=LEVANTER_MAX_PROBABILITY_ERROR)


def test_snowball_h100_hf_export_matches_golden(marin_gpu_client: IrisClient, run_test_job) -> None:
    golden = read_inference_golden(JUNE_67B_A2B.inference_golden_path)
    run_test_job(
        marin_gpu_client,
        JobRequest(
            name=f"snowball-67b-hf-logits-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_snowball_hf_export_matches_golden,
                args=[golden],
            ),
            # 8xH100 nodes have 128 vCPU / 2 TB and (cw-ib TAS) the whole pod must fit on one node,
            # so request only what the load needs: modest CPU + ~134 GB host peak for the BF16 state
            # dict, leaving CPU headroom for the node's system / NHC-verification pods.
            resources=ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="256g", disk="128g"),
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
