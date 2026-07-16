# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Demonstrate loading the June 67B A2B ``grug_moe`` export through the marin-serve path.

This exercises the *serving* load path end-to-end on the real 67B: the exact
``marin.inference.serving_backend.LevanterBackend.load_model`` sequence ``quick_serve`` runs — build
the ``ModelSpec`` quick-serve would build (auto-selected tensor-parallel size, BF16), build the
serving mesh with ``AxisType.Explicit`` axes (Snowball requires them), and load the 67B weights
sharded across the slice — then score the next token against the frozen golden.

``moe_implementation="sonic"`` is passed through ``load_model``'s config-override hatch to match the
golden's Triton gather/combine backend. The golden was produced on 8xH100 with ``sonic`` and
``--xla_gpu_deterministic_ops=true``. Snowball is a reimplementation of that graph, so we assert
parity the tie-insensitive way (like the vLLM export test): the greedy token matches exactly and the
probability error on the golden's token set stays within ``MAX_PROBABILITY_ERROR``.

Run from the repository root:
    uv run pytest tests/vllm/e2e/test_snowball_67b_quick_serve_load.py -o addopts= -vv -s
"""

import logging
import time
import uuid

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from haliax import Axis
from iris.client import IrisClient
from levanter.models.snowball import SnowballLMHeadModel
from levanter.utils.jax_utils import parameter_count

from .june_67b_a2b import JUNE_67B_A2B, InferenceGolden, read_inference_golden
from .remote_job import run_remote_test_job

logger = logging.getLogger(__name__)

PENDING_TIMEOUT = 5 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0
TOP_K = 25
# Rank-independent parity bound (same as the vLLM export test): the golden's top logprobs sit on a
# 1/32 grid with exact ties, so a probability-error bound is the meaningful cross-implementation
# check. Snowball clears it with wide margin (observed max prob error < 0.001).
MAX_PROBABILITY_ERROR = 0.008
JAX_COMPILATION_CACHE_DIR = (
    "s3://marin-us-east-02a/tmp/ttl=30d/compilation-cache/snowball-67b-a2b-step-42150-quick-serve-v1"
)

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


def assert_snowball_quick_serve_load_matches_golden(golden: InferenceGolden) -> None:
    """Load the 67B export via LevanterBackend.load_model and score it against the golden.

    Runs inside the Iris job on the 8-GPU slice: builds the quick-serve ``ModelSpec``, loads the
    model through the marin-serve backend, and compares the next-token top-25 to the frozen golden.
    """
    import haliax as hax  # noqa: PLC0415 -- entrypoint runs in the remote job's interpreter
    from marin.inference.quick_serve import read_attention_heads, select_tensor_parallel_size  # noqa: PLC0415
    from marin.inference.serving_backend import LevanterBackend, ModelSpec  # noqa: PLC0415

    expected_top = golden.top_logprobs
    assert len(expected_top) == TOP_K

    num_chips = jax.device_count()
    # quick-serve auto-selects the tensor-parallel size; for the 67B (20 heads, 5 KV heads) on an
    # 8-GPU slice this is 1, so "data" spans all eight GPUs and FSDP-shards the experts.
    num_heads, num_kv_heads = read_attention_heads(JUNE_67B_A2B.export_uri)
    tensor_parallel_size = select_tensor_parallel_size(num_heads, num_chips, num_kv_heads)
    assert tensor_parallel_size == 1, tensor_parallel_size

    spec = ModelSpec(
        model="snowball-67b-a2b",
        model_path=JUNE_67B_A2B.export_uri,
        num_chips=num_chips,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=None,
        chat_template_content=None,
    )

    load_started = time.perf_counter()
    # sonic matches the golden's backend and is the memory-efficient MoE kernel for 256 experts on
    # GPU; the portable default (scatter) OOMs materializing a dense per-expert buffer.
    with LevanterBackend().load_model(spec, config_overrides={"moe_implementation": "sonic"}) as loaded:
        model = loaded.model
        assert isinstance(model, SnowballLMHeadModel)
        # BF16 truncation on load: the weights land as bfloat16 (cast per-shard on read).
        assert model.transformer.token_embed.dtype == jnp.bfloat16
        jax.block_until_ready(model)
        load_elapsed = time.perf_counter() - load_started
        gib = 1024**3
        logical_gib = parameter_count(model) * model.transformer.token_embed.dtype.itemsize / gib

        tokenizer = loaded.tokenizer
        prompt_token_ids = tokenizer.encode(golden.prompt, add_special_tokens=False)
        assert prompt_token_ids == golden.prompt_token_ids

        # One prompt per device: the batch axis spans "data", matching how the 67B is scored.
        Batch = Axis("batch", num_chips)
        Pos = Axis("position", len(prompt_token_ids))
        input_ids = hax.named(
            jnp.asarray([prompt_token_ids] * num_chips, dtype=jnp.int32),
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

    golden_ids = np.asarray([entry.token_id for entry in expected_top])
    golden_logprobs = np.asarray([entry.logprob for entry in expected_top])

    # Greedy token matches the golden on every device (rank 0 sits 3.98 nats clear of rank 1).
    greedy_ids = logprobs.argmax(axis=-1)
    # Rank-independent probability parity on the golden's token set (insensitive to bf16 tie reorder).
    mine_at_golden = logprobs[:, golden_ids]  # [batch, TOP_K]
    max_prob_error = float(np.max(np.abs(np.exp(mine_at_golden) - np.exp(golden_logprobs)[None, :])))
    logger.info(
        "Snowball quick-serve load: %s",
        {
            "hf_load_seconds": load_elapsed,
            "logical_gib": logical_gib,
            "compile_and_inference_seconds": infer_elapsed,
            "greedy_token": tokenizer.decode([int(greedy_ids[0])]),
            "max_probability_error_vs_golden": max_prob_error,
        },
    )

    np.testing.assert_array_equal(greedy_ids, np.broadcast_to(golden_ids[0], greedy_ids.shape))
    assert max_prob_error <= MAX_PROBABILITY_ERROR, max_prob_error


def test_snowball_h100_quick_serve_load_matches_golden(marin_gpu_client: IrisClient) -> None:
    golden = read_inference_golden(JUNE_67B_A2B.inference_golden_path)
    run_remote_test_job(
        marin_gpu_client,
        JobRequest(
            name=f"snowball-67b-quick-serve-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_snowball_quick_serve_load_matches_golden,
                args=[golden],
            ),
            # 8xH100 nodes have 128 vCPU / 2 TB and (cw-ib TAS) the whole pod must fit on one node,
            # so request only what the load needs: modest CPU + ~134 GB host peak for the BF16 state
            # dict, leaving CPU headroom for the node's system / NHC-verification pods.
            resources=ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="256g", disk="128g"),
            environment=create_environment(
                extras=["gpu"],
                # The entrypoint loads through marin-serve (marin-core) into Snowball (marin-levanter),
                # so sync both workspace members rather than just the model package.
                sync_packages=["marin-levanter", "marin-core"],
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
