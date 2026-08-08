# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify Snowball checkpoint inference against representative-prompt goldens.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

Run from the repository root:
    uv run pytest tests/cluster/vllm/test_snowball_levanter_inference.py \
      -m cluster -o addopts= --import-mode=importlib -vv -s
"""

import dataclasses
import uuid

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from haliax.partitioning import set_mesh
from huggingface_hub import snapshot_download
from iris.client import IrisClient
from iris.rpc import job_pb2
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer

from tests.cluster.vllm.backend_parity import (
    NextTokenObservation,
    ParityReport,
    TokenScore,
    persist_and_validate_exact_report,
)
from tests.cluster.vllm.snowball import (
    SNOWBALL_TPU,
    TOP_K,
    TPU_PROMPT_FIXTURE_URL,
    TPU_REPORT_ROOT,
    RepresentativeGolden,
    RepresentativePromptFixture,
    pad_prompt_batch,
    read_prompt_fixture,
    read_representative_goldens,
    read_tpu_representative_goldens,
)
from tests.cluster.vllm.snowball_checkpoint import (
    VendoredTransformer,
    apply_pending_qb_betas,
    decode_vendored_config,
    load_checkpoint,
    prepare_bf16_parameters,
    read_executor_info,
)

PENDING_TIMEOUT = 5 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0
TPU_PENDING_TIMEOUT = 30 * 60.0
TPU_RUNTIME_TIMEOUT = 60 * 60.0
JAX_COMPILATION_CACHE_DIR = (
    "s3://marin-us-east-02a/tmp/ttl=30d/compilation-cache/june-tpu-67b-a2b-step-42150-sonic-fa4-representative-v2"
)
TPU_JAX_COMPILATION_CACHE_DIR = (
    "gs://marin-us-east5/tmp/ttl=30d/compilation-cache/snowball-step-42150-levanter-native-v1"
)

pytestmark = [pytest.mark.cluster, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


@eqx.filter_jit
def top_k_next_token_logprobs(
    model: VendoredTransformer,
    pending_qb_betas: jax.Array,
    token_ids: jax.Array,
    last_token_indices: jax.Array,
    policy: jmp.Policy,
) -> tuple[jax.Array, jax.Array]:
    """Project only each row's last real token, never the full sequence vocabulary."""
    model = apply_pending_qb_betas(model, pending_qb_betas)
    model = policy.cast_to_compute(model)
    return _top_k_from_parameters(model, token_ids, last_token_indices)


def _top_k_from_parameters(
    model: VendoredTransformer,
    token_ids: jax.Array,
    last_token_indices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    hidden, _ = model(token_ids)
    last_hidden = hidden.at[jnp.arange(token_ids.shape[0]), last_token_indices].get(
        out_sharding=P(("replica_dcn", "data", "expert"))
    )
    logits = jnp.einsum(
        "bh,hv->bv",
        last_hidden,
        model.output_proj,
        out_sharding=P(("replica_dcn", "data", "expert")),
    )
    assert logits.dtype == jnp.bfloat16
    logprobs = jax.nn.log_softmax(logits.astype(jnp.float32))
    return jax.lax.top_k(logprobs, TOP_K)


@eqx.filter_jit
def top_k_prepared_next_token_logprobs(
    model: VendoredTransformer,
    token_ids: jax.Array,
    last_token_indices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _top_k_from_parameters(model, token_ids, last_token_indices)


def _load_tokenizer(prompt_fixture: RepresentativePromptFixture):
    tokenizer = load_tokenizer(
        snapshot_download(
            prompt_fixture.tokenizer,
            revision=prompt_fixture.tokenizer_revision,
            allow_patterns=["tokenizer*", "special_tokens*", "added_tokens*", "chat_template*"],
        )
    )
    assert tokenizer.eos_token_id is not None
    return tokenizer


def compute_checkpoint_inference(
    prompt_fixture: RepresentativePromptFixture,
) -> dict[str, tuple[TokenScore, ...]]:
    """Load one checkpoint and return structured results for production batches."""
    executor_info = read_executor_info()
    assert executor_info["config"]["data"]["tokenizer"] == prompt_fixture.tokenizer
    inference_model_config = dataclasses.replace(
        decode_vendored_config(executor_info),
        moe_implementation="sonic",
        # The checkpoint leaves this unset, selecting quadratic reference attention on GPU, which cannot fit 32K.
        attention_implementation="gpu_fa4_cute",
    )

    policy = jmp.get_policy(executor_info["config"]["mp"])
    tokenizer = _load_tokenizer(prompt_fixture)

    mesh = compact_grug_mesh()
    assert mesh.shape.get("expert", 1) == 1
    with set_mesh(mesh):
        params, pending_qb_betas = load_checkpoint(inference_model_config, mesh)

        computed_cases = {}
        for batch in prompt_fixture.batches:
            token_ids, last_token_indices = pad_prompt_batch(batch, tokenizer.eos_token_id)
            top_logprobs, top_token_ids = top_k_next_token_logprobs(
                params,
                pending_qb_betas,
                jnp.asarray(token_ids),
                jnp.asarray(last_token_indices),
                policy,
            )
            top_logprobs = np.asarray(jax.device_get(top_logprobs))
            top_token_ids = np.asarray(jax.device_get(top_token_ids))
            for row, case in enumerate(batch.cases):
                computed_cases[case.id] = tuple(
                    TokenScore(
                        logprob=float(logprob),
                        token_id=int(token_id),
                    )
                    for logprob, token_id in zip(top_logprobs[row], top_token_ids[row], strict=True)
                )

    return computed_cases


def compute_tpu_checkpoint_inference(
    prompt_fixture: RepresentativePromptFixture,
) -> dict[str, tuple[TokenScore, ...]]:
    """Load the regional checkpoint as BF16 and score every TPU golden case."""
    executor_info = read_executor_info(SNOWBALL_TPU)
    assert executor_info["config"]["data"]["tokenizer"] == prompt_fixture.tokenizer
    inference_model_config = dataclasses.replace(
        decode_vendored_config(executor_info),
        moe_implementation="ring",
        attention_implementation="tpu_splash",
    )
    tokenizer = _load_tokenizer(prompt_fixture)

    mesh = compact_grug_mesh()
    assert mesh.shape.get("expert", 1) == 1
    with set_mesh(mesh):
        params, pending_qb_betas = load_checkpoint(
            inference_model_config,
            mesh,
            model=SNOWBALL_TPU,
            parameter_dtype=jnp.bfloat16,
        )
        params = prepare_bf16_parameters(params, pending_qb_betas)
        del pending_qb_betas

        computed_cases = {}
        for batch in prompt_fixture.batches:
            token_ids, last_token_indices = pad_prompt_batch(batch, tokenizer.eos_token_id)
            top_logprobs, top_token_ids = top_k_prepared_next_token_logprobs(
                params,
                jnp.asarray(token_ids),
                jnp.asarray(last_token_indices),
            )
            top_logprobs = np.asarray(jax.device_get(top_logprobs))
            top_token_ids = np.asarray(jax.device_get(top_token_ids))
            for row, case in enumerate(batch.cases):
                computed_cases[case.id] = tuple(
                    TokenScore(logprob=float(logprob), token_id=int(token_id))
                    for logprob, token_id in zip(top_logprobs[row], top_token_ids[row], strict=True)
                )

    return computed_cases


def assert_checkpoint_inference_matches_golden(
    expected_cases: tuple[RepresentativeGolden, ...],
) -> None:
    prompt_fixture = read_prompt_fixture(expected_cases)
    actual_cases = compute_checkpoint_inference(prompt_fixture)
    for expected in expected_cases:
        assert actual_cases[expected.id] == expected.top_logprobs, expected.id


def assert_tpu_checkpoint_inference_matches_golden(
    expected_cases: tuple[RepresentativeGolden, ...],
    report_uri: str,
) -> None:
    prompt_fixture = read_prompt_fixture(expected_cases, fixture_url=TPU_PROMPT_FIXTURE_URL)
    actual_cases = compute_tpu_checkpoint_inference(prompt_fixture)
    report = ParityReport(
        backend="levanter-native-tpu",
        observations=tuple(
            NextTokenObservation(
                case_id=case.id,
                backend_rank=0,
                emitted_token_id=actual_cases[case.id][0].token_id,
                returned_top_logprobs=actual_cases[case.id],
            )
            for case in expected_cases
        ),
    )
    persist_and_validate_exact_report(
        report,
        report_uri,
        {case.id: case.top_logprobs for case in expected_cases},
    )


def test_snowball_checkpoint_matches_levanter_inference_goldens(marin_gpu_client: IrisClient, run_test_job) -> None:
    expected_cases = read_representative_goldens()
    run_test_job(
        marin_gpu_client,
        JobRequest(
            name=f"snowball-checkpoint-inference-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_checkpoint_inference_matches_golden,
                args=[expected_cases],
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


@pytest.mark.timeout(TPU_PENDING_TIMEOUT + TPU_RUNTIME_TIMEOUT + 300)
def test_snowball_tpu_checkpoint_matches_levanter_inference_goldens(
    iris_client: IrisClient,
    run_test_job,
) -> None:
    expected_cases = read_tpu_representative_goldens()
    run_id = uuid.uuid4().hex
    run_test_job(
        iris_client,
        JobRequest(
            name=f"snowball-tpu-checkpoint-{run_id[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_tpu_checkpoint_inference_matches_golden,
                args=[expected_cases, f"{TPU_REPORT_ROOT}/{run_id}/levanter-native-tpu.json"],
            ),
            resources=ResourceConfig.with_tpu(
                "v6e-8",
                cpu=64,
                ram="256g",
                disk="100g",
                regions=("us-east5",),
            ),
            environment=create_environment(
                extras=["tpu"],
                sync_packages=["marin-levanter", "marin-core"],
                env_vars={
                    "JAX_COMPILATION_CACHE_DIR": TPU_JAX_COMPILATION_CACHE_DIR,
                    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                },
            ),
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        pending_timeout=TPU_PENDING_TIMEOUT,
        runtime_timeout=TPU_RUNTIME_TIMEOUT,
    )
