# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify June 67B A2B Levanter inference against representative-prompt goldens.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

Run from the repository root:
    uv run pytest tests/cluster/vllm/test_june_67b_a2b_levanter_inference.py \
      -m cluster -o addopts= --import-mode=importlib -vv -s
"""

import dataclasses
import hashlib
import json
import uuid
from pathlib import Path

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
from rigging.filesystem import StoragePath

from tests.cluster.vllm.june_67b_a2b import (
    VendoredTransformer,
    apply_pending_qb_betas,
    decode_vendored_config,
    load_checkpoint,
    read_executor_info,
)

PENDING_TIMEOUT = 5 * 60.0
RUNTIME_TIMEOUT = 30 * 60.0
PROMPT_BUCKET_MAX_TOKENS = (256, 1024, 4096, 16384, 32768)
BATCH_SIZE = 8
TOP_K = 25
INFERENCE_GOLDEN_PATH = (
    Path(__file__).parent / "resources" / "june_tpu_67b_a2b_step_42150_representative_eval_golden.json"
)
PROMPT_FIXTURE_SHA256 = "47863868cbfe336739c8097535f113f4d2dae4954f772eb91511c911433596e8"
PROMPT_FIXTURE_URL = (
    "https://storage.googleapis.com/marin-public/test-data/vllm/e2e/representative-eval-prompts/"
    f"{PROMPT_FIXTURE_SHA256}.json"
)
JAX_COMPILATION_CACHE_DIR = (
    "s3://marin-us-east-02a/tmp/ttl=30d/compilation-cache/june-tpu-67b-a2b-step-42150-sonic-fa4-representative-v2"
)

pytestmark = [pytest.mark.cluster, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


@dataclasses.dataclass(frozen=True)
class PromptCase:
    id: str
    prompt_token_ids: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class PromptFixture:
    tokenizer: str
    tokenizer_revision: str
    cases: tuple[PromptCase, ...]


@dataclasses.dataclass(frozen=True)
class PromptBatch:
    max_tokens: int
    cases: tuple[PromptCase, ...]


@dataclasses.dataclass(frozen=True)
class TokenScore:
    logprob: float
    token_id: int


def prompt_batches(cases: tuple[PromptCase, ...]) -> tuple[PromptBatch, ...]:
    batches = []
    remaining_cases = cases
    for bucket_max_tokens in PROMPT_BUCKET_MAX_TOKENS:
        bucket = tuple(
            sorted(
                (case for case in remaining_cases if len(case.prompt_token_ids) <= bucket_max_tokens),
                key=lambda case: case.id,
            )
        )
        remaining_cases = tuple(case for case in remaining_cases if len(case.prompt_token_ids) > bucket_max_tokens)
        batches.extend(
            PromptBatch(max_tokens=bucket_max_tokens, cases=bucket[start : start + BATCH_SIZE])
            for start in range(0, len(bucket), BATCH_SIZE)
        )
    return tuple(batches)


def read_golden() -> dict[str, tuple[TokenScore, ...]]:
    raw = json.loads(INFERENCE_GOLDEN_PATH.read_text())
    return {
        case["id"]: tuple(
            TokenScore(logprob=entry["logprob"], token_id=entry["token_id"]) for entry in case["top_logprobs"]
        )
        for case in raw["cases"]
    }


def read_prompt_fixture(expected_cases: dict[str, tuple[TokenScore, ...]]) -> PromptFixture:
    fixture_bytes = StoragePath(PROMPT_FIXTURE_URL).read_bytes()
    if hashlib.sha256(fixture_bytes).hexdigest() != PROMPT_FIXTURE_SHA256:
        raise ValueError("Prompt fixture SHA-256 mismatch")
    raw = json.loads(fixture_bytes)
    cases = tuple(
        PromptCase(
            id=case["id"],
            prompt_token_ids=tuple(case["prompt_token_ids"]),
        )
        for case in raw["cases"]
    )
    if {case.id for case in cases} != expected_cases.keys():
        raise ValueError("Prompt and golden case IDs differ")
    return PromptFixture(
        tokenizer=raw["tokenizer"],
        tokenizer_revision=raw["tokenizer_revision"],
        cases=cases,
    )


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


def compute_checkpoint_inference(
    prompt_fixture: PromptFixture,
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
    tokenizer = load_tokenizer(
        snapshot_download(
            prompt_fixture.tokenizer,
            revision=prompt_fixture.tokenizer_revision,
            allow_patterns=["tokenizer*", "special_tokens*", "added_tokens*", "chat_template*"],
        )
    )
    assert tokenizer.eos_token_id is not None

    mesh = compact_grug_mesh()
    assert mesh.shape.get("expert", 1) == 1
    with set_mesh(mesh):
        params, pending_qb_betas = load_checkpoint(inference_model_config, mesh)

        computed_cases = {}
        for batch in prompt_batches(prompt_fixture.cases):
            token_ids = np.full((BATCH_SIZE, batch.max_tokens), tokenizer.eos_token_id, dtype=np.int32)
            last_token_indices = np.empty(BATCH_SIZE, dtype=np.int32)
            for row, case in enumerate(batch.cases):
                token_ids[row, : len(case.prompt_token_ids)] = case.prompt_token_ids
                last_token_indices[row] = len(case.prompt_token_ids) - 1
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


def assert_checkpoint_inference_matches_golden(
    expected_cases: dict[str, tuple[TokenScore, ...]],
) -> None:
    prompt_fixture = read_prompt_fixture(expected_cases)
    actual_cases = compute_checkpoint_inference(prompt_fixture)
    for case_id, expected in expected_cases.items():
        assert actual_cases[case_id] == expected, case_id


def test_h100_node_matches_levanter_inference_golden(marin_gpu_client: IrisClient, run_test_job) -> None:
    expected_cases = read_golden()
    run_test_job(
        marin_gpu_client,
        JobRequest(
            name=f"june-67b-checkpoint-inference-{uuid.uuid4().hex[:8]}",
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
