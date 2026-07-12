# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end regression tests for the vendored June TPU 67B checkpoint.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.

Run from the repository root:
    uv run pytest tests/integration/iris/test_june_tpu_67b_a2b_checkpoint.py -o addopts= -vv -s
"""

import contextlib
import dataclasses
import json
import logging
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, JobStatus, ResourceConfig, create_environment
from haliax.partitioning import set_mesh
from iris.cli.connect import open_iris_client
from iris.client import IrisClient
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from iris.test_util import wait_for_condition
from levanter.checkpoint import load_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer
from levanter.utils.jax_utils import parameter_count
from rigging.filesystem import StoragePath
from rigging.timing import Duration

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

logger = logging.getLogger(__name__)

CLUSTER_NAME = "cw-us-east-02a"
MARIN_ROOT = Path(__file__).resolve().parents[3]
PENDING_TIMEOUT = 5 * 60.0
RUNTIME_TIMEOUT = 10 * 60.0
TOP_K = 25
RUN_ROOT = (
    "s3://marin-us-east-02a/marin/grug/"
    "moe_67b_a2b_d2560_ep1_rep8_bs8192_seq8192_sw2k_v4_2048_muon_resume15k_v2_10T-9fcc1f"
)
EXECUTOR_INFO_PATH = f"{RUN_ROOT}/.executor_info"
CHECKPOINT_PATH = f"{RUN_ROOT}/checkpoints/step-18000"
MODEL_CONFIG_RESOURCE = Path(__file__).parent / "resources" / "june_tpu_67b_a2b_step_18000_model_config.json"
LOGPROBS_RESOURCE = Path(__file__).parent / "resources" / "june_tpu_67b_a2b_step_18000_logprobs.json"
JAX_COMPILATION_CACHE_DIR = (
    "s3://marin-us-east-02a/tmp/ttl=30d/compilation-cache/june-tpu-67b-a2b-step-18000-sonic-deterministic-v1"
)

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 60)]


@pytest.fixture
def marin_gpu_client() -> Iterator[IrisClient]:
    # This test drives a live CoreWeave GPU node; it can only run where the cluster's
    # kube credentials are present. In CI (and any workstation without them) opening the
    # client raises ConfigException, so skip rather than error the whole integration run.
    # Import kubernetes lazily: it ships with iris[controller] and is absent from the
    # unit-test env, which still collects (imports) this module before deselecting it.
    from kubernetes.config.config_exception import ConfigException  # noqa: PLC0415

    with contextlib.ExitStack() as stack:
        try:
            client = stack.enter_context(open_iris_client(cluster_name=CLUSTER_NAME, workspace=MARIN_ROOT))
        except ConfigException as exc:
            pytest.skip(f"CoreWeave cluster {CLUSTER_NAME!r} unavailable (no kube-config): {exc}")
        yield client


@eqx.filter_jit
def top_k_next_token_logprobs(
    model: Transformer,
    pending_qb_betas: jax.Array,
    token_ids: jax.Array,
    policy: jmp.Policy,
) -> tuple[jax.Array, jax.Array]:
    assert model.stacked_blocks is not None
    # Mirrors train._apply_qb_betas without importing the training entrypoint.
    router_bias = -pending_qb_betas
    router_bias -= jnp.mean(router_bias, axis=-1, keepdims=True)
    model = eqx.tree_at(lambda tree: tree.stacked_blocks.stacked.mlp.router_bias, model, router_bias)
    model = policy.cast_to_compute(model)
    logits = model.logits(token_ids)
    assert logits.dtype == jnp.bfloat16
    logprobs = jax.nn.log_softmax(logits[:, -1].astype(jnp.float32))
    return jax.lax.top_k(logprobs, TOP_K)


def assert_checkpoint_inference(
    executor_info_path: str,
    checkpoint_path: str,
    expected_model_config: dict,
    expected_inference: dict,
) -> None:
    executor_info = json.loads(StoragePath(executor_info_path).read_text())
    model_config = draccus.decode(GrugModelConfig, executor_info["config"]["model"])
    assert dataclasses.asdict(model_config) == expected_model_config
    assert executor_info["config"]["mp"] == expected_inference["mp"]
    assert executor_info["config"]["data"]["tokenizer"] == expected_inference["tokenizer"]
    expected_top = expected_inference["top_logprobs"]
    assert len(expected_top) == TOP_K
    # With expert parallelism disabled, the source default falls back to nondeterministic scatter-add on GPU.
    inference_model_config = dataclasses.replace(
        model_config, moe_implementation=expected_inference["moe_implementation"]
    )

    mesh = compact_grug_mesh()
    with set_mesh(mesh):
        model = eqx.filter_eval_shape(Transformer.init, inference_model_config, key=jax.random.PRNGKey(0))

        checkpoint_started = time.perf_counter()
        checkpoint_state = load_checkpoint(
            {
                "params": model,
                "pending_qb_betas": jax.ShapeDtypeStruct(
                    (inference_model_config.num_layers, inference_model_config.num_experts), jnp.float32
                ),
            },
            checkpoint_path,
            mesh=mesh,
        )
        jax.block_until_ready(checkpoint_state)
        checkpoint_elapsed = time.perf_counter() - checkpoint_started
        gib = 1024**3
        checkpoint_logical_gib = (
            parameter_count(checkpoint_state["params"]) * checkpoint_state["params"].token_embed.dtype.itemsize / gib
        )

        tokenizer = load_tokenizer(expected_inference["tokenizer"])
        prompt_token_ids = tokenizer.encode(expected_inference["prompt"], add_special_tokens=False)
        assert prompt_token_ids == expected_inference["prompt_token_ids"]
        # The batch axis spans all eight GPUs, so inference requires one prompt per device.
        token_ids = jnp.asarray([prompt_token_ids] * jax.device_count(), dtype=jnp.int32)
        policy = jmp.get_policy(expected_inference["mp"])

        inference_started = time.perf_counter()
        top_logprobs, top_token_ids = top_k_next_token_logprobs(
            checkpoint_state["params"], checkpoint_state["pending_qb_betas"], token_ids, policy
        )
        jax.block_until_ready(top_logprobs)
        inference_elapsed = time.perf_counter() - inference_started

    top_token_ids = np.asarray(jax.device_get(top_token_ids))
    top_logprobs = np.asarray(jax.device_get(top_logprobs))

    expected_token_ids = np.asarray([entry["token_id"] for entry in expected_top])
    expected_logprobs = np.asarray([entry["logprob"] for entry in expected_top])
    np.testing.assert_array_equal(top_token_ids, np.broadcast_to(expected_token_ids, top_token_ids.shape))
    np.testing.assert_allclose(top_logprobs, np.broadcast_to(expected_logprobs, top_logprobs.shape), rtol=0, atol=1e-5)
    assert [tokenizer.decode([int(token_id)]) for token_id in top_token_ids[0]] == [
        entry["text"] for entry in expected_top
    ]
    assert tokenizer.decode([int(top_token_ids[0, 0])]).strip() == "America"

    logger.info(
        "Checkpoint inference timing: %s",
        {
            "checkpoint_load_seconds": checkpoint_elapsed,
            "checkpoint_logical_gib": checkpoint_logical_gib,
            "compile_and_inference_seconds": inference_elapsed,
            "logical_gib_per_second": checkpoint_logical_gib / checkpoint_elapsed,
        },
    )


def test_h100_node_runs_checkpoint_inference(marin_gpu_client: IrisClient) -> None:
    expected_model_config = json.loads(MODEL_CONFIG_RESOURCE.read_text())
    expected_inference = json.loads(LOGPROBS_RESOURCE.read_text())
    client = FrayIrisClient.from_iris_client(marin_gpu_client)
    job = client.submit(
        JobRequest(
            name=f"june-67b-checkpoint-inference-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(
                assert_checkpoint_inference,
                args=[EXECUTOR_INFO_PATH, CHECKPOINT_PATH, expected_model_config, expected_inference],
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
            priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        ),
        adopt_existing=False,
    )

    try:
        task_id = JobName.from_string(job.job_id).task(0)
        wait_for_condition(
            lambda: marin_gpu_client.task_status(task_id).state
            not in (
                job_pb2.TASK_STATE_PENDING,
                job_pb2.TASK_STATE_ASSIGNED,
                job_pb2.TASK_STATE_BUILDING,
            ),
            timeout=Duration.from_seconds(PENDING_TIMEOUT),
            poll_interval=5,
        )
        job.wait(timeout=RUNTIME_TIMEOUT, stream_logs=True)
    finally:
        if not JobStatus.finished(job.status()):
            job.terminate()
