# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remote Iris jobs for comparing Snowball serving backends with goldens.

The 64 content-addressed prompts span short through 32K-context workloads and
are evaluated in the same production-shaped buckets as the checkpoint test.
Levanter projects only each prompt's last hidden state. vLLM receives the exact
token IDs through its OpenAI completions endpoint, with one concurrent request
pinned to each data-parallel rank. The PP=2 vLLM case is a two-task Iris gang:
task 0 owns pipeline stage 0 and task 1 owns pipeline stage 1.
"""

import dataclasses
import logging
import math
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import requests
from haliax import Axis
from jax.sharding import PartitionSpec as P
from levanter.models.snowball import SnowballLMHeadModel
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration, ExponentialBackoff

from marin.inference.backend import ModelSpec
from marin.inference.config import LevanterEngineConfig, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.iris_tpu_vllm import tpu_ray_gang
from marin.inference.iris_vllm import (
    iris_vllm_followers,
    iris_vllm_launch,
    notify_iris_vllm_stopped,
    wait_for_iris_vllm_shutdown,
)
from marin.inference.levanter_backend import LevanterBackend
from marin.inference.model_preparation import read_attention_heads, select_tensor_parallel_size
from marin.inference.vllm_backend import VllmBackend, vllm_launcher
from marin.inference.vllm_server import IsolatedTpuVllm, VllmLauncherWithEnvironment
from marin.testing.inference.backend_parity import (
    NextTokenObservation,
    NextTokenParity,
    assert_same_rank_repeatability,
    cross_rank_diagnostic,
    parity_from_logprob_row,
)
from marin.testing.inference.snowball import (
    BATCH_SIZE,
    MAX_PROBABILITY_ERROR,
    SNOWBALL,
    TPU_CHIPS_PER_HOST,
    TPU_TENSOR_PARALLEL_SIZE,
    RepresentativeCase,
    RepresentativeGolden,
    RepresentativePromptFixture,
    pad_prompt_batch,
    read_prompt_fixture,
    read_tpu_export_manifest,
)

logger = logging.getLogger(__name__)

RUNTIME_TIMEOUT = 30 * 60
HTTP_CONNECT_TIMEOUT = 30.0
HTTP_READ_TIMEOUT = 5 * 60.0
GPU_COUNT = 8
MAX_MODEL_LEN = 32768
MAX_NUM_BATCHED_TOKENS = 512
RETURNED_LOGPROBS = 50
MOE_IMPLEMENTATION = "sonic"
ATTENTION_IMPLEMENTATION = "gpu_fa4_cute"
# Quadratic reference attention fits these production-shaped batches on eight H100s. Use it as the
# independent short-context oracle; larger batches retain FA4 because reference attention cannot fit 32K.
REFERENCE_ATTENTION_MAX_TOKENS = 4096
GRUG_NUM_LAYERS = 26
# Keep the Levanter gate at the shared 0.075 bound. vLLM's launch-scoped BF16
# expert-reduction variation reached 0.1145 while preserving every greedy token
# and exact same-rank recomputation, so its representative-only bound is scoped
# to that measured serving behavior.
VLLM_MAX_PROBABILITY_ERROR = 0.125


def _with_attention_implementation(model: SnowballLMHeadModel, implementation: str) -> SnowballLMHeadModel:
    """Reuse loaded arrays with a different static Snowball attention implementation."""
    config = dataclasses.replace(model.config, attention_implementation=implementation)
    blocks = tuple(
        dataclasses.replace(
            block,
            attn=dataclasses.replace(block.attn, cfg=config),
            mlp=dataclasses.replace(block.mlp, cfg=config),
        )
        for block in model.transformer.blocks
    )
    transformer = dataclasses.replace(model.transformer, blocks=blocks, config=config)
    return dataclasses.replace(model, transformer=transformer, _config=config)


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


_RUNTIME_TOPOLOGY = re.compile(
    r"GrugMoE effective config: TP=(\d+)/(\d+) PP=(\d+)/(\d+) " r"layers=\[(\d+),(\d+)\) DP=(\d+)/(\d+) EP=(\d+)/(\d+)"
)


def _wait_for_exact_vllm_topology(
    logs: Callable[[], str],
    *,
    check_alive: Callable[[], None],
    task_index: int,
    pipeline_parallel_size: int,
) -> None:
    """Assert the worker, GPU, and model-stage facts emitted by this task."""
    start_layer = task_index * GRUG_NUM_LAYERS // pipeline_parallel_size
    end_layer = (task_index + 1) * GRUG_NUM_LAYERS // pipeline_parallel_size
    expected_placements = Counter(
        {
            (
                "Worker placement: "
                f"process_rank={dp_rank * pipeline_parallel_size + task_index} "
                f"node_rank={task_index} "
                f"local_rank={dp_rank if pipeline_parallel_size == 1 else 0} "
                f"DP={dp_rank}/{GPU_COUNT} EP={dp_rank}/{GPU_COUNT} "
                f"PP={task_index}/{pipeline_parallel_size} TP=0/1 GPU={dp_rank}"
            ): 1
            for dp_rank in range(GPU_COUNT)
        }
    )
    expected_runtime: Counter[tuple[int, ...]] = Counter(
        (0, 1, task_index, pipeline_parallel_size, start_layer, end_layer, dp_rank, GPU_COUNT, dp_rank, GPU_COUNT)
        for dp_rank in range(GPU_COUNT)
    )

    def topology_ready() -> bool:
        check_alive()
        text = logs()
        placements = Counter(
            line[line.index("Worker placement:") :].strip() for line in text.splitlines() if "Worker placement:" in line
        )
        runtime = Counter(tuple(int(value) for value in match.groups()) for match in _RUNTIME_TOPOLOGY.finditer(text))
        unexpected_placements = placements - expected_placements
        unexpected_runtime = runtime - expected_runtime
        if unexpected_placements or unexpected_runtime:
            raise AssertionError(
                f"Unexpected vLLM topology on task {task_index}: "
                f"placements={unexpected_placements}, runtime={unexpected_runtime}"
            )
        return placements == expected_placements and runtime == expected_runtime

    ExponentialBackoff(initial=1.0, maximum=30.0).wait_until_or_raise(
        topology_ready,
        timeout=Duration.from_seconds(RUNTIME_TIMEOUT),
        error_message=f"Timed out waiting for exact vLLM topology on task {task_index}",
    )
    logger.info(
        "Verified task %d topology: placements=%s runtime=%s",
        task_index,
        tuple(expected_placements),
        tuple(expected_runtime),
    )


def score_levanter_against_goldens(goldens: tuple[RepresentativeGolden, ...]) -> None:
    """Load the June export once and score every prompt through Levanter."""
    prompt_fixture = read_prompt_fixture(goldens)
    num_chips = jax.device_count()
    assert num_chips == GPU_COUNT, f"expected {GPU_COUNT} H100s, found {num_chips} devices"
    num_heads, num_kv_heads = read_attention_heads(SNOWBALL.export_uri)
    tensor_parallel_size = select_tensor_parallel_size(num_heads, num_chips, num_kv_heads)
    assert tensor_parallel_size == 1, (num_heads, num_kv_heads, num_chips)

    spec = ModelSpec(
        weights=SNOWBALL.export_uri,
        api_model=SNOWBALL.model_name,
        num_chips=num_chips,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        chat_template_content=None,
    )
    config_overrides = {
        "moe_implementation": MOE_IMPLEMENTATION,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
    }
    with LevanterBackend(LevanterEngineConfig()).load_model(spec, config_overrides=config_overrides) as loaded:
        model = loaded.model
        assert isinstance(model, SnowballLMHeadModel)
        assert model.transformer.token_embed.dtype == jnp.bfloat16
        assert loaded.trainer.data_axis_size == BATCH_SIZE
        assert loaded.tokenizer.eos_token_id is not None
        reference_model = _with_attention_implementation(model, "reference")
        logger.info("Levanter compilation cache: %s", jax.config.jax_compilation_cache_dir)

        @hax.named_jit(axis_resources=loaded.trainer.compute_axis_mapping)
        def next_token_logprobs(m: SnowballLMHeadModel, ids, last_positions) -> jax.Array:
            hidden = m.activations(ids).rearrange(("batch", "position", "embed")).array
            last_hidden = hidden.at[jnp.arange(hidden.shape[0]), last_positions.array].get(out_sharding=P("data"))
            logits = jnp.einsum(
                "bh,hv->bv",
                last_hidden,
                m.transformer.output_proj,
                out_sharding=P("data"),
            )
            assert logits.dtype == jnp.bfloat16
            return jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)

        parities: list[NextTokenParity] = []
        Batch = Axis("batch", BATCH_SIZE)
        for batch_index, batch in enumerate(prompt_fixture.batches):
            assert len(batch.cases) == BATCH_SIZE
            if batch.max_tokens <= REFERENCE_ATTENTION_MAX_TOKENS:
                scoring_model = reference_model
                attention_implementation = "reference"
            else:
                scoring_model = model
                attention_implementation = ATTENTION_IMPLEMENTATION
            logger.info(
                "Levanter batch %d/%d: max_tokens=%d attention=%s cases=%s",
                batch_index + 1,
                len(prompt_fixture.batches),
                batch.max_tokens,
                attention_implementation,
                [case.id for case in batch.cases],
            )
            token_ids, last_token_indices = pad_prompt_batch(batch, loaded.tokenizer.eos_token_id)

            Pos = Axis("position", batch.max_tokens)
            input_ids = hax.named(jnp.asarray(token_ids), (Batch, Pos))
            last_positions = hax.named(jnp.asarray(last_token_indices), (Batch,))
            logprobs = np.asarray(jax.device_get(next_token_logprobs(scoring_model, input_ids, last_positions)))
            for row, case in enumerate(batch.cases):
                parities.append(
                    parity_from_logprob_row(
                        case.id,
                        case.top_logprobs,
                        logprobs[row],
                        backend_rank=row,
                    )
                )

    assert len(parities) == len(prompt_fixture.cases)
    _log_parities("levanter-gpu", parities)
    for parity in parities:
        parity.assert_matches(max_probability_error=MAX_PROBABILITY_ERROR)


def _request_vllm_case(
    completions_url: str,
    model_id: str,
    case: RepresentativeCase,
    rank: int,
    request_id: str,
) -> NextTokenObservation:
    context = f"case={case.id} rank={rank} request={request_id}"
    assert len(case.prompt_token_ids) + 1 <= MAX_MODEL_LEN, context
    try:
        response = requests.post(
            completions_url,
            headers={
                "X-data-parallel-rank": str(rank),
                "X-Request-Id": request_id,
            },
            json={
                "model": model_id,
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
            int(token.removeprefix("token_id:")): float(logprob) for token, logprob in returned_top_logprobs.items()
        }
        assert all(math.isfinite(value) for value in actual_logprobs.values()), context
        return NextTokenObservation.from_logprob_map(
            case.id,
            int(greedy_token_id),
            actual_logprobs,
            backend_rank=rank,
        )
    except Exception as error:
        error.add_note(context)
        raise


def score_vllm_against_goldens(
    goldens: tuple[RepresentativeGolden, ...],
    attention_backend: str,
    pipeline_parallel_size: int,
) -> list[NextTokenParity] | None:
    """Run one task in the vLLM gang; the leader scores rank-pinned prompts."""
    configure_coreweave_s3()
    prompt_fixture = read_prompt_fixture(goldens)

    spec = ModelSpec(
        weights=SNOWBALL.export_uri,
        api_model=SNOWBALL.model_name,
        num_chips=GPU_COUNT * pipeline_parallel_size,
        # The Iris topology owns all parallel dimensions, including TP=1.
        tensor_parallel_size=None,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        chat_template_content=None,
    )
    backend = VllmBackend(
        VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            source=VllmSource.MARIN_FORK,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=1,
            extra_args=(
                "--enable-expert-parallel",
                "--model-loader-extra-config",
                '{"distributed":true}',
                "--max-logprobs",
                str(RETURNED_LOGPROBS),
                "--attention-backend",
                attention_backend,
                # Same-rank repeatability compares fresh recomputes. A cache miss
                # followed by a prefix-cache hit exercises two different paths.
                "--no-enable-prefix-caching",
            ),
        ),
    )
    parities: list[NextTokenParity] = []
    with iris_vllm_launch(
        pipeline_parallel_size=pipeline_parallel_size,
        tensor_parallel_size=1,
        data_parallel_size=GPU_COUNT,
    ) as launch:
        logger.info(
            "vLLM requested topology: tasks=%d GPUs/task=%d DP=%d EP=%d PP=%d TP=%d task=%d",
            pipeline_parallel_size,
            GPU_COUNT,
            GPU_COUNT,
            GPU_COUNT,
            pipeline_parallel_size,
            1,
            launch.task_index,
        )
        with backend.start(
            spec,
            extra_args=launch.extra_cli_args,
            subprocess_env=launch.subprocess_env,
        ) as environment:
            if launch.is_leader:
                environment.wait_until_ready()
            _wait_for_exact_vllm_topology(
                environment.logs,
                check_alive=environment.check_alive,
                task_index=launch.task_index,
                pipeline_parallel_size=pipeline_parallel_size,
            )
            if not launch.is_leader:
                wait_for_iris_vllm_shutdown(launch, environment.check_alive)
                environment.publish_compilation_cache()
            else:
                with iris_vllm_followers(launch):
                    parities = score_vllm_endpoint(
                        prompt_fixture,
                        f"{environment.server_url}/completions",
                        spec.api_model,
                        data_parallel_size=GPU_COUNT,
                    )
        if not launch.is_leader:
            notify_iris_vllm_stopped(launch)
            return None
    return parities


def score_vllm_endpoint(
    fixture: RepresentativePromptFixture,
    completions_url: str,
    model_id: str,
    *,
    data_parallel_size: int,
) -> list[NextTokenParity]:
    """Score exact fixture IDs and require fresh same-rank recomputation.

    GPU waves pin one request to each data-parallel rank. TPU waves use rank 0
    for every request; repeatability is checked serially on that single rank.
    """
    parities = []
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        for wave, batch in enumerate(fixture.batches):
            futures = [
                executor.submit(
                    _request_vllm_case,
                    completions_url,
                    model_id,
                    case,
                    index % data_parallel_size,
                    f"wave-{wave}-{case.id}",
                )
                for index, case in enumerate(batch.cases)
            ]
            for case, future in zip(batch.cases, futures, strict=True):
                observation = future.result()
                parities.append(observation.parity_against(case.top_logprobs))

        sentinel = next(case for case in fixture.cases if case.id == "knowledge-longbench-02")
        assert len(sentinel.prompt_token_ids) > 2048
        repeats = []
        for repeat in range(2):
            futures = [
                executor.submit(
                    _request_vllm_case,
                    completions_url,
                    model_id,
                    sentinel,
                    rank,
                    f"same-rank-repeat-{repeat}-{sentinel.id}-rank-{rank}",
                )
                for rank in range(data_parallel_size)
            ]
            repeats.append([future.result() for future in futures])
        assert_same_rank_repeatability(*repeats)
        logger.info("vLLM cross-rank diagnostic only: %s", cross_rank_diagnostic(repeats[0]))
    assert len(parities) == len(fixture.cases)
    _log_parities("vllm", parities)
    for parity in parities:
        parity.assert_matches(max_probability_error=VLLM_MAX_PROBABILITY_ERROR)
    return parities


def tpu_vllm_engine_config() -> VllmEngineConfig:
    """Return the TPU arm configuration, including its integer vLLM timeout."""
    return VllmEngineConfig(
        launcher=VllmLauncherType.TPU,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=BATCH_SIZE,
        startup_timeout_seconds=RUNTIME_TIMEOUT,
        extra_args=("--max-logprobs", str(RETURNED_LOGPROBS), "--no-enable-prefix-caching"),
    )


def score_tpu_vllm_against_goldens(
    goldens: tuple[RepresentativeGolden, ...],
    weights: str,
    cpus_per_host: float,
) -> list[NextTokenParity] | None:
    """Run the same vLLM contract on one four-host v4 slice at TP16."""
    assert weights == SNOWBALL.tpu_export_uri
    read_tpu_export_manifest()
    fixture = read_prompt_fixture(goldens)
    engine = tpu_vllm_engine_config()
    launcher = vllm_launcher(engine)
    assert isinstance(launcher, IsolatedTpuVllm)
    spec = ModelSpec(
        weights=weights,
        api_model=SNOWBALL.model_name,
        num_chips=TPU_TENSOR_PARALLEL_SIZE,
        tensor_parallel_size=TPU_TENSOR_PARALLEL_SIZE,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        chat_template_content=None,
    )
    # Only isolated Ray/serving children use TPU JAX; the coordinator stays CPU.
    # Torchax also needs the CPU backend during checkpoint loading.
    with tpu_ray_gang(
        VllmLauncherWithEnvironment(launcher, {"JAX_PLATFORMS": "tpu,cpu"}),
        cpus_per_host=max(1, math.floor(cpus_per_host)),
        chips_per_host=TPU_CHIPS_PER_HOST,
        timeout_seconds=RUNTIME_TIMEOUT,
    ) as gang:
        if gang.leader:
            with iris_vllm_followers(gang.identity):
                with VllmBackend(engine).start(
                    spec,
                    extra_args=("--distributed-executor-backend", "ray"),
                    subprocess_env={**gang.subprocess_env, "JAX_PLATFORMS": "tpu,cpu"},
                ) as environment:
                    environment.wait_until_ready()
                    parities = score_vllm_endpoint(
                        fixture,
                        environment.server_url + "/completions",
                        spec.api_model,
                        data_parallel_size=1,
                    )
                    gang.check_alive()
                    environment.check_alive()
        else:
            wait_for_iris_vllm_shutdown(gang.identity, gang.check_alive)
    if not gang.leader:
        notify_iris_vllm_stopped(gang.identity)
        return None
    return parities
