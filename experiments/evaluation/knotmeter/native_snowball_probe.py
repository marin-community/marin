# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay the pinned 64K AgentX diagnostic against native Snowball on one accelerator host."""

import argparse
import dataclasses
import json
import logging
import os
import random
import tempfile
import time
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import requests
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.models.snowball import (
    SnowballConfig,
    SnowballLMHeadModel,
    snowball_block_from_state_dict,
    snowball_embeddings_from_state_dict,
    snowball_final_from_state_dict,
)
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.inference.dashboard_server import bind_serving_socket, serve_app_background

from experiments.evaluation.knotmeter.artifacts import PROFILE_EXPORT_RELATIVE_PATH, read_record
from experiments.evaluation.knotmeter.prefill_probe import SUFFIX_LENGTH, measure_prefill
from experiments.evaluation.knotmeter.processes import run_logged
from experiments.evaluation.knotmeter.recipe import benchmark_command, load_series, workload_environment
from experiments.evaluation.knotmeter.workload import check_native_cache, restore_workload

logger = logging.getLogger(__name__)
SERIES = Path(__file__).with_name("snowball-v01-tpu-64k.yaml")
PROFILE_PROMPT_LENGTHS = (512, 51_870)
PROFILE_PREFILL_LENGTH = 51_840


def _request(prompt_tokens: list[int], max_output_tokens: int) -> Request:
    return Request(
        prompt_tokens=prompt_tokens,
        request_id=0,
        n_generations=1,
        decode_params=dataclasses.replace(
            SeqDecodingParams.default(),
            max_num_tokens=jnp.asarray(len(prompt_tokens) + max_output_tokens, dtype=jnp.int32),
        ),
    )


def _profile_decode(engine: InferenceEngine, prompt_tokens: list[int], output: Path) -> None:
    """Capture one warmed decode round for a synthetic long prefix."""
    request = _request(prompt_tokens, 2 * engine.config.max_rounds + 1)
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 1
    options.python_tracer_level = 0
    options.enable_hlo_proto = True
    tracing = False

    def trace_second_round(iteration: int) -> None:
        nonlocal tracing
        if iteration == 1:
            jax.profiler.start_trace(output, profiler_options=options)
            tracing = True

    try:
        result = engine.generate([request], step_callback=trace_second_round)
    finally:
        if tracing:
            jax.profiler.stop_trace()
    if not tracing:
        raise RuntimeError("Synthetic profile request ended before the second decode round")
    logger.info("Profiled decode with %d prompt tokens and %d output tokens", len(prompt_tokens), result.total_generated)


def _profile_prefill(engine: InferenceEngine, output: Path) -> None:
    """Capture one warmed 64K-bucket prefill with and without a cached prefix."""
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 1
    options.python_tracer_level = 0
    options.enable_hlo_proto = False
    generator = random.Random(170)
    for mode in ("fresh", "primed"):
        prompt = [generator.randrange(256, 10_000) for _ in range(PROFILE_PREFILL_LENGTH)]
        if mode == "primed":
            engine.generate([_request(prompt[:-SUFFIX_LENGTH], 1)])
        with jax.profiler.trace(str(output / f"prefill-profile-{mode}"), profiler_options=options):
            engine.generate([_request(prompt, 1)])


def _convert_layers(
    template: SnowballLMHeadModel, state_dict: dict[str, jax.Array], prefix: str | None
) -> SnowballLMHeadModel:
    """Convert a resident HF state dict one layer at a time, releasing the inputs."""
    layer_prefix = f"{prefix}.model.layers." if prefix else "model.layers."
    weights_by_layer: list[dict[str, jax.Array]] = [{} for _ in template.transformer.blocks]
    for name in tuple(state_dict):
        if name.startswith(layer_prefix):
            index, _, suffix = name.removeprefix(layer_prefix).partition(".")
            weights_by_layer[int(index)][f"{layer_prefix}0.{suffix}"] = state_dict.pop(name)

    transformer = eqx.filter_jit(snowball_embeddings_from_state_dict)(template.transformer, state_dict, prefix)
    transformer = eqx.filter_jit(snowball_final_from_state_dict)(transformer, state_dict, prefix)
    jax.block_until_ready(transformer)
    state_dict.clear()

    convert_block = eqx.filter_jit(snowball_block_from_state_dict)
    blocks = []
    for block, weights in zip(transformer.blocks, weights_by_layer, strict=True):
        blocks.append(jax.block_until_ready(convert_block(block, weights, 0, prefix)))
        weights.clear()
    transformer = eqx.tree_at(lambda value: value.blocks, transformer, tuple(blocks))
    return eqx.tree_at(lambda model: model.transformer, template, transformer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    benchmark = parser.add_mutually_exclusive_group(required=True)
    benchmark.add_argument("--workload")
    benchmark.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--measure-prefill", action="store_true", help="Also measure prefill after AgentX")
    parser.add_argument("--accelerator", choices=("tpu", "h100x8"), default="tpu")
    parser.add_argument("--chips", type=int, choices=(4, 8))
    parser.add_argument("--layerwise-load", action="store_true", help="Bound checkpoint conversion HBM by one layer")
    parser.add_argument("--attention", choices=("tpu", "tpu_fp32", "tpu_v3", "gpu", "reference"), default="tpu_fp32")
    parser.add_argument("--decode-layers", choices=("scan", "unrolled", "unrolled_split_cache"), default="scan")
    parser.add_argument("--decode-experts", choices=("grouped", "split"), default="grouped")
    parser.add_argument("--decode-token-layout", choices=("packed", "replicated"), default="packed")
    parser.add_argument("--kv-sharding", choices=("model", "expert"), default="model")
    parser.add_argument("--kv-append", choices=("separate", "fused"), default="separate")
    parser.add_argument("--kv-cache-contexts", type=int, default=1)
    parser.add_argument("--prefill-buckets", type=int, nargs="+", default=())
    parser.add_argument("--max-unrolled-prefill-tokens", type=int)
    parser.add_argument("--prefix-cache", choices=("disabled", "last_prompt", "recent_prompts"), default="disabled")
    parser.add_argument("--profile-decode", action="store_true", help="Trace short and long decode after the AgentX run")
    parser.add_argument(
        "--profile-prefill", action="store_true", help="Trace fresh and primed prefill after measurement"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    series = load_series(SERIES)
    assert (series["max_context"], series["duration"]) == (65536, 180)
    devices = jax.devices()
    if args.accelerator == "h100x8":
        expert_shards, platform, device_kind = 8, "gpu", "H100"
        if args.chips not in (None, expert_shards):
            parser.error("--accelerator h100x8 requires eight devices")
    else:
        expert_shards, platform, device_kind = args.chips or 4, "tpu", "TPU"
    if len(devices) != expert_shards or any(
        device.platform != platform or device_kind not in device.device_kind for device in devices
    ):
        raise RuntimeError(f"Expected {expert_shards} {device_kind} devices, got {devices}")

    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    workload = read_record(args.workload) if args.workload else None
    trainer = TrainerConfig(
        mp=jmp.get_policy("p=bf16,c=bf16"),
        mesh=MeshConfig(axes={"replica": 1, "data": 1, "model": 1, "expert": expert_shards, "context": 1}),
        use_explicit_mesh_axes=True,
    )
    service = InferenceEngineConfig(
        max_seq_len=series["max_context"],
        max_seqs=1,
        max_seqs_in_prefill=1,
        max_queued_tokens=32,
        max_tokens_per_round=expert_shards,  # Pad the one active decode token to the expert mesh.
        prefill_bucket_sizes=tuple(args.prefill_buckets),
        prefix_cache=args.prefix_cache,
    )
    service = dataclasses.replace(service, max_pages=service.max_pages_per_seq * args.kv_cache_contexts)
    config = InferenceServerConfig(trainer=trainer, model_name=series["model"], service=service)

    with tempfile.TemporaryDirectory(prefix="agentx-native-") as workdir:
        bundle = Path(workdir) / "bundle"
        if workload is not None:
            restore_workload(workload, bundle)
            check_native_cache(bundle, SERIES, output / "workload-check.log")

        started = time.monotonic()
        converter = HFCheckpointConverter.from_hf(args.weights)
        model_config = converter.default_config
        if not isinstance(model_config, SnowballConfig):
            raise TypeError(f"Expected a Snowball checkpoint, got {type(model_config).__name__}")
        model_config = dataclasses.replace(
            model_config,
            inference_attention_implementation=args.attention,
            inference_decode_implementation=args.decode_layers,
            inference_expert_implementation=args.decode_experts,
            inference_decode_token_layout=args.decode_token_layout,
            inference_kv_sharding=args.kv_sharding,
            inference_kv_append=args.kv_append,
        )
        if args.max_unrolled_prefill_tokens is not None:
            model_config = dataclasses.replace(
                model_config, inference_max_unrolled_prefill_tokens=args.max_unrolled_prefill_tokens
            )
        tokenizer = load_tokenizer(series["tokenizer"], revision=series["tokenizer_revision"])
        with trainer.use_device_mesh():
            if args.layerwise_load:
                dtype = trainer.mp.compute_dtype
                state_dict = converter.load_state_dict(args.weights, dtype)
                prefix = converter.ignore_prefix
                if prefix and not any(name.startswith(f"{prefix}.") for name in state_dict):
                    prefix = None
                vocab = converter.Vocab.resize(model_config.vocab_size)
                template = eqx.filter_eval_shape(
                    SnowballLMHeadModel.init, vocab, model_config, key=jax.random.PRNGKey(0)
                )
                with hax.axis_mapping(trainer.parameter_axis_mapping):
                    model = _convert_layers(template, state_dict, prefix)
            else:
                model = converter.load_pretrained(
                    model_config.model_type,
                    ref=args.weights,
                    config=model_config,
                    dtype=trainer.mp.compute_dtype,
                    axis_mapping=trainer.parameter_axis_mapping,
                )
            jax.block_until_ready(model)
            load_seconds = time.monotonic() - started
            logger.info("Loaded Snowball in %.1f seconds", load_seconds)
            server = InferenceServer.create(config, model=model, tokenizer=tokenizer)

        sock = bind_serving_socket("127.0.0.1", 0)
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        command = benchmark_command(series, 1, url, output / "aiperf_artifacts")
        metrics_arg = command.index("--server-metrics")
        del command[metrics_arg : metrics_arg + 2]
        evidence = {
            "backend": "levanter",
            "series": series,
            "weights": args.weights,
            "workload_uri": args.workload,
            "workload_sha256": workload["archive_sha256"] if workload is not None else None,
            "devices": [device.device_kind for device in devices],
            "jax_version": jax.__version__,
            "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS"),
            "mesh": {"model": 1, "expert": expert_shards},
            "max_pages": service.max_pages,
            "max_tokens_per_round": service.max_tokens_per_round,
            "attention": args.attention,
            "decode_layers": args.decode_layers,
            "decode_experts": args.decode_experts,
            "decode_token_layout": args.decode_token_layout,
            "kv_sharding": args.kv_sharding,
            "kv_append": args.kv_append,
            "prefill_buckets": service.prefill_bucket_sizes,
            "max_unrolled_prefill_tokens": model_config.inference_max_unrolled_prefill_tokens,
            "prefix_cache": service.prefix_cache,
            "load_seconds": load_seconds,
            "layerwise_load": args.layerwise_load,
            "command": command,
        }
        try:
            with serve_app_background(server.app, sock, name="snowball-native-agentx"):
                probe_path = output / "probe.json"
                probe_path.write_text(json.dumps(evidence, indent=2))
                started = time.monotonic()
                response = requests.post(
                    f"{url}/v1/chat/completions",
                    json={
                        "model": series["model"],
                        "messages": [{"role": "user", "content": "Reply with one short greeting."}],
                        "max_tokens": 2,
                        "temperature": 0,
                    },
                    timeout=1800,
                )
                if not response.ok:
                    raise RuntimeError(f"Native preflight failed with HTTP {response.status_code}: {response.text}")
                if response.json()["usage"]["completion_tokens"] < 1:
                    raise RuntimeError("Native preflight generated no tokens")
                evidence["cold_completion_seconds"] = time.monotonic() - started
                if service.prefill_bucket_sizes:
                    context = server.inference_context
                    filler = tokenizer.encode(" example", add_special_tokens=False)[0]
                    with (
                        context.model_lock,
                        trainer.use_device_mesh(),
                        hax.axis_mapping(trainer.compute_axis_mapping),
                    ):
                        for index, previous_bucket in enumerate(service.prefill_bucket_sizes):
                            prompt = [filler] * (previous_bucket + 1)
                            # Distinct prefixes force each prefill buffer to compile.
                            prompt[0] = index
                            context.engine.generate([_request(prompt, 1)])
                        context.engine.reset()
                    evidence["bucket_warmup_seconds"] = time.monotonic() - started - evidence["cold_completion_seconds"]
                probe_path.write_text(json.dumps(evidence, indent=2))
                try:
                    if workload is not None:
                        run_logged(
                            command,
                            workload_environment(bundle, series),
                            output / "agentx.log",
                            timeout=series["warmup_grace_period"] + series["duration"] + 600,
                            sample=output / "client-load.jsonl",
                        )
                        profile = json.loads((output / PROFILE_EXPORT_RELATIVE_PATH).read_text())
                        if profile.get("was_cancelled") or profile.get("request_count", {}).get("avg", 0) == 0:
                            raise RuntimeError("AgentX did not complete any measured requests")
                    if args.prefill_only or args.measure_prefill:
                        measure_prefill(f"{url}/v1", series["model"], output / "prefill.json")
                finally:
                    server.pause_generation()
                if args.profile_prefill:
                    context = server.inference_context
                    with (
                        context.model_lock,
                        trainer.use_device_mesh(),
                        hax.axis_mapping(trainer.compute_axis_mapping),
                    ):
                        _profile_prefill(context.engine, output)
                if args.profile_decode:
                    token_ids = tokenizer.encode(
                        "Read this passage and continue it with a detailed technical explanation.\n",
                        add_special_tokens=False,
                    )
                    context = server.inference_context
                    with (
                        context.model_lock,
                        trainer.use_device_mesh(),
                        hax.axis_mapping(trainer.compute_axis_mapping),
                    ):
                        for length in PROFILE_PROMPT_LENGTHS:
                            repeats = (length + len(token_ids) - 1) // len(token_ids)
                            prompt = (token_ids * repeats)[:length]
                            _profile_decode(context.engine, prompt, output / f"decode-profile-{length}")
        finally:
            server.shutdown()


if __name__ == "__main__":
    main()
