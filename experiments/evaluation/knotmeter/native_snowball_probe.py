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
import threading
import time
from pathlib import Path

import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import requests
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.inference.snowball_engine_replay import generate_snowball_replay
from levanter.inference.snowball_loading import load_snowball_serving_model
from levanter.models.snowball import SnowballConfig
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    benchmark = parser.add_mutually_exclusive_group(required=True)
    benchmark.add_argument("--workload")
    benchmark.add_argument("--ready-file", type=Path)
    benchmark.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--measure-prefill", action="store_true", help="Also measure prefill after AgentX")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-seqs-in-prefill", type=int, default=16)
    parser.add_argument("--max-rounds", type=int, default=32)
    parser.add_argument("--max-tokens-per-round", type=int, help="Cap decode tokens independently of client concurrency")
    parser.add_argument("--accelerator", choices=("tpu", "h100x4", "h100x8"), default="tpu")
    parser.add_argument("--chips", type=int, choices=(4, 8))
    parser.add_argument("--layerwise-load", action="store_true", help="Bound checkpoint conversion HBM by one layer")
    parser.add_argument("--pack-expert-weights", action="store_true", help="Pack gate and up weights for inference")
    parser.add_argument("--attention", choices=("tpu", "tpu_fp32", "tpu_v3", "gpu", "reference"), default="tpu_fp32")
    parser.add_argument("--decode-layers", choices=("scan", "unrolled", "unrolled_split_cache"), default="scan")
    parser.add_argument("--decode-experts", choices=("grouped", "separate", "split"), default="grouped")
    parser.add_argument("--decode-token-layout", choices=("packed", "replicated"), default="packed")
    parser.add_argument("--replicate-window-kv", action="store_true")
    parser.add_argument("--kv-sharding", choices=("model", "expert"), default="model")
    parser.add_argument("--kv-append", choices=("separate", "fused"), default="separate")
    parser.add_argument("--kv-cache-contexts", type=int, default=1)
    parser.add_argument("--window-cache-pages", type=int, help="Separate capacity for sliding-window KV")
    parser.add_argument(
        "--window-cache-retained-spans", type=int, default=1, help="Sliding-window spans retained per cached prompt"
    )
    parser.add_argument("--prefill-buckets", type=int, nargs="+", default=())
    parser.add_argument("--decode-buckets", type=int, nargs="+", default=())
    parser.add_argument("--max-unrolled-prefill-tokens", type=int)
    parser.add_argument("--prefix-cache", choices=("disabled", "last_prompt", "recent_prompts"), default="disabled")
    parser.add_argument("--max-retained-prompts", type=int)
    parser.add_argument("--capture-routes", action="store_true", help="Request Snowball routes in AgentX responses")
    parser.add_argument("--return-policy-logprobs", action="store_true", help="Request raw policy logprobs in responses")
    parser.add_argument("--profile-decode", action="store_true", help="Trace short and long decode after the AgentX run")
    parser.add_argument(
        "--profile-prefill", action="store_true", help="Trace fresh and primed prefill after measurement"
    )
    args = parser.parse_args()
    if args.pack_expert_weights and not args.layerwise_load:
        parser.error("--pack-expert-weights requires --layerwise-load")
    if min(args.concurrency, args.max_seqs_in_prefill, args.max_rounds) < 1:
        parser.error("--concurrency, --max-seqs-in-prefill, and --max-rounds must be positive")
    if args.concurrency > 1 and (args.prefix_cache == "last_prompt" or args.decode_token_layout == "replicated"):
        parser.error("last_prompt and replicated decode require --concurrency 1")
    if args.replicate_window_kv and (
        args.decode_token_layout != "replicated"
        or args.kv_sharding != "expert"
        or args.decode_layers != "unrolled_split_cache"
    ):
        raise ValueError("Window KV replication requires replicated tokens and split expert-sharded KV buffers")
    logging.basicConfig(level=logging.INFO)

    series = load_series(SERIES)
    assert (series["max_context"], series["duration"]) == (65536, 180)
    devices = jax.devices()
    if args.accelerator.startswith("h100x"):
        expert_shards, platform, device_kind = int(args.accelerator[-1]), "gpu", "H100"
        if args.chips not in (None, expert_shards):
            parser.error("--chips must match --accelerator")
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
    packed_tokens = ((args.concurrency + expert_shards - 1) // expert_shards) * expert_shards
    decode_tokens = 1 if args.decode_token_layout == "replicated" else packed_tokens
    if args.max_tokens_per_round is not None:
        if args.decode_token_layout == "replicated":
            parser.error("--max-tokens-per-round requires packed decode")
        if not 0 < args.max_tokens_per_round <= packed_tokens or args.max_tokens_per_round % expert_shards:
            parser.error("--max-tokens-per-round must be positive, mesh-aligned, and at most the packed concurrency")
        decode_tokens = args.max_tokens_per_round
    if any(size % expert_shards for size in args.decode_buckets):
        parser.error("--decode-buckets must be divisible by the expert mesh size")
    service = InferenceEngineConfig(
        max_seq_len=series["max_context"],
        max_seqs=args.concurrency,
        max_seqs_in_prefill=min(args.concurrency, args.max_seqs_in_prefill),
        max_queued_tokens=1 if args.decode_token_layout == "replicated" else max(32, packed_tokens),
        max_tokens_per_round=decode_tokens,
        max_rounds=args.max_rounds,
        prefill_bucket_sizes=tuple(args.prefill_buckets),
        decode_bucket_sizes=tuple(args.decode_buckets),
        prefix_cache=args.prefix_cache,
        max_retained_prompts=args.max_retained_prompts,
        window_cache_max_pages=args.window_cache_pages,
        window_cache_retained_spans=args.window_cache_retained_spans,
        return_policy_logprobs=args.return_policy_logprobs,
    )
    service = dataclasses.replace(service, max_pages=service.max_pages_per_seq * args.kv_cache_contexts)
    config = InferenceServerConfig(
        trainer=trainer,
        model_name=series["model"],
        service=service,
        batch_timeout=0.001,
        capture_routes=args.capture_routes,
        initial_policy_id=str(args.weights),
    )
    response_fields = {
        name: enabled
        for name, enabled in (
            ("return_route_captures", args.capture_routes),
            ("return_policy_logprobs", args.return_policy_logprobs),
        )
        if enabled
    }

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
            inference_replicate_window_kv=args.replicate_window_kv,
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
                model = load_snowball_serving_model(
                    converter,
                    args.weights,
                    model_config,
                    dtype=trainer.mp.compute_dtype,
                    axis_mapping=trainer.parameter_axis_mapping,
                    pack_expert_weights=args.pack_expert_weights,
                )
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
        command = benchmark_command(series, args.concurrency, url, output / "aiperf_artifacts")
        metrics_arg = command.index("--server-metrics")
        del command[metrics_arg : metrics_arg + 2]
        if response_fields:
            command.extend(("--extra-inputs", json.dumps(response_fields)))
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
            "concurrency": args.concurrency,
            "max_seqs": service.max_seqs,
            "max_seqs_in_prefill": service.max_seqs_in_prefill,
            "max_pages": service.max_pages,
            "window_cache_max_pages": service.window_cache_max_pages,
            "window_cache_retained_spans": service.window_cache_retained_spans,
            "max_queued_tokens": service.max_queued_tokens,
            "max_tokens_per_round": service.max_tokens_per_round,
            "max_rounds": service.max_rounds,
            "max_retained_prompts": service.max_retained_prompts,
            "attention": args.attention,
            "decode_layers": args.decode_layers,
            "decode_experts": args.decode_experts,
            "decode_token_layout": args.decode_token_layout,
            "replicate_window_kv": args.replicate_window_kv,
            "kv_sharding": args.kv_sharding,
            "kv_append": args.kv_append,
            "prefill_buckets": service.prefill_bucket_sizes,
            "decode_buckets": service.decode_bucket_sizes,
            "max_unrolled_prefill_tokens": model_config.inference_max_unrolled_prefill_tokens,
            "prefix_cache": service.prefix_cache,
            "response_fields": response_fields,
            "load_seconds": load_seconds,
            "layerwise_load": args.layerwise_load,
            "pack_expert_weights": args.pack_expert_weights,
            "batch_timeout": config.batch_timeout,
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
                        **response_fields,
                    },
                    timeout=1800,
                )
                if not response.ok:
                    raise RuntimeError(f"Native preflight failed with HTTP {response.status_code}: {response.text}")
                if response.json()["usage"]["completion_tokens"] < 1:
                    raise RuntimeError("Native preflight generated no tokens")
                evidence["cold_completion_seconds"] = time.monotonic() - started
                if service.prefill_bucket_sizes or service.decode_bucket_sizes:
                    context = server.inference_context
                    filler = tokenizer.encode(" example", add_special_tokens=False)[0]
                    with (
                        context.model_lock,
                        trainer.use_device_mesh(),
                        hax.axis_mapping(trainer.compute_axis_mapping),
                    ):

                        def warm(requests):
                            if args.capture_routes:
                                generate_snowball_replay(context.engine, requests, policy_id=context.policy_id)
                            else:
                                context.engine.generate(requests)

                        for index, previous_bucket in enumerate(service.prefill_bucket_sizes):
                            prompt = [filler] * (previous_bucket + 1)
                            # Distinct prefixes force each prefill buffer to compile.
                            prompt[0] = index
                            warm([_request(prompt, 1)])
                        for previous_bucket in service.decode_bucket_sizes:
                            warm(
                                [
                                    dataclasses.replace(_request([filler], 2), request_id=index)
                                    for index in range(previous_bucket + 1)
                                ]
                            )
                        if args.decode_token_layout == "replicated":
                            # Warm a round that takes the per-layer decode cache as input.
                            warm([_request([filler], 2 * service.max_rounds + 1)])
                        context.engine.reset()
                    evidence["bucket_warmup_seconds"] = time.monotonic() - started - evidence["cold_completion_seconds"]
                probe_path.write_text(json.dumps(evidence, indent=2))
                if args.ready_file is not None:
                    args.ready_file.write_text(url)
                    threading.Event().wait()
                    return
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
                        if args.measure_prefill:
                            # The replay can time out while the server still has live requests.
                            server.pause_generation()
                            server.resume_generation()
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
