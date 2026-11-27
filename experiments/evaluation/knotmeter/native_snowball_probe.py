# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen 64K AgentX replay against native Snowball on one TPU slice."""

import argparse
import dataclasses
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import haliax as hax
import httpx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.inference.binary import (
    EXPERT_IDS,
    EXPERT_VALID,
    GENERATE_PATH,
    MEDIA_TYPE,
    POLICY_LOGPROBS,
    PROMPT_TOKEN_IDS,
    TOKEN_IDS,
    choice_expert_routes,
    choice_tensor_key,
    decode_frame,
    encode_frame,
)
from levanter.inference.engine import Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.layers.sampler import LogprobMode
from levanter.models.snowball import SnowballConfig
from levanter.tokenizers import load_tokenizer
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.inference.config import DEFAULT_SNOWBALL_AGENTX_PROFILE, SNOWBALL_AGENTX_PROFILES
from marin.inference.dashboard_server import bind_serving_socket, serve_app_background
from marin.inference.snowball_agentx import (
    AGENTX_BATCH_TIMEOUT,
    AGENTX_CONTEXT,
    agentx_engine_config,
    load_agentx_model,
    require_agentx_devices,
)

from experiments.evaluation.knotmeter.artifacts import PROFILE_EXPORT_RELATIVE_PATH, read_record
from experiments.evaluation.knotmeter.processes import run_logged
from experiments.evaluation.knotmeter.recipe import SERIES_PATH, benchmark_command, load_series, workload_environment
from experiments.evaluation.knotmeter.workload import check_native_cache, restore_workload

logger = logging.getLogger(__name__)

_PROBE_HTTP_TIMEOUT = 180.0


def _request(tokens: list[int], request_id: int, generated: int) -> Request:
    params = dataclasses.replace(SeqDecodingParams.default(), max_num_tokens=jnp.int32(len(tokens) + generated))
    return Request(tokens, request_id, params, n_generations=1)


def _warmup(server: InferenceServer, capture_routes: bool, filler: int) -> None:
    ctx = server.inference_context
    config = ctx.engine.config
    with ctx.model_lock, ctx.config.trainer.use_device_mesh(), hax.axis_mapping(ctx.config.trainer.compute_axis_mapping):
        ctx.engine.generate([_request([filler], 0, 2)], capture_routes=capture_routes)
        for index, lower in enumerate(config.prefill_bucket_sizes):
            length = lower + 1 if lower < config.prefill_bucket_sizes[-1] else config.max_seq_len - 1
            prompt = [index, *([filler] * (length - 1))]
            result = ctx.engine.generate([_request(prompt, 0, 1)], capture_routes=capture_routes)
            if result.total_generated != 1:
                raise RuntimeError(f"Expected one token after a {length}-token AgentX probe prompt")
            logger.info("Warmed native prefill with %d prompt tokens", length)
        for lower in config.decode_bucket_sizes:
            requests = [_request([filler], i, 2) for i in range(lower + 1)]
            ctx.engine.generate(requests, capture_routes=capture_routes)
        ctx.engine.reset()


def _check_chat_stream(server_url: str, model: str, capture_routes: bool) -> None:
    request = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with one short greeting."}],
        "max_tokens": 2,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_route_captures": capture_routes,
    }
    completed = False
    route_frame = False
    with httpx.stream(
        "POST", f"{server_url}/v1/chat/completions", json=request, timeout=_PROBE_HTTP_TIMEOUT
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                completed = True
                break
            route_frame |= "levanter_route_frame" in json.loads(payload)
    if not completed or (capture_routes and not route_frame):
        raise RuntimeError("Native chat preflight did not complete its requested stream")
    logger.info("Native chat stream completed with route capture %s", capture_routes)


def _check_binary_context_limit(server_url: str, token: int, max_context: int, *, capture_routes: bool) -> None:
    prompt = np.full(max_context - 1, token, dtype=np.int32)
    options = {
        "max_tokens": 1,
        "temperature": 0,
        "expected_logprob_mode": LogprobMode.POLICY,
        "return_route_captures": capture_routes,
    }
    response = httpx.post(
        f"{server_url}{GENERATE_PATH}",
        headers={"content-type": MEDIA_TYPE},
        content=encode_frame(options, {PROMPT_TOKEN_IDS: prompt}),
        timeout=_PROBE_HTTP_TIMEOUT,
    )
    response.raise_for_status()
    _, tensors = decode_frame(response.content)
    output_ids = tensors[choice_tensor_key(0, TOKEN_IDS)]
    scores = tensors[choice_tensor_key(0, POLICY_LOGPROBS)]
    if not np.array_equal(tensors[PROMPT_TOKEN_IDS], prompt) or output_ids.shape != (1,) or output_ids.dtype != np.int32:
        raise RuntimeError("Native binary generation did not round-trip the context-boundary request")
    if scores.shape != output_ids.shape or scores.dtype != np.float32 or not np.isfinite(scores).all():
        raise RuntimeError("Native binary generation did not return a finite policy score for the sampled token")
    if capture_routes:
        _, valid = choice_expert_routes(tensors, 0, np.concatenate((prompt, output_ids)))
        if not valid[:-1].all() or valid[-1]:
            raise RuntimeError("Native binary generation returned misaligned expert routes at the context boundary")
    elif any(choice_tensor_key(0, field) in tensors for field in (EXPERT_IDS, EXPERT_VALID)):
        raise RuntimeError("Native binary generation returned expert routes with capture disabled")
    logger.info(
        "Native binary generation completed with %d prompt tokens, one output token, and route capture %s",
        prompt.size,
        capture_routes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--profile", choices=tuple(SNOWBALL_AGENTX_PROFILES), default=DEFAULT_SNOWBALL_AGENTX_PROFILE)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--capture-routes", action="store_true")
    args = parser.parse_args()
    geometry = SNOWBALL_AGENTX_PROFILES[args.profile]
    require_agentx_devices(geometry)
    config = agentx_engine_config(geometry, args.concurrency, max_rounds=args.max_rounds)
    logging.basicConfig(level=logging.INFO)
    series = load_series()
    if series["max_context"] != AGENTX_CONTEXT:
        raise ValueError("The workload context does not match the native AgentX profile")
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    receipt = read_record(args.workload)
    trainer = TrainerConfig(
        mp=jmp.get_policy("p=bf16,c=bf16"),
        mesh=MeshConfig(axes={"replica": 1, "data": 1, "model": 1, "expert": geometry.expert_shards, "context": 1}),
        use_explicit_mesh_axes=True,
    )
    server_config = InferenceServerConfig(
        trainer=trainer,
        model_name=series["model"],
        service=config,
        batch_timeout=AGENTX_BATCH_TIMEOUT,
        initial_policy_id=args.weights,
    )
    with tempfile.TemporaryDirectory(prefix="snowball-agentx-") as directory:
        bundle = Path(directory) / "bundle"
        restore_workload(receipt, bundle)
        check_native_cache(bundle, SERIES_PATH, output / "workload-check.log")
        revision = series["model_revision"] if args.weights == series["model"] else None
        ref = RepoRef(args.weights, revision)
        converter = HFCheckpointConverter.from_hf(ref)
        model_config = converter.default_config
        if not isinstance(model_config, SnowballConfig):
            raise ValueError(f"Expected a Snowball checkpoint, got {type(model_config).__name__}")
        repo = "models--" + series["tokenizer"].replace("/", "--")
        snapshot = bundle / "hf" / "hub" / repo / "snapshots" / series["tokenizer_revision"]
        tokenizer = load_tokenizer(str(snapshot))
        started = time.monotonic()
        with trainer.use_device_mesh():
            model = load_agentx_model(converter, ref, model_config, trainer.parameter_axis_mapping, geometry)
            server = InferenceServer.create(server_config, model, tokenizer)
        logger.info("Loaded Snowball and KV in %.1f seconds", time.monotonic() - started)
        try:
            warmup_started = time.monotonic()
            filler = tokenizer.encode(" example", add_special_tokens=False)[0]
            _warmup(server, args.capture_routes, filler)
            with bind_serving_socket("127.0.0.1", 0) as sock:
                server_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
                command = benchmark_command(series, args.concurrency, server_url, output / "aiperf_artifacts")
                metrics_arg = command.index("--server-metrics")
                del command[metrics_arg : metrics_arg + 2]
                if args.capture_routes:
                    command.extend(("--extra-inputs", json.dumps({"return_route_captures": True})))
                evidence = {
                    "backend": "levanter",
                    "agentx_profile": args.profile,
                    "agentx_geometry": dataclasses.asdict(geometry),
                    "series": series,
                    "weights": args.weights,
                    "workload_uri": args.workload,
                    "workload_sha256": receipt["archive_sha256"],
                    "devices": [device.device_kind for device in jax.devices()],
                    "concurrency": args.concurrency,
                    "capture_routes": args.capture_routes,
                    "rounds": config.max_rounds,
                    "bucket_warmup_seconds": time.monotonic() - warmup_started,
                    "command": command,
                }
                (output / "probe.json").write_text(json.dumps(evidence, indent=2))
                with serve_app_background(server.app, sock, name="snowball-native-agentx"):
                    _check_chat_stream(server_url, series["model"], args.capture_routes)
                    run_logged(
                        command,
                        workload_environment(bundle, series),
                        output / "agentx.log",
                        timeout=series["warmup_grace_period"] + series["duration"] + 300,
                        sample=output / "client-load.jsonl",
                    )
                    _check_binary_context_limit(
                        server_url, filler, config.max_seq_len, capture_routes=args.capture_routes
                    )
                profile = json.loads((output / PROFILE_EXPORT_RELATIVE_PATH).read_text())
                if profile.get("was_cancelled") or profile.get("request_count", {}).get("avg", 0) == 0:
                    raise RuntimeError("AgentX reported no completed timed requests")
        finally:
            server.shutdown()


if __name__ == "__main__":
    main()
