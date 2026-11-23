# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen 64K AgentX replay against native Snowball on one v5p-8."""

import argparse
import dataclasses
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import haliax as hax
import jax
import jax.numpy as jnp
import jmp
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.inference.engine import InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.inference.snowball_loading import load_snowball_serving_model
from levanter.layers.sampler import LogprobMode
from levanter.models.snowball import SnowballConfig
from levanter.tokenizers import load_tokenizer
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.inference.dashboard_server import bind_serving_socket, serve_app_background

from experiments.evaluation.knotmeter.artifacts import PROFILE_EXPORT_RELATIVE_PATH, read_record
from experiments.evaluation.knotmeter.processes import run_logged
from experiments.evaluation.knotmeter.recipe import SERIES_PATH, benchmark_command, load_series, workload_environment
from experiments.evaluation.knotmeter.workload import check_native_cache, restore_workload

logger = logging.getLogger(__name__)
_TPU_CHIPS = 4
_PREFILL_BUCKETS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 49152)
_DECODE_BUCKETS = (4, 8, 16)


def _request(tokens: list[int], request_id: int, generated: int) -> Request:
    params = dataclasses.replace(SeqDecodingParams.default(), max_num_tokens=jnp.int32(len(tokens) + generated))
    return Request(tokens, request_id, params, n_generations=1)


def _warmup(server: InferenceServer, capture_routes: bool) -> None:
    ctx = server.inference_context
    config = ctx.engine.config
    filler = ctx.tokenizer.encode(" example", add_special_tokens=False)[0]
    with ctx.model_lock, ctx.config.trainer.use_device_mesh(), hax.axis_mapping(ctx.config.trainer.compute_axis_mapping):
        ctx.engine.generate([_request([filler], 0, 2)], capture_routes=capture_routes)
        for index, lower in enumerate(config.prefill_bucket_sizes):
            prompt = [index, *([filler] * lower)]
            ctx.engine.generate([_request(prompt, 0, 1)], capture_routes=capture_routes)
        for lower in config.decode_bucket_sizes:
            requests = [_request([filler], i, 2) for i in range(lower + 1)]
            ctx.engine.generate(requests, capture_routes=capture_routes)
        ctx.engine.reset()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--capture-routes", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    devices = jax.devices()
    if len(devices) != _TPU_CHIPS or any(device.device_kind != "TPU v5" for device in devices):
        raise RuntimeError(f"Expected four v5p chips, got {devices}")
    logging.basicConfig(level=logging.INFO)
    series = load_series()
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    receipt = read_record(args.workload)
    config = InferenceEngineConfig(
        max_seq_len=series["max_context"],
        max_seqs=args.concurrency,
        max_pages=49152,
        window_cache_max_pages=5120,
        window_cache_retained_spans=2,
        max_retained_prompts=512,
        max_queued_tokens=max(32, args.concurrency),
        max_seqs_in_prefill=min(4, args.concurrency),
        max_prefill_size=series["max_context"],
        max_rounds=args.max_rounds,
        max_tokens_per_round=32,
        prefill_bucket_sizes=_PREFILL_BUCKETS,
        decode_bucket_sizes=tuple(size for size in _DECODE_BUCKETS if size < args.concurrency),
        compute_dtype=jnp.bfloat16,
        logprob_mode=LogprobMode.POLICY,
    )
    trainer = TrainerConfig(
        mp=jmp.get_policy("p=bf16,c=bf16"),
        mesh=MeshConfig(axes={"replica": 1, "data": 1, "model": 1, "expert": _TPU_CHIPS, "context": 1}),
        use_explicit_mesh_axes=True,
    )
    server_config = InferenceServerConfig(
        trainer=trainer, model_name=series["model"], service=config, batch_timeout=0.001, initial_policy_id=args.weights
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
            model = load_snowball_serving_model(
                converter,
                ref,
                model_config,
                dtype=trainer.mp.compute_dtype,
                axis_mapping=trainer.parameter_axis_mapping,
                pack_expert_weights=True,
            )
            model = model.with_paged_attention("tpu_v3").with_split_decode_cache()
            model = model.with_expert_sharded_kv().with_selected_ep_decode()
            server = InferenceServer.create(server_config, model, tokenizer)
        logger.info("Loaded Snowball and KV in %.1f seconds", time.monotonic() - started)
        try:
            warmup_started = time.monotonic()
            _warmup(server, args.capture_routes)
            with bind_serving_socket("127.0.0.1", 0) as sock:
                server_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
                command = benchmark_command(series, args.concurrency, server_url, output / "aiperf_artifacts")
                metrics_arg = command.index("--server-metrics")
                del command[metrics_arg : metrics_arg + 2]
                if args.capture_routes:
                    command.extend(("--extra-inputs", json.dumps({"return_route_captures": True})))
                evidence = {
                    "backend": "levanter",
                    "series": series,
                    "weights": args.weights,
                    "workload_uri": args.workload,
                    "workload_sha256": receipt["archive_sha256"],
                    "devices": [device.device_kind for device in devices],
                    "concurrency": args.concurrency,
                    "capture_routes": args.capture_routes,
                    "rounds": config.max_rounds,
                    "bucket_warmup_seconds": time.monotonic() - warmup_started,
                    "command": command,
                }
                (output / "probe.json").write_text(json.dumps(evidence, indent=2))
                with serve_app_background(server.app, sock, name="snowball-native-agentx"):
                    run_logged(
                        command,
                        workload_environment(bundle, series),
                        output / "agentx.log",
                        timeout=series["warmup_grace_period"] + series["duration"] + 300,
                        sample=output / "client-load.jsonl",
                    )
                profile = json.loads((output / PROFILE_EXPORT_RELATIVE_PATH).read_text())
                if profile.get("was_cancelled") or profile.get("request_count", {}).get("avg", 0) == 0:
                    raise RuntimeError("AgentX reported no completed timed requests")
        finally:
            server.shutdown()


if __name__ == "__main__":
    main()
