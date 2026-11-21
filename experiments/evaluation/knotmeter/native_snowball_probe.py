# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay the pinned one-lane 64K AgentX diagnostic against native Snowball on v5p-8."""

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
import requests
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.models.snowball import SnowballConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.inference.dashboard_server import bind_serving_socket, serve_app_background

from experiments.evaluation.knotmeter.artifacts import PROFILE_EXPORT_RELATIVE_PATH, read_record
from experiments.evaluation.knotmeter.processes import run_logged
from experiments.evaluation.knotmeter.recipe import benchmark_command, load_series, workload_environment
from experiments.evaluation.knotmeter.workload import check_native_cache, restore_workload

logger = logging.getLogger(__name__)
SERIES = Path(__file__).with_name("snowball-v01-tpu-64k.yaml")
PROFILE_PROMPT_LENGTHS = (512, 51_870)


def _profile_decode(engine: InferenceEngine, prompt_tokens: list[int], output: Path) -> None:
    """Capture one warmed decode round for a synthetic long prefix."""
    max_output_tokens = 2 * engine.config.max_rounds + 1
    request = Request(
        prompt_tokens=prompt_tokens,
        request_id=0,
        n_generations=1,
        decode_params=dataclasses.replace(
            SeqDecodingParams.default(),
            max_num_tokens=jnp.asarray(len(prompt_tokens) + max_output_tokens, dtype=jnp.int32),
        ),
    )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--attention", choices=("tpu", "tpu_fp32"), default="tpu_fp32")
    parser.add_argument("--decode-layers", choices=("scan", "unrolled"), default="scan")
    parser.add_argument("--profile-decode", action="store_true", help="Trace short and long decode after the AgentX run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    series = load_series(SERIES)
    assert (series["max_context"], series["duration"]) == (65536, 180)
    devices = jax.devices()
    if len(devices) != 4 or any(device.platform != "tpu" for device in devices):
        raise RuntimeError(f"The v5p-8 probe requires four TPU devices, got {devices}")

    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    workload = read_record(args.workload)
    trainer = TrainerConfig(
        mp=jmp.get_policy("p=bf16,c=bf16"),
        mesh=MeshConfig(axes={"replica": 1, "data": 1, "model": 1, "expert": 4, "context": 1}),
        use_explicit_mesh_axes=True,
    )
    service = InferenceEngineConfig(
        max_seq_len=series["max_context"],
        max_seqs=1,
        max_seqs_in_prefill=1,
        max_queued_tokens=32,
        max_tokens_per_round=4,  # Pad the one active decode token to the four-way expert mesh.
    )
    service = dataclasses.replace(service, max_pages=service.max_pages_per_seq)
    config = InferenceServerConfig(trainer=trainer, model_name=series["model"], service=service)

    with tempfile.TemporaryDirectory(prefix="agentx-native-tpu-") as workdir:
        bundle = Path(workdir) / "bundle"
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
        )
        tokenizer = load_tokenizer(series["tokenizer"], revision=series["tokenizer_revision"])
        with trainer.use_device_mesh():
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
            "workload_sha256": workload["archive_sha256"],
            "devices": [device.device_kind for device in devices],
            "mesh": {"model": 1, "expert": 4},
            "max_pages": service.max_pages,
            "max_tokens_per_round": service.max_tokens_per_round,
            "attention": args.attention,
            "decode_layers": args.decode_layers,
            "load_seconds": load_seconds,
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
                probe_path.write_text(json.dumps(evidence, indent=2))
                try:
                    run_logged(
                        command,
                        workload_environment(bundle, series),
                        output / "agentx.log",
                        timeout=series["warmup_grace_period"] + series["duration"] + 600,
                        sample=output / "client-load.jsonl",
                    )
                finally:
                    server.pause_generation()
                profile = json.loads((output / PROFILE_EXPORT_RELATIVE_PATH).read_text())
                if profile.get("was_cancelled") or profile.get("request_count", {}).get("avg", 0) == 0:
                    raise RuntimeError("AgentX did not complete any measured requests")
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
