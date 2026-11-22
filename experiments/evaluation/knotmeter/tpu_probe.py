# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark compiled vLLM on one TPU host."""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import requests
from marin.inference.config import ServedModelConfig, VllmCompilationCacheMode, VllmEngineConfig, VllmLauncherType
from marin.inference.serve import local_inference

from experiments.evaluation.knotmeter.artifacts import PROFILE_EXPORT_RELATIVE_PATH, pack_server_metrics, read_record
from experiments.evaluation.knotmeter.prefill_probe import measure_prefill
from experiments.evaluation.knotmeter.processes import run_logged
from experiments.evaluation.knotmeter.recipe import benchmark_command, load_series, workload_environment
from experiments.evaluation.knotmeter.workload import check_native_cache, restore_workload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    benchmark = parser.add_mutually_exclusive_group(required=True)
    benchmark.add_argument("--workload", help="Immutable prepared workload receipt URI")
    benchmark.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--measure-prefill", action="store_true", help="Also measure prefill after AgentX")
    parser.add_argument("--weights", required=True, help="Regional mirror of the pinned checkpoint")
    parser.add_argument("--chips", type=int, required=True)
    parser.add_argument("--tensor-parallel", type=int, required=True)
    parser.add_argument("--expert-parallel", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-seqs", type=int, default=8)
    parser.add_argument("--compilation-cache-dir", help="Persistent JAX cache in the worker's region")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    series = load_series(args.series)
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    workload = read_record(args.workload) if args.workload else None
    model = ServedModelConfig(
        weights=args.weights,
        api_model=series["model"],
        tokenizer=series["tokenizer"],
        tokenizer_revision=series["tokenizer_revision"],
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        max_model_len=series["max_context"],
    )
    extra_args = ("--enable-prefix-caching", "--gpu-memory-utilization", "0.9")
    if args.expert_parallel > 1:
        sharding = {"sharding": {"sharding_strategy": {"expert_parallelism": args.expert_parallel}}}
        extra_args += ("--enable-expert-parallel", "--additional-config", json.dumps(sharding))
    compilation_cache = VllmCompilationCacheMode.MANAGED
    if args.compilation_cache_dir:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = args.compilation_cache_dir
        os.environ["VLLM_XLA_CACHE_PATH"] = args.compilation_cache_dir
        compilation_cache = VllmCompilationCacheMode.CALLER_MANAGED
    engine = VllmEngineConfig(
        launcher=VllmLauncherType.TPU,
        startup_timeout_seconds=3600,
        max_num_batched_tokens=8192,
        max_num_seqs=args.max_seqs,
        extra_args=extra_args,
        compilation_cache=compilation_cache,
    )

    with tempfile.TemporaryDirectory(prefix="agentx-tpu-") as workdir:
        bundle = Path(workdir) / "bundle"
        if workload is not None:
            restore_workload(workload, bundle)
            check_native_cache(bundle, args.series, output / "workload-check.log")
        with local_inference(model, engine, num_chips=args.chips) as session:
            endpoint = session.model.endpoint
            assert endpoint.model == series["model"]
            requests.get(f"{endpoint.base_url}/models", timeout=30).raise_for_status()
            command = benchmark_command(
                series,
                args.concurrency,
                endpoint.base_url.removesuffix("/v1"),
                output / "aiperf_artifacts",
            )
            (output / "probe.json").write_text(
                json.dumps(
                    {
                        "backend": session.backend_name,
                        "series": series,
                        "weights": args.weights,
                        "workload_sha256": workload["archive_sha256"] if workload is not None else None,
                        "chips": args.chips,
                        "tensor_parallel": args.tensor_parallel,
                        "expert_parallel": args.expert_parallel,
                        "concurrency": args.concurrency,
                        "max_num_batched_tokens": engine.max_num_batched_tokens,
                        "max_num_seqs": engine.max_num_seqs,
                        "compilation_cache_dir": args.compilation_cache_dir,
                        "command": command,
                    },
                    indent=2,
                )
            )
            if args.prefill_only:
                measure_prefill(endpoint.base_url, series["model"], output / "prefill.json")
                session.check_alive()
                return
            try:
                run_logged(
                    command,
                    workload_environment(bundle, series),
                    output / "agentx.log",
                    timeout=series["warmup_grace_period"] + series["duration"] + 600,
                    sample=output / "client-load.jsonl",
                )
                profile = json.loads((output / PROFILE_EXPORT_RELATIVE_PATH).read_text())
                if profile.get("was_cancelled"):
                    raise RuntimeError("AgentX cancelled the probe")
                if args.measure_prefill:
                    measure_prefill(endpoint.base_url, series["model"], output / "prefill.json")
                session.check_alive()
            finally:
                pack_server_metrics(output)


if __name__ == "__main__":
    main()
