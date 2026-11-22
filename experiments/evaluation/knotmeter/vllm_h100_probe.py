# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned Snowball AgentX comparison against compiled vLLM on H100."""

import argparse
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import ServedModelConfig, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.serve import local_inference

from experiments.evaluation.knotmeter.artifacts import PROFILE_EXPORT_RELATIVE_PATH, read_record
from experiments.evaluation.knotmeter.processes import run_logged
from experiments.evaluation.knotmeter.recipe import SERIES_PATH, benchmark_command, load_series, workload_environment
from experiments.evaluation.knotmeter.workload import check_native_cache, restore_workload

H100_COUNT = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--workload", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(devices) != H100_COUNT or any("H100" not in device for device in devices):
        raise RuntimeError(f"Expected eight H100 devices, got {devices}")
    series = load_series(SERIES_PATH)
    workload = read_record(args.workload)
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    model = ServedModelConfig(
        weights=args.weights,
        api_model=series["model"],
        tokenizer=series["tokenizer"],
        tokenizer_revision=series["tokenizer_revision"],
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=series["max_context"],
    )
    engine = VllmEngineConfig(
        launcher=VllmLauncherType.CUDA,
        source=VllmSource.MARIN_FORK,
        max_num_batched_tokens=8192,
        max_num_seqs=8,
        extra_args=(
            "--enable-prefix-caching",
            "--gpu-memory-utilization",
            "0.9",
            "--data-parallel-size",
            str(H100_COUNT),
            "--enable-expert-parallel",
            "--all2all-backend",
            "allgather_reducescatter",
            "--attention-backend",
            "FLASH_ATTN",
        ),
    )

    with tempfile.TemporaryDirectory(prefix="agentx-vllm-h100-") as workdir:
        bundle = Path(workdir) / "bundle"
        restore_workload(workload, bundle)
        check_native_cache(bundle, SERIES_PATH, output / "workload-check.log")
        with local_inference(model, engine, num_chips=H100_COUNT) as session:
            endpoint = session.model.endpoint
            url = endpoint.base_url.removesuffix("/v1")
            command = benchmark_command(series, 1, url, output / "aiperf_artifacts")
            (output / "probe.json").write_text(
                json.dumps(
                    {
                        "backend": session.backend_name,
                        "vllm_release": VLLM_GPU_RELEASE.release_tag,
                        "series": series,
                        "weights": args.weights,
                        "workload_sha256": workload["archive_sha256"],
                        "devices": devices,
                        "tensor_parallel": model.tensor_parallel_size,
                        "engine_args": engine.extra_args,
                        "max_num_batched_tokens": engine.max_num_batched_tokens,
                        "max_num_seqs": engine.max_num_seqs,
                        "command": command,
                    },
                    indent=2,
                )
            )
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
            session.check_alive()


if __name__ == "__main__":
    main()
