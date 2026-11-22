# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate Mochi continuations and seal verifier features without an RL run."""

import argparse
import asyncio
import json
import logging
import tempfile
from pathlib import Path

import httpx
import torch
from iris.cluster.client.job_info import get_job_info
from marin.evaluation.inference_metrics import InferenceMetricWindow
from marin.inference.backend import ModelSpec
from marin.inference.config import ServingGeometry, VllmEngineConfig
from marin.inference.iris_vllm import (
    iris_vllm_followers,
    iris_vllm_launch,
    notify_iris_vllm_stopped,
    wait_for_iris_vllm_shutdown,
)
from marin.inference.vllm_backend import VllmBackend
from marin.inference.vllm_server import VllmEnvironment
from rigging.filesystem.storage_path import StoragePath, prefix_join
from safetensors.torch import load_file
from skyrl_train.inference_engines.base import request_id_for_session
from skyrl_train.io import io
from skyrl_train.trajectory_runners.routed_experts import normalize_routed_experts

from experiments.post_training.long_loop.throughput import graph_capture_observed, write_json
from experiments.post_training.mochi_eagle3 import (
    AUXILIARY_LAYERS,
    TARGET_IDENTITY,
    TARGET_URI,
    TOKENIZER,
    TOKENIZER_REVISION,
)

logger = logging.getLogger(__name__)


def qualify_windows(spec: dict, results: list[dict]) -> list[dict]:
    """Check two regional feature windows against their exact generated trajectories."""
    requests = {row["request_id"]: row for row in results}
    checked = []
    root = StoragePath(spec["output"]) / "captures"
    with tempfile.TemporaryDirectory(prefix="mochi-eagle3-window-check-") as scratch:
        for rank in range(spec["geometry"]["data_parallel_size"]):
            directory = root / f"rank-{rank:05d}"
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest["target_revision"] != TARGET_IDENTITY or tuple(manifest["aux_layer_ids"]) != AUXILIARY_LAYERS:
                raise ValueError(f"Capture rank {rank} target or auxiliary slots changed")
            if manifest["head_input_semantics"] != "post_final_norm_target_lm_head_input":
                raise ValueError("Training requires the post-final-norm target head input")
            for window in manifest["windows"]:
                if len(checked) == 2:
                    break
                request_id = window["request_id"].removeprefix("cmpl-").removesuffix("-0")
                request = json.loads(StoragePath(requests[request_id]["artifact"]).read_text())
                trajectory = torch.tensor(request["prompt_token_ids"] + request["output_token_ids"])
                path = Path(scratch) / "window.safetensors"
                (directory / window["path"]).download_to(str(path))
                tensors = load_file(path)
                positions = tensors["position_ids"]
                if not torch.equal(tensors["input_ids"], trajectory[positions]):
                    raise ValueError("Captured token IDs differ from the generated trajectory")
                if not torch.equal(tensors["loss_mask"], positions >= len(request["prompt_token_ids"])):
                    raise ValueError("Capture loss mask includes prompt or excludes continuation tokens")
                for name, width in (("hidden_states", 18432), ("head_input_hidden_states", 6144)):
                    if tensors[name].shape != (positions.numel(), width) or not torch.isfinite(tensors[name]).all():
                        raise ValueError(f"Invalid captured feature tensor: {name}")
                checked.append(
                    {
                        "request_id": request_id,
                        "tokens": positions.numel(),
                        "first_position": int(positions[0]),
                        "last_position": int(positions[-1]),
                        "supervised_tokens": int(tensors["loss_mask"].sum()),
                    }
                )
    if len(checked) != 2:
        raise ValueError(f"Need two qualified target sequences, got {len(checked)}")
    return checked


async def generate(environment: VllmEnvironment, spec: dict) -> None:
    """Retain exact Mochi continuations and seal bounded per-rank verifier features."""
    rows = [json.loads(line) for line in StoragePath(spec["prompts"]).read_text().splitlines()]
    rows = rows[: spec["prompt_count"]]
    base_url = environment.server_url
    metric_window = await asyncio.to_thread(InferenceMetricWindow.start, environment, speculative=True)
    semaphore = asyncio.Semaphore(spec["concurrency"])
    async with httpx.AsyncClient(timeout=spec["request_timeout"], limits=httpx.Limits(max_connections=1024)) as client:

        async def continuation(index: int, row: dict) -> dict:
            request_id = request_id_for_session(f"{row['group']}_0")
            async with semaphore:
                response = await client.post(
                    f"{base_url}/completions",
                    headers={"x-request-id": request_id},
                    json={
                        "model": "mochi",
                        "prompt": row["prompt_token_ids"],
                        "max_tokens": spec["max_new_tokens"],
                        "temperature": 0.7,
                        "top_p": 1.0,
                        "seed": spec["seed"] + index,
                        "stop_token_ids": [128001, 128009],
                        "return_token_ids": True,
                        "routed_experts_prompt_start": len(row["prompt_token_ids"]) - 1,
                    },
                )
                response.raise_for_status()
                choice = response.json()["choices"][0]
                if not choice["token_ids"]:
                    raise ValueError(f"Empty continuation for {row['identity']}")
                # The final prompt row predicts the first generated token.
                routes = normalize_routed_experts(
                    choice["routed_experts"], row["prompt_token_ids"][-1:], choice["token_ids"]
                )
                if routes.shape[1:] != (48, 8) or routes.max() >= 384:
                    raise ValueError("Mochi expert routes changed shape or expert identity range")
                result = {
                    **row,
                    "request_id": request_id,
                    "output_token_ids": choice["token_ids"],
                    "finish_reason": choice["finish_reason"],
                    "target_identity": TARGET_IDENTITY,
                    "routed_experts": choice["routed_experts"],
                    "routed_experts_shape": list(routes.shape),
                }
                artifact_name = f"continuations/{index:06d}.json"
                write_json(spec["output"], artifact_name, result)
                logger.info("Published continuation %d/%d (%d tokens)", index + 1, len(rows), len(choice["token_ids"]))
                return {
                    "identity": row["identity"],
                    "tokens": len(choice["token_ids"]),
                    "request_id": request_id,
                    "artifact": prefix_join(spec["output"], artifact_name),
                }

        started = await client.post(
            f"{base_url.removesuffix('/v1')}/collective_rpc",
            json={
                "method": "begin_mochi_capture",
                "args": [
                    json.dumps(
                        {
                            "step": 1,
                            "max_tokens": spec["capture_tokens"],
                            "max_window_tokens": spec["window_tokens"],
                            "target_revision": TARGET_IDENTITY,
                            "draft_revision": spec["seed_identity"],
                        }
                    )
                ],
            },
        )
        started.raise_for_status()
        write_json(spec["output"], "capture-start.json", started.json())
        results = await asyncio.gather(*(continuation(i, row) for i, row in enumerate(rows)))
        metrics = await asyncio.to_thread(metric_window.finish)
        write_json(spec["output"], "generation-metrics.json", metrics.model_dump(mode="json"))
        sealed = await client.post(
            f"{base_url.removesuffix('/v1')}/collective_rpc",
            json={"method": "seal_mochi_capture", "args": [prefix_join(spec["output"], "captures")], "timeout": 1800},
        )
        sealed.raise_for_status()
        checked = await asyncio.to_thread(qualify_windows, spec, results)
        root = StoragePath(spec["output"]) / "captures"
        windows = [
            window
            for rank in range(spec["geometry"]["data_parallel_size"])
            for window in json.loads((root / f"rank-{rank:05d}/manifest.json").read_text())["windows"]
        ]
        write_json(
            spec["output"],
            "generation.json",
            {
                "spec": spec,
                "requests": results,
                "seal": sealed.json(),
                "qualified_windows": checked,
                "generated_tokens": sum(row["tokens"] for row in results),
                "captured_supervised_tokens": sum(window["supervised_tokens"] for window in windows),
                "windows_reaching_context_tokens": {
                    str(bound): sum(window["last_position"] + 1 >= bound for window in windows)
                    for bound in (8192, 32768, 65536, 98304)
                },
            },
        )


def mochi_model_spec() -> ModelSpec:
    """Pin the adopted TP1/BF16 Mochi verifier and its tokenizer."""
    return ModelSpec(
        weights=TARGET_URI,
        api_model="mochi",
        num_chips=None,
        tensor_parallel_size=None,
        dtype="bfloat16",
        max_model_len=98304,
        chat_template_content=None,
        tokenizer=TOKENIZER,
        tokenizer_revision=TOKENIZER_REVISION,
    )


def mochi_engine_config(geometry: ServingGeometry, draft: Path | None) -> VllmEngineConfig:
    """Use matched native settings for capture and target/draft evaluation arms."""
    # The GPU-only package is optional in the CPU coordinator environment.
    from vllm.config import CompilationConfig  # noqa: PLC0415

    compilation = CompilationConfig(mode=3, cudagraph_mode="PIECEWISE")
    compilation.set_splitting_ops_for_v1("allgather_reducescatter", data_parallel_size=geometry.data_parallel_size)
    assert compilation.splitting_ops is not None
    compilation.splitting_ops.append("vllm::grug_moe_short_conv")
    arguments = (
        "--enable-expert-parallel",
        "--attention-backend",
        "FLASH_ATTN",
        "--moe-backend",
        "triton",
        "--enable-return-routed-experts",
        "--load-format",
        "runai_streamer",
        "--model-loader-extra-config",
        json.dumps({"distributed": False, "concurrency": 4, "memory_limit": 8589934592}),
        "--gpu-memory-utilization",
        "0.85",
        "--no-enable-prefix-caching",
        "--mamba-cache-mode",
        "none",
        "--no-async-scheduling",
        "--worker-extension-cls",
        "experiments.post_training.mochi_eagle3_worker.MochiCaptureWorker",
        "--compilation-config",
        json.dumps(
            {
                "mode": 3,
                "cudagraph_mode": "PIECEWISE",
                "splitting_ops": compilation.splitting_ops,
            }
        ),
    )
    if draft is not None:
        arguments += (
            "--speculative-config",
            json.dumps(
                {
                    "method": "eagle3",
                    "model": str(draft),
                    "num_speculative_tokens": 3,
                    "max_model_len": 98304,
                }
            ),
        )
    return VllmEngineConfig(
        startup_timeout_seconds=7200,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        extra_args=arguments,
        extra_metric_families=frozenset(
            {
                "vllm:spec_decode_num_drafts",
                "vllm:spec_decode_num_draft_tokens",
                "vllm:spec_decode_num_accepted_tokens",
            }
        ),
    )


def run(spec: dict) -> None:
    """Own the native Iris serving gang for one offline capture interval."""
    info = get_job_info()
    if info is None:
        raise RuntimeError("Run regional capture in Iris")
    if not spec["output"].startswith("s3://marin-us-east-08a/"):
        raise ValueError("Mochi captures must remain in regional 08a S3")
    if (StoragePath(spec["output"]) / "generation.json").exists():
        raise FileExistsError("Completed capture output already exists")
    geometry = ServingGeometry(**spec["geometry"])
    model = mochi_model_spec()
    with tempfile.TemporaryDirectory(prefix="mochi-eagle3-seed-") as scratch:
        draft = Path(scratch) / "draft"
        io.download_directory(spec["seed_draft"], str(draft))
        engine = mochi_engine_config(geometry, draft)
        with iris_vllm_launch(geometry) as launch:
            # Bounded verifier-feature capture is implemented in the V1 runner.
            environment_variables = {
                **launch.subprocess_env,
                "VLLM_SERVER_DEV_MODE": "1",
                "VLLM_USE_V2_MODEL_RUNNER": "0",
            }
            with VllmBackend(engine).start(
                model, extra_args=launch.extra_cli_args, subprocess_env=environment_variables
            ) as environment:
                if launch.is_leader:
                    with iris_vllm_followers(launch):
                        environment.wait_until_ready()
                        asyncio.run(generate(environment, spec))
                else:
                    wait_for_iris_vllm_shutdown(launch, environment.check_alive)
                logs = environment.logs()
                (StoragePath(spec["output"]) / f"task-{info.task_index}-vllm.log").write_text(logs)
                if not graph_capture_observed(logs):
                    raise ValueError(f"CUDA graph capture unconfirmed on task {info.task_index}")
            if not launch.is_leader:
                notify_iris_vllm_stopped(launch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    spec = json.loads(StoragePath(args.spec).read_text())
    spec["source_revision"] = args.source_revision
    run(spec)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
