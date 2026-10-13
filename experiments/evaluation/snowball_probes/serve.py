# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris coordinator for pinned candidate inference servers."""

import argparse
import contextlib
import hashlib
import logging
import threading
import time
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI
from huggingface_hub import HfApi
from iris.client.client import iris_ctx, iris_ctx_scope
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import EndpointAccess
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.external_dependencies import VLLM_GPU_RELEASE, VllmGpuRelease
from marin.inference.config import VllmEngineConfig
from marin.inference.dashboard_server import bind_serving_socket, serve_app_background
from marin.inference.iris import remote_inference
from pydantic import TypeAdapter
from rigging.log_setup import configure_logging

from experiments.evaluation.snowball_probes.config import ORTET_RELAYS, Candidate, ReasoningFormat, load_candidates

logger = logging.getLogger(__name__)
GRUG_PARSER_SOURCE_COMMIT = "ac3b5ea5e8510c002f3d349ea5c0f8b3f20e74ac"
SERVING_STARTUP_TIMEOUT = 3 * 60 * 60
WEIGHT_REUSE_CHANGED_FILES = frozenset({"README.md", "chat_template.jinja"})


def verify_weight_snapshot(candidate: Candidate) -> None:
    """Reject cached weight reuse when runtime artifacts differ from the requested revision."""
    if candidate.weights_revision is None:
        return
    api = HfApi()
    snapshots = []
    for revision in (candidate.weights_revision, candidate.revision):
        info = api.model_info(candidate.checkpoint, revision=revision, files_metadata=True)
        if info.siblings is None:
            raise ValueError("Hugging Face did not return checkpoint artifact metadata")
        snapshots.append(
            {
                item.rfilename: (item.size, item.lfs.sha256 if item.lfs else item.blob_id)
                for item in info.siblings
                if item.rfilename not in WEIGHT_REUSE_CHANGED_FILES
            }
        )
    if snapshots[0] != snapshots[1]:
        raise ValueError("Cached snapshot runtime artifacts differ from the requested checkpoint revision")


def model_config(candidate: Candidate) -> ModelConfig:
    args = ("--enable-prefix-caching",)
    if candidate.reasoning_format == ReasoningFormat.GRUG_INLINE or candidate.reasoning_parser in {
        "grug",
        "grug_optional",
    }:
        args += ("--enable-expert-parallel", "--model-loader-extra-config", '{"distributed":true}')
    if candidate.reasoning_parser == "grug_optional":
        args += ("--reasoning-parser-plugin", str(Path(__file__).with_name("reasoning_parser.py")))
    elif candidate.reasoning_parser == "grug":
        args += ("--reasoning-parser-plugin", str(Path(__file__).with_name("grug_reasoning_parser.py")))
    return ModelConfig(
        name=candidate.key,
        location=candidate.checkpoint,
        revision=candidate.weights_revision or candidate.revision,
        tokenizer=candidate.checkpoint,
        tokenizer_revision=candidate.revision,
        apply_chat_template=True,
        resource_hint=ResourceHint(gpu={"H100": candidate.gpu_count}, memory="256g", disk="500g", cpu=8),
        serve=ServeConfig(
            tensor_parallel_size=candidate.tensor_parallel_size,
            data_parallel_size=candidate.data_parallel_size,
            max_model_len=65536,
            max_num_batched_tokens=4096,
            max_num_seqs=16,
            reasoning_parser=candidate.reasoning_parser,
            tool_call_parser=candidate.tool_call_parser,
            vllm_extra_args=args,
        ),
    )


def run(candidates: tuple[Candidate, ...], cluster: str, gpu_release: VllmGpuRelease, startup_timeout: float) -> None:
    ctx = iris_ctx()
    info = get_job_info()
    assert info is not None
    assert ctx.client is not None
    stop = threading.Event()
    states = {candidate.key: {**candidate.metadata(), "state": "starting"} for candidate in candidates}
    parser_revision = hashlib.sha256(Path(__file__).with_name("reasoning_parser.py").read_bytes()).hexdigest()
    for candidate in candidates:
        states[candidate.key]["startup_timeout"] = startup_timeout
        states[candidate.key]["vllm_release"] = gpu_release.release_tag
        states[candidate.key]["vllm_source_commit"] = gpu_release.source_commit
        if candidate.reasoning_parser == "grug_optional":
            states[candidate.key]["reasoning_parser_revision"] = parser_revision
        elif candidate.reasoning_parser == "grug":
            states[candidate.key]["reasoning_parser_revision"] = hashlib.sha256(
                Path(__file__).with_name("grug_reasoning_parser.py").read_bytes()
            ).hexdigest()
            states[candidate.key]["reasoning_parser_source_commit"] = GRUG_PARSER_SOURCE_COMMIT
    lock = threading.Lock()

    def serve_candidate(candidate: Candidate) -> None:
        with iris_ctx_scope(ctx):
            try:
                verify_weight_snapshot(candidate)
                if candidate.weights_revision is not None:
                    with lock:
                        states[candidate.key]["weights_equivalence"] = "HF runtime artifact sizes and hashes match"
                config = inference_config_for_model(
                    model_config(candidate),
                    AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=candidate.gpu_count),
                    env_vars={},
                    priority=0,
                    api_model=candidate.key,
                )
                assert isinstance(config.engine, VllmEngineConfig)
                config = replace(
                    config,
                    engine=replace(config.engine, gpu_release=gpu_release, startup_timeout_seconds=int(startup_timeout)),
                    iris=replace(config.iris, max_retries_failure=3, endpoint_ready_timeout_seconds=startup_timeout),
                )
                with remote_inference(config) as session:
                    with lock:
                        states[candidate.key].update(
                            state="ready",
                            endpoint=session.endpoint_name,
                            jobs=[str(job.job_id) for job in session.jobs],
                        )
                    while not stop.wait(15):
                        session.check_alive()
            except Exception as error:
                logger.exception("Candidate server failed: %s", candidate.key)
                with lock:
                    states[candidate.key].update(state="failed", error=str(error))

    api = FastAPI()

    @api.get("/status")
    def status() -> dict:
        with lock:
            return {"cluster": cluster, "candidates": [dict(value) for value in states.values()]}

    threads = [threading.Thread(target=serve_candidate, args=(candidate,), daemon=True) for candidate in candidates]
    with contextlib.ExitStack() as stack:
        sock = stack.enter_context(bind_serving_socket(info.advertise_host, 0))
        server = stack.enter_context(serve_app_background(api, sock, name="snowball-probes"))
        stack.enter_context(
            ctx.registry.registered(
                "probes",
                f"http://{info.advertise_host}:{sock.getsockname()[1]}",
                metadata={"kind": "snowball-probes"},
                access=EndpointAccess.ENDPOINT_ACCESS_LINK,
            )
        )
        for thread in threads:
            thread.start()
        try:
            while server.is_alive():
                time.sleep(15)
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=30)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", choices=ORTET_RELAYS, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--vllm-release", type=Path)
    parser.add_argument("--startup-timeout", type=float, default=SERVING_STARTUP_TIMEOUT)
    args = parser.parse_args()
    release = (
        TypeAdapter(VllmGpuRelease).validate_json(args.vllm_release.read_text())
        if args.vllm_release is not None
        else VLLM_GPU_RELEASE
    )
    run(load_candidates(args.candidates), args.cluster, release, args.startup_timeout)


if __name__ == "__main__":
    main()
