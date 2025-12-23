# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Literal

from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator
from marin.evaluation.helmet.config import HelmetEvalName
from marin.evaluation.utils import is_remote_path
from marin.utils import fsspec_copy_path_into_dir
from marin.utils import remove_tpu_lockfile_on_exit


def _local_path_for_gcsfuse_mount(path: str, *, local_mount_root: str = "/opt/gcsfuse_mount") -> str:
    marker = "gcsfuse_mount/"
    if marker not in path:
        raise ValueError(f"Expected a path under {marker}, got: {path}")
    relative = path.split(marker, 1)[1].lstrip("/")
    return os.path.join(local_mount_root, relative)


def _looks_like_hf_repo_id(value: str) -> bool:
    parts = value.split("/")
    return len(parts) == 2 and all(parts)


def _model_config_for_vllm(*, run_name: str, model_name_or_path: str) -> ModelConfig:
    if _looks_like_hf_repo_id(model_name_or_path):
        return ModelConfig(name=model_name_or_path, path=None, engine_kwargs={})

    if "gcsfuse_mount/" in model_name_or_path:
        local = _local_path_for_gcsfuse_mount(model_name_or_path)
        return ModelConfig(name=local, path=None, engine_kwargs={})

    if is_remote_path(model_name_or_path):
        return ModelConfig(name=run_name, path=model_name_or_path, engine_kwargs={})

    return ModelConfig(name=model_name_or_path, path=None, engine_kwargs={})


def _sync_local_dir_to_output(local_dir: str, output_path: str) -> None:
    fsspec_copy_path_into_dir(src_path=local_dir, dst_path=output_path)


@dataclass(frozen=True)
class HelmetRunConfig:
    run_name: str
    model_name_or_path: str

    helmet_repo_url: str
    helmet_repo_sha: str

    helmet_data_output_path: str
    """Executor-resolved output path of the data step (must be under `gcsfuse_mount/`)."""

    evals: tuple[HelmetEvalName, ...]
    config_variant: Literal["full", "short"] = "full"

    use_chat_template: bool = False
    seed: int = 42
    tag: str = "v1"

    output_path: str = field(default_factory=str)
    resource_config: ResourceConfig = field(default_factory=ResourceConfig)

    vllm_serve_args: tuple[str, ...] = ()


def run_helmet(config: HelmetRunConfig) -> None:
    """Executor step entrypoint: launch a single TPU job to run 1+ HELMET configs."""

    def _run():
        with remove_tpu_lockfile_on_exit():
            _run_on_tpu(config)

    job_request = JobRequest(
        name=f"helmet:{config.run_name}",
        entrypoint=Entrypoint.from_callable(_run),
        resources=config.resource_config,
        environment=EnvironmentConfig.create(extras=["eval", "tpu", "helmet"]),
    )

    cluster = current_cluster()
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)


def _run_on_tpu(config: HelmetRunConfig) -> None:
    start = time.time()

    helmet_data_dir = _local_path_for_gcsfuse_mount(config.helmet_data_output_path)
    if not os.path.exists(os.path.join(helmet_data_dir, "_SUCCESS")):
        # list the files that may or may not exist, for debugging. this dir and parent
        parent_dir = os.path.dirname(helmet_data_dir)
        print(f"Checking existence of HELMET data dir: {os.listdir('/opt/gcsfuse_mount')}")
        print(f"Contents of parent dir: {parent_dir}: {os.listdir(parent_dir)}")
        print(f"Contents of helmet data dir: {helmet_data_dir}: {os.listdir(helmet_data_dir)}")
        raise RuntimeError(f"HELMET data directory is not ready: {helmet_data_dir}")

    with tempfile.TemporaryDirectory(prefix="helmet_repo_") as tmpdir:
        repo_dir = os.path.join(tmpdir, "HELMET")
        subprocess.run(["git", "clone", config.helmet_repo_url, repo_dir], check=True)
        subprocess.run(["git", "checkout", config.helmet_repo_sha], check=True, cwd=repo_dir)

        data_src = os.path.join(helmet_data_dir, "data")
        data_dst = os.path.join(repo_dir, "data")
        if os.path.lexists(data_dst):
            if os.path.islink(data_dst) or os.path.isfile(data_dst):
                os.remove(data_dst)
            else:
                shutil.rmtree(data_dst)
        os.symlink(data_src, data_dst)

        model = _model_config_for_vllm(run_name=config.run_name, model_name_or_path=config.model_name_or_path)
        model_name_or_path = VllmTpuEvaluator.download_model(model)

        server_url = VllmTpuEvaluator.start_vllm_server_in_background(
            model=model,
            host="127.0.0.1",
            port=8000,
            timeout_seconds=3600,
            extra_args=list(config.vllm_serve_args) if config.vllm_serve_args else None,
        )
        endpoint_url = f"{server_url}/"

        local_output_dir = os.path.join(tmpdir, "output")

        def config_path(eval_name: HelmetEvalName) -> str:
            suffix = "" if config.config_variant == "full" else "_short"
            return os.path.join(repo_dir, "configs", f"{eval_name}{suffix}.yaml")

        ran = []
        try:
            for eval_name in config.evals:
                cfg = config_path(eval_name)
                if not os.path.exists(cfg):
                    raise FileNotFoundError(f"Missing HELMET config: {cfg}")

                subprocess.run(
                    [
                        sys.executable,
                        "eval.py",
                        "--config",
                        cfg,
                        "--seed",
                        str(config.seed),
                        "--output_dir",
                        local_output_dir,
                        "--tag",
                        config.tag,
                        "--model_name_or_path",
                        model_name_or_path,
                        "--use_chat_template",
                        str(bool(config.use_chat_template)),
                        "--use_vllm_serving",
                        "--endpoint_url",
                        endpoint_url,
                        "--api_key",
                        "EMPTY",
                        "--overwrite",
                        "--no_cuda",
                    ],
                    check=True,
                    cwd=repo_dir,
                )
                ran.append(eval_name)
        finally:
            VllmTpuEvaluator.cleanup(model, vllm_port=8000)

        os.makedirs(local_output_dir, exist_ok=True)
        with open(os.path.join(local_output_dir, "marin_metadata.json"), "w") as f:
            json.dump(
                {
                    "run_name": config.run_name,
                    "model_name_or_path": config.model_name_or_path,
                    "helmet_repo_url": config.helmet_repo_url,
                    "helmet_repo_sha": config.helmet_repo_sha,
                    "helmet_data_output_path": config.helmet_data_output_path,
                    "evals": list(ran),
                    "config_variant": config.config_variant,
                    "use_chat_template": config.use_chat_template,
                    "seed": config.seed,
                    "tag": config.tag,
                    "wall_time_seconds": time.time() - start,
                },
                f,
                indent=2,
            )

        _sync_local_dir_to_output(local_output_dir, config.output_path)
