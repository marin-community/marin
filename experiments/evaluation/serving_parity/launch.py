# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the first 100 GSM8K test items on an H100 node or a v4-32 slice."""

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from finestore.reader import ReadView
from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import Entrypoint, EnvironmentSpec
from marin.evaluation.evalchemy.client import CONFIG_ENV_KEY
from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.evalchemy.runner import _evalchemy_client_command, _run_config_json
from marin.evaluation.lm_eval_samples import read_evalchemy_results
from marin.inference.config import ServedModelConfig, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.iris import _vllm_model_spec
from marin.inference.iris_tpu_vllm import RAY_PORTS, tpu_ray_gang
from marin.inference.serve import local_inference
from marin.inference.types import OpenAIEndpoint, RunningModel
from marin.inference.vllm_backend import VllmBackend, vllm_launcher
from marin.inference.vllm_server import IsolatedTpuVllm
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.evaluation.evals import evalchemy_run_config

MODEL = "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02"
REVISION = "50fe2f488aee7730bbccc721024b35229075602e"
ITEM_COUNT = 100
RUNTIME_SECONDS = 3600
POLICY = "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/evalchemy-configs/gsm8k-0shot.yaml"


def evaluate(model, output: StoragePath, timeout: float) -> list[dict]:
    """Use the existing campaign evaluator, retaining its logs and FineStore samples."""
    config = dataclasses.replace(
        evalchemy_run_config("gsm8k-0shot", load_evalchemy_config(Path(POLICY))),
        apply_chat_template=True,
        max_length=8192,
        max_eval_instances=ITEM_COUNT,
        num_concurrent=16,
        seed=42,
        max_gen_toks=2048,
        extra_gen_kwargs={"temperature": "0", "top_p": "1", "top_k": "-1"},
    )
    archive = str(output / "eval")
    env = dict(
        os.environ,
        **{CONFIG_ENV_KEY: _run_config_json(model, config, archive)},
        JAX_PLATFORMS="cpu",
        CUDA_VISIBLE_DEVICES="",
    )
    env.pop("PYTHONPATH", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    command = [*_evalchemy_client_command(config.runtime), "lib/marin/src/marin/evaluation/evalchemy/client.py"]
    with tempfile.TemporaryFile(mode="w+") as log:
        try:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
        finally:
            log.seek(0)
            (output / "eval.log").write_text(log.read())
    documents = read_evalchemy_results(archive)
    if not documents or any(
        outcome["status"] != "succeeded" or outcome["scored_count"] != ITEM_COUNT
        for document in documents.values()
        for outcome in document["task_outcomes"].values()
    ):
        raise ValueError("GSM8K did not finish all 100 items; see eval.log")
    return collect_items(archive)


def collect_items(archive: str) -> list[dict]:
    """Read scored items from the evaluator archive, including Arrow map metrics."""
    items = []
    for row in ReadView(archive).iter_rows("samples"):
        metrics = dict(row["metrics"])
        if row["filter"] != "flexible-extract" or "exact_match" not in metrics:
            continue
        document = json.dumps(json.loads(row["doc"]), sort_keys=True, separators=(",", ":"))
        items.append(
            dict(
                id=str(row["doc_id"]),
                sha256=hashlib.sha256(document.encode()).hexdigest(),
                correct=float(metrics["exact_match"]),
            )
        )
    if len(items) != ITEM_COUNT or len({item["id"] for item in items}) != ITEM_COUNT:
        raise ValueError("GSM8K sample coverage is incomplete or duplicated")
    return sorted(items, key=lambda item: item["id"])


@contextlib.contextmanager
def serving_model(hardware, model, engine):
    """Reuse the single-host backend or the multi-host TPU launcher."""
    if hardware == "h100":
        with local_inference(model, engine, num_chips=8) as session:
            yield session.model
            session.check_alive()
        return
    original_command = IsolatedTpuVllm.command

    def command(launcher):
        bootstrap = Path("experiments/evaluation/serving_parity/patch_v4_tiles.py").resolve()
        return [*original_command(launcher)[:-1], "python", str(bootstrap), "vllm"]

    # Patching a private uv environment leaves the shared pin and other jobs untouched.
    with tempfile.TemporaryDirectory(prefix=".v4-score-", dir=Path.cwd()) as directory:
        temporary = Path(directory) / "tmp"
        temporary.mkdir()
        with patch.dict(os.environ, UV_CACHE_DIR=str(Path(directory) / "uv"), TMPDIR=str(temporary)):
            with patch.object(IsolatedTpuVllm, "command", command):
                launcher = vllm_launcher(engine)
                assert isinstance(launcher, IsolatedTpuVllm)
                with tpu_ray_gang(launcher, chips_per_host=4, timeout_seconds=2400) as gang:
                    if not gang.leader:
                        yield None
                        return
                    spec = _vllm_model_spec(model, 16, 16)
                    with VllmBackend(engine).start(
                        spec, extra_args=("--distributed-executor-backend", "ray"), subprocess_env=gang.subprocess_env
                    ) as environment:
                        environment.wait_until_ready()
                        url = environment.server_url.rstrip("/").removesuffix("/v1") + "/v1"
                        yield RunningModel(OpenAIEndpoint(url, MODEL), tokenizer=MODEL)
                        gang.check_alive()
                        environment.check_alive()


def run_arm(hardware: str, weights: str, output: str) -> None:
    """Own the server and evaluator in one bounded Iris job; release on exit."""
    configure_coreweave_s3()
    root = StoragePath(output)
    info = get_job_info()
    if info is None:
        raise RuntimeError("Run this experiment inside an Iris job")
    if info.task_index == 0:
        if (root / "started.json").exists():
            raise ValueError("Use a fresh output prefix for each run")
        (root / "started.json").write_text(json.dumps(dict(hardware=hardware, epoch=time.time())))
    started = time.monotonic()
    gpu = hardware == "h100"
    model = ServedModelConfig(
        weights=weights,
        api_model=MODEL,
        tokenizer=MODEL,
        tokenizer_revision=REVISION,
        revision=REVISION if weights == MODEL else None,
        max_model_len=8192,
        tensor_parallel_size=1 if gpu else 16,
    )
    engine = VllmEngineConfig(
        launcher=VllmLauncherType.CUDA if gpu else VllmLauncherType.TPU,
        source=VllmSource.MARIN_FORK if gpu else VllmSource.UPSTREAM,
        startup_timeout_seconds=2400,
        max_num_batched_tokens=512,
        max_num_seqs=16,
        extra_args=("--trust-remote-code", "--load-format", "runai_streamer")
        + (
            (
                "--data-parallel-size",
                "8",
                "--enable-expert-parallel",
                "--model-loader-extra-config",
                '{"distributed":true}',
            )
            if gpu
            else ()
        ),
    )
    with serving_model(hardware, model, engine) as running_model:
        if running_model is None:
            return
        ready = time.monotonic()
        items = evaluate(running_model, root, max(1, RUNTIME_SECONDS - (ready - started) - 60))
    result = dict(
        hardware=hardware,
        model=MODEL,
        revision=REVISION,
        items=items,
        score=sum(item["correct"] for item in items) / ITEM_COUNT,
        startup_seconds=ready - started,
        wall_seconds=time.monotonic() - started,
    )
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "items"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", choices=("h100", "v4"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--weights", required=True, help="Pinned regional Step12 checkpoint")
    parser.add_argument("--output", required=True, help="Fresh S3 (H100) or GCS (TPU) prefix")
    parser.add_argument("--submit", action="store_true", help="Otherwise print the plan only")
    args = parser.parse_args()
    gpu = args.hardware == "h100"
    resources = (
        ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="512g", disk="200g", target_cluster="cw-rno2a")
        if gpu
        else ResourceConfig.with_tpu(
            "v4-32", cpu=32, ram="360g", disk="100g", regions=["us-central2"], zone="us-central2-b", preemptible=False
        )
    )
    ports = [*RAY_PORTS] if args.hardware == "v4" else []
    coscheduling = resolve_coscheduling(resources, resources.replicas)
    print(
        json.dumps(dict(vars(args), replicas=resources.replicas, ports=ports, coscheduling=str(coscheduling)), indent=2)
    )
    if not args.submit:
        return
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, workspace=Path.cwd(), credentials=endpoint.credentials) as client:
            job = client.submit(
                name=args.name,
                entrypoint=Entrypoint.from_callable(run_arm, args.hardware, args.weights, args.output),
                replicas=resources.replicas,
                coscheduling=coscheduling,
                ports=ports,
                resources=convert_resources(resources),
                constraints=convert_constraints(resources),
                environment=EnvironmentSpec(
                    extras=("cpu",) if gpu else ("tpu",),
                    env_vars={
                        key: os.environ[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET", "HF_TOKEN") if key in os.environ
                    },
                ),
                timeout=Duration.from_seconds(RUNTIME_SECONDS),
                scheduling_timeout=Duration.from_seconds(3600),
                max_retries_failure=0,
                max_retries_preemption=0,
                max_task_failures=0,
            )
            print(job)


if __name__ == "__main__":
    main()
