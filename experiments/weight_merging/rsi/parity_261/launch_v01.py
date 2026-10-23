"""Task261 v0.1 six-policy parity suite on the validated multi-host launcher."""

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import requests
from finestore.reader import ReadView
from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.evalchemy.runner import _run_config_json
from marin.inference.iris_tpu_vllm import RAY_PORTS
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.evaluation.evals import evalchemy_run_config
from experiments.weight_merging.rsi.parity_261 import launch as core

REPO = "open-athena/Marin-Snowball-67B-A2B-v0.1"
REVISION = "adc0078a19a4383bc998fcdb43867b3360baf5f3"
MODEL = "gs://marin-us-central2/scratch/karan/rsi-merge/parents-clean/v01--open-athena_Marin-Snowball-67B-A2B-v0.1@adc0078a19a4/"
OUTPUT = "gs://marin-us-central2/scratch/karan/task261/v01-r1/"
NAME = "rsi261-v01-v4-32-r1"
COMMIT = "c1d5c62cdc3939b42f64411af44fd50716b53454"
KEYS = ("math500", "gsm8k-0shot", "boolq", "piqa", "humanevalplus", "aime24")
RUNTIME_SECONDS = 6 * 3600


def policy_plan(key, base_url="http://resolved-worker-local/v1"):
    path = core.HERE / "policy_configs" / f"{key}.yaml"
    source = load_evalchemy_config(path)
    config = dataclasses.replace(
        evalchemy_run_config(key, source), apply_chat_template=True, num_concurrent=16, max_length=73664
    )
    if any(not task.generation for task in config.tasks) and (
        config.max_gen_toks is not None or config.extra_gen_kwargs
    ):
        raise ValueError("Loglikelihood tasks must not carry generation overrides")
    payload = json.loads(
        _run_config_json(RunningModel(OpenAIEndpoint(base_url, REPO), tokenizer=REPO), config, OUTPUT + key + "/")
    )
    return dict(
        config=payload,
        extras=sorted({"serve-eval", *source.runtime_extras}),
        policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        yaml=path.read_text(),
    )


def run_suite(base_url, remaining_seconds):
    deadline = time.monotonic() + remaining_seconds
    summary = []
    source = core.HERE / "policy_client.py"
    client_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    for key in KEYS:
        plan = policy_plan(key, base_url)
        payload = plan["config"]
        command = [
            "uvx",
            "--no-config",
            "--python",
            "3.12",
            "--from",
            f"evalchemy[{','.join(plan['extras'])}] @ git+https://github.com/marin-community/evalchemy.git@{COMMIT}",
            "--with",
            "s3fs",
            "--with",
            "gcsfs",
        ]
        for arch in ("x86_64", "aarch64"):
            command += [
                "--with",
                f'torch @ https://download.pytorch.org/whl/cpu/torch-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_{arch}.whl ; sys_platform == "linux" and platform_machine == "{arch}"',
            ]
        root = StoragePath(OUTPUT) / key
        metadata = {
            **plan,
            "config": {**payload, "base_url": "<worker-local omitted>"},
            "evalchemy_commit": COMMIT,
            "client_sha256": client_sha,
            "model_revision": REVISION,
            "server_context": 73728,
            "max_num_seqs": 16,
            "settings_note": "Campaign YAML verbatim; no seed override. Effective73664 and concurrency16 match task249.",
        }
        (root / "settings.json").write_text(json.dumps(metadata, indent=2))
        metrics_url = base_url.removesuffix("/v1") + "/metrics"
        (root / "metrics-before.txt").write_text(requests.get(metrics_url, timeout=30).text)
        env = dict(
            os.environ,
            EVALCHEMY_CLIENT_CONFIG=json.dumps(payload),
            JAX_PLATFORMS="cpu",
            CUDA_VISIBLE_DEVICES="",
            OPENAI_API_KEY="local-endpoint",
            HF_ALLOW_CODE_EVAL="1",
            TQDM_MININTERVAL="30",
        )
        for name in ("PYTHONPATH", "UV_PROJECT_ENVIRONMENT"):
            env.pop(name, None)
        log = Path(env["TMPDIR"]) / f"eval-{key}.log"
        started = time.time()
        try:
            with log.open("w") as stream:
                result = subprocess.run(
                    [*command, "python", str(source.resolve())],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=max(1, deadline - time.monotonic()),
                    check=False,
                )
        finally:
            (root / "client.log").write_bytes(log.read_bytes())
            (root / "metrics-after.txt").write_text(requests.get(metrics_url, timeout=30).text)
        (root / "process.json").write_text(
            json.dumps(
                dict(started_epoch=started, elapsed_seconds=time.time() - started, returncode=result.returncode),
                indent=2,
            )
        )
        if result.returncode:
            raise RuntimeError(f"{key} client exited{result.returncode}; inspect archived log")
        table = ReadView(str(root)).scan("evalchemy_results")
        if table is None:
            raise RuntimeError(f"{key}: no aggregate result")
        rows = table.to_pylist(maps_as_pydicts="strict")
        (root / "results.json").write_text(json.dumps(rows, indent=2, default=str))
        if not rows or not all(row.get("scored") is True for row in rows):
            raise RuntimeError(f"{key}: unscored aggregate result")
        documents = [json.loads(row["document"]) for row in rows]
        record = dict(
            eval=key,
            wall_seconds=time.time() - started,
            results=[doc.get("results") for doc in documents],
            outcomes=[doc.get("task_outcomes") for doc in documents],
        )
        summary.append(record)
        (StoragePath(OUTPUT) / "suite-summary.json").write_text(json.dumps(summary, indent=2))
        print("TASK261_V01_RESULT", json.dumps(record), flush=True)


def run_host():
    # All worker-local bindings occur inside the serialized entrypoint. Imported
    # modules are restored by reference, not by serializing submit-host globals.
    core.MODEL = MODEL
    core.REPO = REPO
    core.REVISION = REVISION
    core.OUTPUT = OUTPUT
    core.NAME = NAME
    core.RUNTIME_SECONDS = RUNTIME_SECONDS
    core.run_evaluation = run_suite
    original_build = core.build_service

    def build():
        service = original_build()
        return dataclasses.replace(service, engine=dataclasses.replace(service.engine, max_num_seqs=16), timeout_hours=6)

    core.build_service = build
    core.run_host()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    resource = core.resources()
    core.build_service()  # Validate full service types before allocation.
    plan = dict(
        name=NAME,
        model=MODEL,
        revision=REVISION,
        output=OUTPUT,
        TP=16,
        PP=1,
        hosts=4,
        context=73728,
        max_num_seqs=16,
        runtime_hours=6,
        queue_hours=4,
        coscheduling=str(resolve_coscheduling(resource, resource.replicas)),
        policies={key: policy_plan(key) for key in KEYS},
    )
    print(json.dumps(plan, indent=2))
    if not args.submit:
        return
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, workspace=Path.cwd(), credentials=endpoint.credentials) as client:
            job = client.submit(
                name=NAME,
                entrypoint=Entrypoint.from_callable(run_host),
                resources=convert_resources(resource),
                constraints=convert_constraints(resource),
                replicas=resource.replicas,
                coscheduling=resolve_coscheduling(resource, resource.replicas),
                ports=["http", *RAY_PORTS],
                environment=EnvironmentSpec(extras=("tpu",)),
                timeout=Duration.from_seconds(RUNTIME_SECONDS),
                scheduling_timeout=Duration.from_seconds(14400),
                max_retries_failure=0,
                max_retries_preemption=0,
                max_task_failures=0,
            )
            print(job, flush=True)


if __name__ == "__main__":
    main()
