"""Task289 merge screening: serve one model on a reserved v4-32 and run four policy evals twice against it.

Reuses the task261 multi-host launcher (PR #10535 branch) and the task249/261 policy YAMLs, Evalchemy pin and
client. Dry-run by default; `--submit` allocates one v4-32.
"""

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
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.constraints import Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec
from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.evalchemy.runner import _run_config_json
from marin.inference.iris_tpu_vllm import RAY_PORTS
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.evaluation.evals import evalchemy_run_config
from experiments.weight_merging.rsi.parity_261 import launch as core

HERE = Path("experiments/weight_merging/rsi/screen_289")
ROOT = "gs://marin-us-central2/scratch/karan/rsi-merge/"
MODELS = {
    "S": ROOT + "parents/S--open-athena_Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02@50fe2f488aee/",
    "C": ROOT + "parents/C--open-athena_Grug-67B-A2B-Merge-Campaign3-DARE-Linear-02-2026.10.12@271c4b3c6b38/",
    "avg-SC": ROOT + "merges/avg-SC/",
    "ta-S-C-A5": ROOT + "merges/ta-S-C-A5/",
    "avg-SM": ROOT + "merges/avg-SM/",
}
# All five checkpoints share S's tokenizer.json (md5 checked); the client resolves it from the public S repo.
TOKENIZER = "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02"
COMMIT = "c1d5c62cdc3939b42f64411af44fd50716b53454"
KEYS = ("math500", "gsm8k-0shot", "humanevalplus", "ifbench")
RUNS = (1, 2)
RUNTIME_SECONDS = 6 * 3600
# v4-32: reserved, 4 hosts, TP16 (task261 parity). v5p-8: preemptible, 1 host, TP4 (task249 OlympiadBench parity);
# preemptions retry the whole job, evals finished before it are kept and Evalchemy resumes the interrupted one.
PLATFORMS = {
    "v4-32": dict(tp=16, multihost=True, preemption_retries=0),
    "v5p-8": dict(tp=4, multihost=False, preemption_retries=30),
}
# Slice t1v-n-70779c29 hosts the #205 CPU merges (2026-10-23); keep the serve off it.
AVOID_TPU_NAMES = ("t1v-n-70779c29",)


def policy_plan(label, key, output, base_url="http://resolved-worker-local/v1"):
    path = HERE / "policy_configs" / f"{key}.yaml"
    source = load_evalchemy_config(path)
    config = dataclasses.replace(
        evalchemy_run_config(key, source), apply_chat_template=True, num_concurrent=16, max_length=73664
    )
    if any(not task.generation for task in config.tasks) and (
        config.max_gen_toks is not None or config.extra_gen_kwargs
    ):
        raise ValueError("Loglikelihood tasks must not carry generation overrides")
    payload = json.loads(
        _run_config_json(RunningModel(OpenAIEndpoint(base_url, label), tokenizer=TOKENIZER), config, output)
    )
    return dict(
        config=payload,
        extras=sorted({"serve-eval", *source.runtime_extras}),
        policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        yaml=path.read_text(),
    )


def evalchemy_command(extras):
    command = [
        "uvx",
        "--no-config",
        "--python",
        "3.12",
        "--from",
        f"evalchemy[{','.join(extras)}] @ git+https://github.com/marin-community/evalchemy.git@{COMMIT}",
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
    return command


def run_one(label, platform, key, run, base_url, deadline, client_sha):
    root = StoragePath(core.OUTPUT) / key / f"r{run}"
    try:
        rows = json.loads((root / "results.json").read_text())
    except FileNotFoundError:
        rows = None
    if rows and all(row.get("scored") is True for row in rows):
        return dict(eval=key, run=run, kept_from_earlier_attempt=True)
    plan = policy_plan(label, key, str(root) + "/", base_url)
    payload = plan["config"]
    (root / "settings.json").write_text(
        json.dumps(
            {
                **plan,
                "config": {**payload, "base_url": "<worker-local omitted>"},
                "model": label,
                "platform": platform,
                "tensor_parallel_size": PLATFORMS[platform]["tp"],
                "weights": core.MODEL,
                "evalchemy_commit": COMMIT,
                "client_sha256": client_sha,
                "server_context": 73728,
                "max_num_seqs": 16,
                "run": run,
                "settings_note": "Policy YAML verbatim; server-default sampling (Evalchemy strips temperature/seed).",
            },
            indent=2,
        )
    )
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
    log = Path(env["TMPDIR"]) / f"eval-{key}-r{run}.log"
    source = core.HERE / "policy_client.py"
    started = time.time()
    try:
        with log.open("w") as stream:
            result = subprocess.run(
                [*evalchemy_command(plan["extras"]), "python", str(source.resolve())],
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
        json.dumps(dict(started_epoch=started, elapsed_seconds=time.time() - started, returncode=result.returncode))
    )
    if result.returncode:
        raise RuntimeError(f"{key} r{run} client exited {result.returncode}; inspect archived log")
    table = ReadView(str(root)).scan("evalchemy_results")
    if table is None:
        raise RuntimeError(f"{key} r{run}: no aggregate result")
    rows = table.to_pylist(maps_as_pydicts="strict")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    if not rows or not all(row.get("scored") is True for row in rows):
        raise RuntimeError(f"{key} r{run}: unscored aggregate result")
    documents = [json.loads(row["document"]) for row in rows]
    return dict(
        eval=key,
        run=run,
        wall_seconds=time.time() - started,
        results=[doc.get("results") for doc in documents],
        outcomes=[doc.get("task_outcomes") for doc in documents],
    )


def make_suite(label, platform):
    def run_suite(base_url, remaining_seconds):
        deadline = time.monotonic() + remaining_seconds
        client_sha = hashlib.sha256((core.HERE / "policy_client.py").read_bytes()).hexdigest()
        summary, failures = [], []
        # Run order r1 of every eval, then r2, so a late failure still leaves one full pass.
        for run in RUNS:
            for key in KEYS:
                try:
                    record = run_one(label, platform, key, run, base_url, deadline, client_sha)
                except Exception as exc:  # Keep going: one failed eval should not discard the others.
                    record = dict(eval=key, run=run, error=f"{type(exc).__name__}: {exc}")
                    failures.append(record)
                summary.append(record)
                (StoragePath(core.OUTPUT) / "suite-summary.json").write_text(json.dumps(summary, indent=2))
                print("TASK289_RESULT", json.dumps(record), flush=True)
        if failures:
            raise RuntimeError(f"{len(failures)} eval runs failed: {failures}")

    return run_suite


def names(label, platform, attempt):
    # The first v4-32 attempts used .../task289/<label>-r1/; keep that path for v4-32.
    suffix = "" if platform == "v4-32" else f"-{platform}"
    return (
        f"rsi289-{label}-{platform}-r{attempt}",
        f"gs://marin-us-central2/scratch/karan/task289/{label}{suffix}-r{attempt}/",
    )


def configure(label, platform, attempt):
    """Point the task261 core launcher at one model and platform (runs on submit host and on workers)."""
    name, output = names(label, platform, attempt)
    spec = PLATFORMS[platform]
    core.MODEL, core.REPO, core.REVISION, core.OUTPUT, core.NAME = MODELS[label], label, label, output, name
    core.RUNTIME_SECONDS = RUNTIME_SECONDS
    core.MULTIHOST = spec["multihost"]
    if platform != "v4-32":
        core.resources = lambda: ResourceConfig.with_tpu(platform, cpu=32, ram="400g", disk="80g", preemptible=True)
    original_build = core.build_service

    def build():
        service = original_build()
        return dataclasses.replace(
            service,
            model=dataclasses.replace(service.model, tensor_parallel_size=spec["tp"]),
            engine=dataclasses.replace(service.engine, max_num_seqs=16),
            timeout_hours=6,
        )

    core.build_service = build


def make_host(label, platform, attempt):

    def run_host():
        # Bind worker-local settings inside the serialized entrypoint (see task261 r1 failure).
        configure(label, platform, attempt)
        core.run_evaluation = make_suite(label, platform)
        core.run_host()

    return run_host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--platform", choices=sorted(PLATFORMS), default="v4-32")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    name, output = names(args.model, args.platform, args.attempt)
    configure(args.model, args.platform, args.attempt)
    spec = PLATFORMS[args.platform]
    resource = core.resources()
    service = core.build_service()  # Validate full service types before allocation.
    constraints = convert_constraints(resource)
    if spec["multihost"]:
        constraints += [Constraint.create(key="tpu-name", op=ConstraintOp.NE, value=v) for v in AVOID_TPU_NAMES]
    ports = ["http", *RAY_PORTS] if spec["multihost"] else ["http"]
    plan = dict(
        name=name,
        model=MODELS[args.model],
        output=output,
        platform=args.platform,
        TP=service.model.tensor_parallel_size,
        hosts=resource.replicas,
        preemption_retries=spec["preemption_retries"],
        context=73728,
        max_num_seqs=16,
        runtime_hours=6,
        queue_hours=8,
        constraints=[str(c) for c in constraints],
        coscheduling=str(resolve_coscheduling(resource, resource.replicas)),
        policies={key: policy_plan(args.model, key, output + key + "/r1/")["config"] for key in KEYS},
    )
    print(json.dumps(plan, indent=2))
    if not args.submit:
        return
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, workspace=Path.cwd(), credentials=endpoint.credentials) as client:
            job = client.submit(
                name=name,
                entrypoint=Entrypoint.from_callable(make_host(args.model, args.platform, args.attempt)),
                resources=convert_resources(resource),
                constraints=constraints,
                replicas=resource.replicas,
                coscheduling=resolve_coscheduling(resource, resource.replicas) if spec["multihost"] else None,
                ports=ports,
                environment=EnvironmentSpec(extras=("tpu",)),
                timeout=Duration.from_seconds(RUNTIME_SECONDS),
                scheduling_timeout=Duration.from_seconds(8 * 3600),
                max_retries_failure=0,
                max_retries_preemption=spec["preemption_retries"],
                max_task_failures=0,
            )
            print(job, flush=True)


if __name__ == "__main__":
    main()
