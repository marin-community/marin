"""Bounded Step12 MATH500 using the task260 multi-host serving path; dry-run by default."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from fray.types import EnvironmentConfig, ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient, iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import Entrypoint, EnvironmentSpec
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

HERE = Path("experiments/weight_merging/rsi/parity_261")
MODEL = "gs://marin-us-central2/scratch/karan/rsi-merge/parents/S--open-athena_Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02@50fe2f488aee/"
REPO = "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02"
REVISION = "50fe2f488aee7730bbccc721024b35229075602e"
VLLM = "vllm @ git+https://github.com/marin-community/vllm.git@fb02daf1d7139d2adaeb0588448649591657d1e3"
TPU = "tpu-inference @ git+https://github.com/marin-community/tpu-inference.git@ae387b8bb7a644bbe7e17a9ce11bd701a012aa8d"
EVALCHEMY = "3bb8278078e7ad4b5a6b1f81e8a08db3dd206ae6"
GLOBALS = {3, 7, 11, 15, 19, 23, 25}
LAYERS = ["full_attention" if i in GLOBALS else "sliding_attention" for i in range(26)]
OUTPUT = "gs://marin-us-central2/scratch/karan/task261/step12-r3/"
NAME = "rsi261-step12-v4-32-r3"
RUNTIME_SECONDS = 4 * 3600


def resources():
    return ResourceConfig.with_tpu(
        "v4-32", cpu=32, ram="400g", disk="80g", regions=["us-central2"], zone="us-central2-b", preemptible=False
    )


def save(name, value):
    (StoragePath(OUTPUT) / name).write_text(json.dumps(value, indent=2, default=str))


def slice_preflight():
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError
    from iris.cluster.worker.env_probe import _probe_tpu_metadata

    info, ctx = get_job_info(), iris_ctx()
    slice_id, variant, _, worker_id, _ = _probe_tpu_metadata()
    if not slice_id or not worker_id or variant != "v4-32":
        raise RuntimeError("Invalid v4-32 identity")
    name = f"{NAME}-identity-"
    registration = ctx.registry.register(
        name + str(info.task_index),
        f"{info.advertise_host}:{info.ports['ray-gcs']}",
        metadata={"slice": slice_id, "worker": worker_id},
    )
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        entries = []
        for rank in range(4):
            try:
                result = ctx.resolver.resolve(name + str(rank))
                if not result.is_empty:
                    entries.append(dict(result.first().metadata))
            except ConnectError as exc:
                if exc.code not in (Code.NOT_FOUND, Code.UNAVAILABLE):
                    raise
        if len(entries) == 4:
            if {entry["slice"] for entry in entries} != {slice_id} or len({entry["worker"] for entry in entries}) != 4:
                raise RuntimeError(f"Gang crosses slice boundaries: {entries}")
            save(f"host-{info.task_index}/identity.json", entries)
            return registration
        time.sleep(3)
    raise TimeoutError("Slice identity preflight timed out")


def run_evaluation(base_url, remaining_seconds):
    from marin.evaluation.evalchemy.runner import EvalchemyRunConfig, _run_config_json
    from marin.evaluation.evaluation_config import EvalTaskConfig
    from marin.inference.types import OpenAIEndpoint, RunningModel

    config = EvalchemyRunConfig(
        name="task261-step12-math500",
        tasks=(EvalTaskConfig("MATH500", 0, task_alias="math500", generation=True),),
        apply_chat_template=True,
        seed=19311,
        num_concurrent=8,
        max_length=32704,
    )
    payload = json.loads(
        _run_config_json(RunningModel(OpenAIEndpoint(base_url, REPO), tokenizer=REPO), config, OUTPUT + "math500/")
    )
    source = HERE / "historical_client.py"
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "eefd4118bb5c93cd98488904a1a13cb6025faad6718e7526356f043b2bc8e93c"
    )
    command = [
        "uvx",
        "--no-config",
        "--python",
        "3.12",
        "--from",
        f"evalchemy[serve-eval,math500] @ git+https://github.com/marin-community/evalchemy.git@{EVALCHEMY}",
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
    save(
        "eval-settings.json",
        dict(
            config={**payload, "base_url": "<local server omitted>"},
            command=command,
            revision=REVISION,
            server_context=73728,
            reference_server_context=32768,
            effective_eval_max_length=32704,
            client_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ),
    )
    env = dict(
        os.environ,
        EVALCHEMY_CLIENT_CONFIG=json.dumps(payload),
        JAX_PLATFORMS="cpu",
        CUDA_VISIBLE_DEVICES="",
        OPENAI_API_KEY="local-endpoint",
        HF_ALLOW_CODE_EVAL="1",
        TQDM_MININTERVAL="30",
    )
    for key in ("PYTHONPATH", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    start = time.time()
    log = Path(env["TMPDIR"]) / "eval.log"
    with log.open("w") as stream:
        try:
            result = subprocess.run(
                [*command, "python", str(source)],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=remaining_seconds,
                check=False,
            )
        finally:
            (StoragePath(OUTPUT) / "eval.log").write_bytes(log.read_bytes())
    save("eval-process.json", dict(start_epoch=start, elapsed_seconds=time.time() - start, returncode=result.returncode))
    if result.returncode:
        raise RuntimeError(f"Evaluator exit {result.returncode}; inspect archived eval.log")
    from finestore.reader import ReadView

    table = ReadView(OUTPUT + "math500/").scan("evalchemy_results")
    if table is None:
        raise RuntimeError("No committed aggregate result; client exit status is insufficient")
    rows = table.to_pylist(maps_as_pydicts="strict")
    save("eval-results.json", rows)
    print("TASK261_RESULT", json.dumps(rows, default=str), flush=True)
    if not any(row.get("scored") is True for row in rows):
        raise RuntimeError("Aggregate result is not scored; inspect task status and coverage")


def throughput_probe(base_url):
    import requests

    def one(_):
        start = time.monotonic()
        response = requests.post(
            base_url + "/completions",
            json=dict(
                model=REPO,
                prompt="Continue listing integers in ascending order: 1, 2, 3,",
                max_tokens=128,
                min_tokens=128,
                ignore_eos=True,
                temperature=0,
            ),
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        return dict(
            seconds=time.monotonic() - start, usage=data["usage"], finish_reason=data["choices"][0]["finish_reason"]
        )

    one(0)  # Warm the same fixed-output request before timing.
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, range(16)))
    wall = time.monotonic() - start
    save(
        "warm-throughput.json",
        dict(
            concurrency=8,
            requests=16,
            output_tokens=sum(row["usage"]["completion_tokens"] for row in rows),
            wall_seconds=wall,
            tokens_per_second=sum(row["usage"]["completion_tokens"] for row in rows) / wall,
            rows=rows,
            note="Fixed-output warm microbenchmark, separate from MATH500 generation throughput.",
        ),
    )


def build_service():
    from marin.inference import iris as serving
    from marin.inference.config import IrisConfig, ServedModelConfig, VllmEngineConfig, VllmLauncherType

    return serving.IrisServiceConfig(
        model=ServedModelConfig(
            weights=MODEL,
            api_model=REPO,
            tokenizer=MODEL,
            max_model_len=73728,
            tensor_parallel_size=16,
            chat_template_content=(HERE / "S-chat-template.jinja").read_text(),
        ),
        engine=VllmEngineConfig(
            launcher=VllmLauncherType.TPU,
            startup_timeout_seconds=3600,
            max_num_batched_tokens=512,
            max_num_seqs=8,
            extra_args=(
                "--trust-remote-code",
                "--load-format",
                "runai_streamer",
                "--hf-overrides",
                json.dumps({"layer_types": LAYERS}),
            ),
        ),
        iris=IrisConfig(
            worker_resources=resources(),
            worker_environment=EnvironmentConfig(workspace=str(Path.cwd()), extras=("tpu",)),
            max_retries_failure=0,
            max_retries_preemption=0,
        ),
        endpoint_name="/serve/" + NAME,
        timeout_hours=4,
    )


def run_host():
    from marin.inference import iris as serving
    from marin.inference import vllm_backend
    from marin.inference.vllm_server import IsolatedTpuVllm

    started = time.time()
    info = get_job_info()
    root = Path(tempfile.mkdtemp(prefix=".task261-", dir=Path.cwd()))
    (root / "tmp").mkdir()
    os.environ.update(
        UV_CACHE_DIR=str(root / "uv-cache"),
        TMPDIR=str(root / "tmp"),
        RAY_DEDUP_LOGS="0",
        HF_HUB_DISABLE_XET="1",
        TPU_MULTIHOST_BACKEND="ray",
    )
    registration = slice_preflight()
    original_command = IsolatedTpuVllm.command
    IsolatedTpuVllm.command = lambda self: [
        *original_command(self)[:-1],
        "python",
        str((HERE / "bootstrap.py").resolve()),
        "vllm",
    ]
    vllm_backend.vllm_launcher = lambda _: IsolatedTpuVllm(vllm_ref=VLLM, tpu_inference_ref=TPU, python_version="3.12")
    original_start = vllm_backend.VllmBackend.start
    holder = {}

    @contextlib.contextmanager
    def capture(self, *args, **kwargs):
        with original_start(self, *args, **kwargs) as environment:
            holder["environment"] = environment
            try:
                yield environment
            finally:
                (StoragePath(OUTPUT) / "server.log").write_text(environment.logs())

    vllm_backend.VllmBackend.start = capture

    def evaluate(check_alive, timeout_hours):
        import requests

        environment = holder["environment"]
        logs = environment.logs()
        matches = re.findall(
            r"RSI236_LAYER_STATIC\s+model\.layers\.(\d+)\.self_attn\.attn\s+sliding_window=(None|2048)", logs
        )
        observed = {int(index): window for index, window in matches}
        expected = {i: "None" if i in GLOBALS else "2048" for i in range(26)}
        gate = dict(
            observed=observed,
            expected=expected,
            kv_groups_4=bool(re.search(r"num_kv_cache_groups\s*=\s*4", logs)),
            ready_seconds=time.time() - started,
        )
        save("gate.json", gate)
        if observed != expected or not gate["kv_groups_4"]:
            raise RuntimeError("Layer/KV gate failed; no evaluation submitted")
        base = environment.server_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        response = requests.get(base + "/models", timeout=30)
        response.raise_for_status()
        save("models.json", response.json())
        print("TASK261_READY", json.dumps(gate), flush=True)
        check_alive()
        throughput_probe(base)
        metrics_url = base.removesuffix("/v1") + "/metrics"
        before = requests.get(metrics_url, timeout=30)
        (StoragePath(OUTPUT) / "metrics-before.txt").write_text(before.text)
        run_evaluation(base, max(1, RUNTIME_SECONDS - (time.time() - started) - 180))
        after = requests.get(metrics_url, timeout=30)
        (StoragePath(OUTPUT) / "metrics-after.txt").write_text(after.text)
        check_alive()

    serving._block_until_timeout = evaluate
    service = build_service()
    try:
        serving.run_iris_service(service)
    except BaseException as exc:
        save(f"host-{info.task_index}/error.json", dict(type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        save(f"host-{info.task_index}/wall.json", dict(start_epoch=started, elapsed_seconds=time.time() - started))
        iris_ctx().registry.unregister(registration)


def main():
    from marin.inference.iris_tpu_vllm import RAY_PORTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    resource = resources()
    build_service()  # Validate the complete service config before allocating.
    plan = dict(
        name=NAME,
        output=OUTPUT,
        model=MODEL,
        TP=16,
        PP=1,
        replicas=resource.replicas,
        server_context=73728,
        eval_context=32704,
        runtime_hours=4,
        queue_hours=4,
        ports=["http", *RAY_PORTS],
        coscheduling=str(resolve_coscheduling(resource, resource.replicas)),
        evalchemy=EVALCHEMY,
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
                ports=plan["ports"],
                environment=EnvironmentSpec(extras=("tpu",)),
                timeout=Duration.from_seconds(14400),
                scheduling_timeout=Duration.from_seconds(14400),
                max_retries_failure=0,
                max_retries_preemption=0,
                max_task_failures=0,
            )
            print(job, flush=True)


if __name__ == "__main__":
    main()
