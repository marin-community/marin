# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own one bounded difficulty solver campaign and its Iris serving lifetime."""

import argparse
import hashlib
import json
import logging
import os
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

from finestore.rl.difficulty import PROTOCOL_ID, PROTOCOLS, SOLVER_MODELS, SamplingSettings, Solver
from huggingface_hub import hf_hub_download
from iris.rpc import job_pb2
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.serving_config import inference_config_for_model
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import VllmEngineConfig
from marin.inference.iris import remote_inference
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import SECRETS_DIR_ENV, forwarded_secrets, require_secret
from rigging.timing import Duration
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url
from experiments.post_training.task_curation.difficulty_config import CandidateRun, JudgeRun
from experiments.post_training.task_curation.difficulty_evidence import write_json_evidence
from experiments.post_training.task_curation.difficulty_preparation import prepare_source
from experiments.post_training.task_curation.difficulty_protocol import (
    GLM_RELAY_JOB,
    PINS,
    TARGET_CLUSTER,
    serving_model,
)
from experiments.post_training.task_curation.difficulty_provider import ProviderLimits
from experiments.post_training.task_curation.difficulty_samples import DifficultySource
from experiments.post_training.task_curation.difficulty_sampling import request_parameters, sampling_settings
from experiments.post_training.task_curation.difficulty_serving_probe import probe_endpoint

LOGGER = logging.getLogger(__name__)
PRIORITY = job_pb2.PRIORITY_BAND_PRODUCTION
MODEL_CACHE_ROOT = "s3://marin-rno2a"
JOB_TIMEOUT = Duration.from_seconds(259200)
NATIVE_CONTEXT_SAFETY_BUFFER = 0


def worker_limits(root: Path) -> dict:
    """Delegate CPU and memory only beneath this disposable Iris container's scope."""
    os.environ["DEBIAN_FRONTEND"] = "noninteractive"
    os.environ["PATH"] = "/usr/sbin:/sbin:" + os.environ["PATH"]
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(
        [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "bubblewrap",
            "e2fsprogs",
            "util-linux",
            "umoci",
            "skopeo",
        ],
        check=True,
    )
    membership = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    parent = (Path("/sys/fs/cgroup") / membership.removeprefix("0::").lstrip("/")).resolve()
    if not parent.name.startswith("cri-containerd-") or not parent.name.endswith(".scope"):
        raise RuntimeError("Refuse cgroup delegation outside this disposable container's own scope")
    fields = ("cgroup.type", "cgroup.controllers", "cgroup.subtree_control", "cpu.max", "memory.max")
    before = {name: (parent / name).read_text().strip() for name in fields}
    if before["cgroup.type"] != "domain" or not {"cpu", "memory"} <= set(before["cgroup.controllers"].split()):
        raise RuntimeError("Worker has no CPU/memory cgroup v2 domain")
    driver = parent / "atlas-difficulty-driver"
    driver.mkdir()
    for _ in range(5):
        pids = (parent / "cgroup.procs").read_text().split()
        if not pids:
            break
        for pid in pids:
            (driver / "cgroup.procs").write_text(pid)
    if (parent / "cgroup.procs").read_text().strip():
        raise RuntimeError("Container scope retains internal processes")
    (parent / "cgroup.subtree_control").write_text("+cpu +memory")
    (root / "worker-cgroup.json").write_text(
        json.dumps(
            {
                "parent": str(parent),
                "driver": str(driver),
                "before": before,
                "after": {name: (parent / name).read_text().strip() for name in fields},
            }
        )
        + "\n"
    )
    return {"cgroup_parent": str(parent), "disk_root": str(root / "sandbox-disks")}


@contextmanager
def solver_endpoint(solver: Solver, settings: SamplingSettings):
    if solver == Solver.HOSTED:
        yield resolve_glm_base_url(GLM_RELAY_JOB), GLM_MODEL, require_secret(GLM_BULK_TOKEN_ENV), []
        return
    definition = serving_model(solver, settings.context_tokens)
    hardware = AcceleratorChoice(
        Platform.GPU, gpu_type="H100", gpu_count=PINS[solver].gpus, target_cluster=TARGET_CLUSTER
    )
    config = inference_config_for_model(
        definition,
        hardware,
        env_vars={"MARIN_TEMP_PREFIX": MODEL_CACHE_ROOT},
        priority=PRIORITY,
        api_model=SOLVER_MODELS[solver],
    )
    if not isinstance(config.engine, VllmEngineConfig):
        raise ValueError("Difficulty local solvers require the vLLM engine")
    config = replace(
        config,
        engine=replace(config.engine, startup_timeout_seconds=259200),
        iris=replace(
            config.iris,
            worker_timeout=JOB_TIMEOUT,
            # The explicitly targeted root has already federated. Its whole job tree
            # stays on that peer; another cluster directive would request a second handoff.
            worker_resources=replace(config.iris.worker_resources, target_cluster=None, regions=None),
            endpoint_ready_timeout_seconds=259200,
        ),
    )
    with remote_inference(config) as session:
        ready = session.ready_direct_session()
        if ready.effective_serving is None or ready.effective_serving.max_model_len < settings.context_tokens:
            raise ValueError("Running inference does not confirm the required context window")
        endpoint = ready.model.endpoint
        yield endpoint.base_url, endpoint.model, endpoint.api_key or "EMPTY", [str(job.job_id) for job in session.jobs]


def run_trial_worker(
    prepared_path: Path,
    config_path: Path,
    local_root: Path,
    output: StoragePath,
    api_key: str,
    recovery_arguments: list[str],
    judge: JudgeRun | None,
) -> None:
    """Launch the locked Harbor runtime with private host-side secret files."""
    secrets_dir = local_root / "framework-secrets"
    secrets_dir.mkdir(mode=0o700)
    secrets = {**forwarded_secrets(), "ATLAS_DIFFICULTY_CANDIDATE_TOKEN": api_key}
    if judge is not None:
        secrets[judge.api_key_secret] = require_secret(judge.api_key_secret)
    for name, value in secrets.items():
        path = secrets_dir / name
        path.write_text(value)
        path.chmod(0o600)
    subprocess.run(
        [
            "uv",
            "run",
            "--project",
            "vendor/harbor",
            "--frozen",
            "--extra",
            "runtime",
            "python",
            "-m",
            "experiments.post_training.task_curation.difficulty_trials",
            "--sources",
            str(prepared_path),
            "--config",
            str(config_path),
            "--output-root",
            str(output / "trials"),
            "--local-root",
            str(local_root / "trials"),
            *recovery_arguments,
        ],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path.cwd()),
            SECRETS_DIR_ENV: str(secrets_dir),
            "HARBOR_VLLM_CONTEXT_SAFETY_BUFFER": str(NATIVE_CONTEXT_SAFETY_BUFFER),
        },
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Solver, choices=list(Solver), required=True)
    parser.add_argument("--previous-campaign")
    parser.add_argument("--previous-serving-sha256")
    parser.add_argument("--sources", required=True)
    parser.add_argument("--sources-sha256", required=True)
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--attempts-per-task", type=int, required=True)
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--registry-auth-secret")
    args = parser.parse_args()
    if bool(args.previous_campaign) != bool(args.previous_serving_sha256):
        raise ValueError("Continuation requires both the previous campaign and its serving receipt digest")
    configure_coreweave_s3()
    output = StoragePath(args.output_root)
    if args.local_root.exists() or output.exists():
        raise FileExistsError("Preserve prior campaigns and select a new run version")
    source_path = StoragePath(args.sources)
    if file_sha256(source_path) != args.sources_sha256:
        raise ValueError("Source queue differs from the frozen digest")
    sources = [DifficultySource.model_validate(row) for row in json.loads(source_path.read_text())]
    if args.source_id:
        sources = [source for source in sources if source.declaration.id in args.source_id]
        if {source.declaration.id for source in sources} != set(args.source_id):
            raise ValueError("Requested source is absent from the frozen selection")
    pin = PINS[args.solver]
    args.local_root.mkdir(parents=True)
    options = worker_limits(args.local_root)
    if args.registry_auth_secret is not None:
        authfile = args.local_root / "registry-auth.json"
        authfile.write_text(require_secret(args.registry_auth_secret))
        authfile.chmod(0o600)
        options["registry_authfile"] = str(authfile)
    prepared = [prepare_source(source, args.local_root / "prepared") for source in sources]
    has_judge = any(task.judge_model is not None and task.reason is None for source in prepared for task in source.tasks)
    for selected_solver in {args.solver, Solver.HOSTED} if has_judge else {args.solver}:
        selected_pin = PINS[selected_solver]
        template = Path(
            hf_hub_download(SOLVER_MODELS[selected_solver], "chat_template.jinja", revision=selected_pin.revision)
        )
        if hashlib.sha256(template.read_bytes()).hexdigest() != selected_pin.template_sha256:
            raise ValueError("Pinned chat template differs from the resolved model identity")
    prepared_path = args.local_root / "prepared-sources.json"
    prepared_path.write_text(json.dumps([source.model_dump(mode="json") for source in prepared]) + "\n")
    write_json_evidence(
        output / "prepared.json",
        {
            "sources": [source.model_dump(mode="json") for source in prepared],
            "source_queue": args.sources,
            "source_queue_sha256": args.sources_sha256,
            "worker_limits": json.loads((args.local_root / "worker-cgroup.json").read_text()),
        },
    )
    deviations = (
        ["Hosted weight revision is opaque; tokenizer uses the production config pin. Server fingerprint is archived."]
        if args.solver == Solver.HOSTED
        else []
    )
    settings = sampling_settings(args.solver, deviations)
    provider_limits = ProviderLimits(
        max_in_flight=4 if args.solver == Solver.HOSTED else args.workers,
        attempts=6 if args.solver == Solver.HOSTED else 1,
    )
    judge_limits = ProviderLimits(max_in_flight=1, attempts=6) if has_judge else None
    with solver_endpoint(args.solver, settings) as (endpoint, api_model, api_key, jobs):
        probe = probe_endpoint(args.solver, endpoint, api_model, api_key, settings)
        judge_pin = PINS[Solver.HOSTED]
        judge_metadata = None
        judge_endpoint = None
        if has_judge:
            judge_endpoint = resolve_glm_base_url(GLM_RELAY_JOB)
            judge_probe = (
                probe
                if args.solver == Solver.HOSTED
                else probe_endpoint(
                    Solver.HOSTED,
                    judge_endpoint,
                    GLM_MODEL,
                    require_secret(GLM_BULK_TOKEN_ENV),
                    sampling_settings(Solver.HOSTED, []),
                )
            )
            judge_metadata = {
                "model": SOLVER_MODELS[Solver.HOSTED],
                "api_model": GLM_MODEL,
                "revision": f"unexposed:{judge_probe['tools_thinking_probe']['completion']['system_fingerprint']}",
                "tokenizer": SOLVER_MODELS[Solver.HOSTED],
                "tokenizer_revision": judge_pin.revision,
                "chat_template_sha256": judge_pin.template_sha256,
                "effective_temperature": PROTOCOLS[PROTOCOL_ID].temperature,
                "token_budget": "original source request; production service enforces its served context",
                "basis": "Host override; original TaskSpec and requested judge parameters are retained.",
                "serving_probe": judge_probe,
            }
        completion = probe["tools_thinking_probe"]["completion"]
        revision = f"unexposed:{completion['system_fingerprint']}" if args.solver == Solver.HOSTED else pin.revision
        receipt = write_json_evidence(
            output / "serving.json",
            {
                "protocol": PROTOCOL_ID,
                "solver": args.solver.value,
                "model": SOLVER_MODELS[args.solver],
                "requested_revision": pin.revision,
                "tokenizer_revision": pin.revision,
                "chat_template_sha256": pin.template_sha256,
                "model_definition": (
                    None if args.solver == Solver.HOSTED else asdict(serving_model(args.solver, settings.context_tokens))
                ),
                "serving_jobs": jobs,
                "placement": {
                    "root_target_cluster": TARGET_CLUSTER,
                    "child_region_constraint": None,
                    "model_cache_root": MODEL_CACHE_ROOT,
                    "basis": "The explicitly targeted coordinator and its entire child job tree share one peer.",
                },
                "sampling": request_parameters(settings),
                "provider_limits": asdict(provider_limits),
                "judge_provider_limits": None if judge_limits is None else asdict(judge_limits),
                "judge": judge_metadata,
                "native_client_context_safety_buffer_tokens": NATIVE_CONTEXT_SAFETY_BUFFER,
                "compilation_and_cuda_graphs_requested": True,
                "vllm_gpu_release": None if args.solver == Solver.HOSTED else asdict(VLLM_GPU_RELEASE),
                **probe,
            },
        )
        config = {
            "solver": args.solver,
            "identity": {
                "model": SOLVER_MODELS[args.solver],
                "revision": revision,
                "tokenizer": SOLVER_MODELS[args.solver],
                "tokenizer_revision": pin.revision,
                "chat_template_sha256": pin.template_sha256,
                "serving_receipt": receipt.model_dump(mode="json"),
            },
            "settings": settings.model_dump(mode="json"),
            "endpoint": endpoint,
            "api_model": api_model,
            "api_key_secret": "ATLAS_DIFFICULTY_CANDIDATE_TOKEN",
            "provider_limits": asdict(provider_limits),
            "environment_options": options,
            "workers": args.workers,
            "gpu_count": pin.gpus,
            "attempts_per_task": args.attempts_per_task,
            "task_limit": args.task_limit,
            "judge": (
                None
                if judge_metadata is None
                else {
                    "endpoint": judge_endpoint,
                    "api_model": GLM_MODEL,
                    "api_key_secret": GLM_BULK_TOKEN_ENV,
                    "provider_limits": asdict(judge_limits),
                    "identity": {
                        **{
                            key: judge_metadata[key]
                            for key in ("model", "revision", "tokenizer", "tokenizer_revision", "chat_template_sha256")
                        },
                        "serving_receipt": receipt.model_dump(mode="json"),
                    },
                }
            ),
        }
        config_path = args.local_root / "candidate.json"
        candidate = CandidateRun.model_validate(config)
        config_path.write_text(candidate.model_dump_json() + "\n")
        recovery_arguments = []
        if args.previous_campaign is not None:
            recovery_arguments = [
                "--previous-campaign",
                args.previous_campaign,
                "--previous-serving-sha256",
                args.previous_serving_sha256,
                "--source-queue-sha256",
                args.sources_sha256,
            ]
        run_trial_worker(
            prepared_path, config_path, args.local_root, output, api_key, recovery_arguments, candidate.judge
        )
        write_json_evidence(output / "completion.json", {"solver": args.solver, "serving_jobs": jobs, "worker_exit": 0})


if __name__ == "__main__":
    main()
