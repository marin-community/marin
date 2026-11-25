# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve one immutable reconciler batch with the established solver and judge protocol."""

import argparse
import hashlib
import json
import logging
import os
import subprocess
from collections.abc import Iterable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType

import httpx
from finestore.rl.difficulty import PROTOCOL_ID, PROTOCOLS, SOLVER_MODELS, SamplingSettings, Solver
from huggingface_hub import hf_hub_download
from iris.client.client import iris_ctx
from iris.rpc.proto_display import priority_band_value
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.serving_config import inference_config_for_model
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import VllmEngineConfig
from marin.inference.iris import remote_inference
from marin.inference.types import OpenAIEndpoint
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import SECRETS_DIR_ENV, forwarded_secrets, require_secret, secret_path
from taskcompendium.pipeline.reviewer_spec import JudgeTier
from transformers import AutoTokenizer

from experiments.post_training.glm import GLM_MODEL
from experiments.post_training.task_curation.difficulty.campaign import JUDGE_VERSION, TPU_SERVING_ID, campaign_record
from experiments.post_training.task_curation.difficulty.launch import JOB_TIMEOUT
from experiments.post_training.task_curation.difficulty.ledger import record_action_failure
from experiments.post_training.task_curation.difficulty.state import ActionFailure, GapReason
from experiments.post_training.task_curation.difficulty_config import (
    DATA_ACCOUNT,
    DECLARED_JUDGE_PINS,
    SELF_HOSTED_SOLVERS,
    SUBSTITUTE_JUDGE_MODEL,
    SUBSTITUTE_JUDGE_PIN,
    TASK_PROVIDER_SECRETS,
    TOGETHER_API_KEY_ENV,
    CandidateRun,
    JudgeRun,
)
from experiments.post_training.task_curation.difficulty_evidence import write_json_evidence
from experiments.post_training.task_curation.difficulty_preparation import PreparedSource, prepare_source
from experiments.post_training.task_curation.difficulty_prepared import attempt_judge_model
from experiments.post_training.task_curation.difficulty_protocol import (
    PINS,
    TARGET_CLUSTER,
    serving_model,
    solver_vllm_release,
)
from experiments.post_training.task_curation.difficulty_provider import (
    DECLARED_JUDGE_REQUEST_TIMEOUT,
    MODEL_REQUEST_TIMEOUT,
    ProviderLimits,
)
from experiments.post_training.task_curation.difficulty_samples import load_queue_sources
from experiments.post_training.task_curation.difficulty_sampling import request_parameters, sampling_settings
from experiments.post_training.task_curation.difficulty_serving_probe import (
    probe_declared_judge_endpoint,
    probe_endpoint,
    probe_judge_endpoint,
)
from experiments.post_training.task_curation.judges import JUDGE_TIERS, LocalJudgeEndpoint

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_MODEL = "zai-org/GLM-5.3"

LOCAL_IMAGE_CACHE = "/cache/atlas-difficulty/images"
"""Node-local NVMe scratch for pulled actor images.

The container's root filesystem and /tmp are memory-backed on CoreWeave bare
metal: a 55GB image pull charged to them is unreclaimable cgroup memory and
kills the pod. The /cache mount is a hostPath on the node NVMe, so the pull
lands on disk and persists across pods on that node.
"""


def hosted_liveness_response(base_url: str, api_key: str, model: str) -> httpx.Response:
    """Return the provider's response to one bounded liveness request."""
    with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=60) as client:
        return client.post(
            base_url.rstrip("/") + "/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "liveness"}], "max_tokens": 1},
        )


def hosted_endpoint_liveness(base_url: str, api_key: str, model: str) -> None:
    """Fail in seconds when the hosted endpoint is dead, instead of a 30-minute probe timeout.

    One short chat completion with a 60-second timeout separates a live serving
    endpoint from a dead one (a dead relay once sat in the running state with its
    serving pod deleted, and the campaign's serving probe hung for its full window).
    """
    try:
        hosted_liveness_response(base_url, api_key, model).raise_for_status()
    except Exception as error:
        raise RuntimeError(
            f"The hosted endpoint {base_url} (model {model}) did not answer a liveness completion "
            f"within 60s: {type(error).__name__}: {str(error)[:160]}. Check the endpoint and its "
            "credential before resubmitting."
        ) from error


LOGGER = logging.getLogger(__name__)
# The scheduler preempts lower bands; a production campaign evicts interactive
# peers such as the Snowball ablation roots. Every launch declares its band.
PRIORITY_CHOICES = ("production", "interactive", "batch", "system")
MODEL_CACHE_ROOT = "s3://marin-rno2a"
NATIVE_CONTEXT_SAFETY_BUFFER = 0
# A hosted endpoint queues probes behind hosted traffic; retry after its queue drains.
SERVING_PROBE_ATTEMPTS = 3


def solver_pool_record(solver: Solver, registry: str) -> dict:
    """Load a measured pool receipt and verify its frozen solver identity."""
    record = json.loads((StoragePath(registry) / f"{solver.value}.json").read_text())
    if record["model"] != SOLVER_MODELS[solver] or record["revision"] != PINS[solver].revision:
        raise ValueError("Solver pool differs from the protocol's pinned checkpoint")
    return record


def solver_pool_connection(record: dict, routing_key: str = "") -> OpenAIEndpoint:
    names = [record["endpoint_name"]] if record.get("endpoint_name") else record["endpoint_names"]
    selected = int(hashlib.sha256(routing_key.encode()).hexdigest(), 16) % len(names)
    address = iris_ctx().client.resolve_endpoint(names[selected]).rstrip("/")
    return OpenAIEndpoint(base_url=address + "/v1", model=record["model"], api_key="EMPTY")


def probe_with_transient_retries(
    solver: Solver, endpoint: str, api_model: str, api_key: str, settings: SamplingSettings
):
    """Probe a hosted endpoint, retrying the queue-drain timeouts."""
    for attempt_number in range(1, SERVING_PROBE_ATTEMPTS + 1):
        try:
            return probe_endpoint(solver, endpoint, api_model, api_key, settings)
        except httpx.TimeoutException as error:
            if attempt_number == SERVING_PROBE_ATTEMPTS:
                raise
            LOGGER.warning("Serving probe attempt %d timed out: %s", attempt_number, error)
    raise AssertionError("the retry loop always returns or raises on the final attempt")


# Judge tiers Difficulty hosts itself, with the pinned model YAML and its H100 count.
# Every non-GLM judge model a source declares must be one of these tiers.
JUDGE_TIER_SERVING: Mapping[JudgeTier, tuple[Path, int]] = MappingProxyType(
    {
        JudgeTier.LIGHT: (
            Path("experiments/post_training/task_curation/judge_model_configs/openai-gpt-oss-120b-131k.yaml"),
            2,
        ),
        JudgeTier.MEDIUM: (
            Path("experiments/post_training/task_curation/judge_model_configs/MiniMaxAI-MiniMax-M3-MXFP8-262k.yaml"),
            8,
        ),
    }
)


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
    cgroup_root = Path("/sys/fs/cgroup").resolve()
    parent = (cgroup_root / membership.removeprefix("0::").lstrip("/")).resolve()
    containerd_scope = parent.name.startswith("cri-containerd-") and parent.name.endswith(".scope")
    # Iris's GCP Docker workers use a private cgroup namespace: its root is
    # this container's scope, not the host hierarchy. Privileged jobs expose
    # that scope writable. Require the container marker and PID1 membership
    # before accepting this layout; a bare host root must still fail closed.
    private_container_root = (
        parent == cgroup_root
        and membership == "0::/"
        and Path("/.dockerenv").exists()
        and Path("/proc/1/cgroup").read_text().strip() == "0::/"
    )
    if not (containerd_scope or private_container_root):
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
def solver_endpoint(solver: Solver, settings: SamplingSettings, priority: int):
    if solver == Solver.HOSTED:
        base_url = TOGETHER_BASE_URL
        hosted_endpoint_liveness(base_url, require_secret(TOGETHER_API_KEY_ENV), TOGETHER_MODEL)
        yield base_url, TOGETHER_MODEL, require_secret(TOGETHER_API_KEY_ENV), []
        return
    definition = serving_model(solver, settings.context_tokens)
    hardware = AcceleratorChoice(
        Platform.GPU, gpu_type="H100", gpu_count=PINS[solver].gpus, target_cluster=TARGET_CLUSTER
    )
    config = inference_config_for_model(
        definition,
        hardware,
        env_vars={"MARIN_TEMP_PREFIX": MODEL_CACHE_ROOT},
        priority=priority,
        api_model=SOLVER_MODELS[solver],
    )
    if not isinstance(config.engine, VllmEngineConfig):
        raise ValueError("Difficulty local solvers require the vLLM engine")
    config = replace(
        config,
        engine=replace(config.engine, gpu_release=solver_vllm_release(solver), startup_timeout_seconds=259200),
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


@contextmanager
def judge_serving(tier: JudgeTier | None, priority: int, shared_model: str | None = None):
    """Yield the campaign's judge route: the shared GLM relay, a declared judge, or one locally hosted tier.

    A tier is hosted for the whole campaign lifetime; the endpoint is validated
    against its tier pin before the first judged trial dispatches.
    """
    if tier is None:
        yield TOGETHER_BASE_URL, shared_model or TOGETHER_MODEL, require_secret(TOGETHER_API_KEY_ENV), []
        return
    if tier not in JUDGE_TIER_SERVING:
        raise ValueError(f"Difficulty hosts no model configuration for judge tier {tier.value}")
    definition = JUDGE_TIERS[tier]
    model_config, gpu_count = JUDGE_TIER_SERVING[tier]
    endpoint = LocalJudgeEndpoint(
        tier=tier,
        model_config=model_config,
        context_window_tokens=definition.context_window_tokens,
        accelerator=AcceleratorChoice(Platform.GPU, gpu_type="H100", gpu_count=gpu_count, target_cluster=TARGET_CLUSTER),
        priority=priority,
        env_vars={"MARIN_TEMP_PREFIX": MODEL_CACHE_ROOT},
        worker_timeout=JOB_TIMEOUT,
    )
    with endpoint.serve() as session:
        serving = session.model.endpoint
        yield serving.base_url, serving.model, serving.api_key or "EMPTY", [str(job.job_id) for job in session.jobs]


def queue_judge_tier(judged_models: Iterable[str]) -> JudgeTier | None:
    """The single hosted judge tier a queue's judge models require, when any.

    A run version serves exactly one judge route. Models on the shared GLM
    route share the relay; models declaring a hosted tier require that tier,
    and mixing hosted tiers inside one queue is a split-the-queue error.
    """
    tiers = {
        tier
        for tier, definition in JUDGE_TIERS.items()
        if definition.spec.model in judged_models and definition.spec.model != GLM_MODEL
    }
    if not tiers:
        return None
    if len(tiers) > 1:
        raise ValueError(f"One campaign serves one judge tier; split the source queue: {sorted(tiers)}")
    tier = next(iter(tiers))
    if tier not in JUDGE_TIER_SERVING:
        raise ValueError(f"Difficulty hosts no model configuration for judge tier {tier.value}")
    return tier


def batch_judge_route(prepared: list[PreparedSource]) -> tuple[JudgeTier | None, str | None]:
    """The judge tier and shared route model this run must host, when its sources are judged.

    A run version serves exactly one judge route. GLM-judged sources share the
    Together route's model; a bundle's declared judge (REWARDKIT_JUDGE) must be
    a qualified pin or the standing-ruling substitute route, and cannot share
    its queue with GLM or a hosted tier, since the shared relay forces one model.
    """
    judged_models = {
        model
        for source in prepared
        for task in source.tasks
        if task.reason is None and (model := attempt_judge_model(task)) is not None
    }
    tiers = {
        tier
        for tier, definition in JUDGE_TIERS.items()
        if definition.spec.model in judged_models and definition.spec.model != GLM_MODEL
    }
    declared = {model for model in judged_models if model in DECLARED_JUDGE_PINS or model == SUBSTITUTE_JUDGE_MODEL}
    tier_models = {definition.spec.model for definition in JUDGE_TIERS.values()}
    unqualified = judged_models - tier_models - declared - {GLM_MODEL, TOGETHER_MODEL}
    if unqualified:
        raise ValueError(f"Judged models this lane has not qualified: {sorted(unqualified)}")
    if len(tiers) > 1:
        raise ValueError(f"One campaign serves one judge tier; split the source queue: {sorted(tiers)}")
    if declared and (tiers or {GLM_MODEL, TOGETHER_MODEL} & judged_models):
        raise ValueError("Declared judges require their own queue; the shared GLM route serves one model")
    if len(declared) > 1:
        raise ValueError(f"One campaign serves one judge route; split the source queue: {sorted(declared)}")
    if tiers:
        return next(iter(tiers)), None
    if declared:
        return None, next(iter(declared))
    return None, None


def require_daytona_connection_environment(daytona_api_key: str | None) -> None:
    """Fail unless the coordinator environment carries the full Daytona connection.

    Harbor resolves the connection from DAYTONA_API_KEY, DAYTONA_API_URL and
    DAYTONA_TARGET in the worker's own environment; only the API url defaults,
    so a job spec that drops one name surfaces at the first Daytona trial
    instead of at submit.
    """
    missing = [
        name
        for name, value in (
            ("DAYTONA_DATA_API_KEY", daytona_api_key),
            ("DAYTONA_API_URL", os.environ.get("DAYTONA_API_URL")),
            ("DAYTONA_TARGET", os.environ.get("DAYTONA_TARGET")),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Daytona tasks require the connection environment on the campaign: " f"missing {', '.join(missing)}"
        )


def run_trial_worker(
    prepared_path: Path,
    config_path: Path,
    local_root: Path,
    output: StoragePath,
    api_key: str,
    judge: JudgeRun | None,
    ledger: str,
    action_manifest: str,
    daytona_api_key: str | None = None,
) -> None:
    """Launch the locked Harbor runtime with private host-side secret files."""
    secrets_dir = local_root / "framework-secrets"
    secrets_dir.mkdir(mode=0o700)
    secrets = {**forwarded_secrets(), "ATLAS_DIFFICULTY_CANDIDATE_TOKEN": api_key}
    if judge is not None and judge.api_key_secret is not None:
        secrets[judge.api_key_secret] = require_secret(judge.api_key_secret)
    # Task bundles resolve provider credential templates from their granted
    # guest environment; the worker's dispatch scan fails closed when a bundle
    # references one this wave cannot grant. Forward the job bundle's provider
    # credential files so that scan can resolve them without a second copy of
    # the value in this process.
    for name in sorted(TASK_PROVIDER_SECRETS):
        source = secret_path(name)
        if source.is_file():
            target = secrets_dir / name
            target.write_bytes(source.read_bytes())
            target.chmod(0o600)
        else:
            LOGGER.info(
                "Provider credential %s is absent from the job bundle; tasks referencing it fail at dispatch", name
            )
    if daytona_api_key is not None:
        secrets["DAYTONA_DATA_API_KEY"] = daytona_api_key
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
            "experiments.post_training.task_curation.difficulty.ledger_worker",
            "--sources",
            str(prepared_path),
            "--config",
            str(config_path),
            "--output-root",
            str(output / "trials"),
            "--local-root",
            str(local_root / "trials"),
            "--ledger",
            ledger,
            "--action-manifest",
            action_manifest,
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
    parser.add_argument("--solver-pool", help="Immutable measured registry for persistent regional engines")
    parser.add_argument("--sources", required=True)
    parser.add_argument("--sources-sha256", required=True)
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--attempts-per-task", type=int, required=True)
    parser.add_argument("--registry-auth-secret")
    parser.add_argument(
        "--priority",
        required=True,
        choices=PRIORITY_CHOICES,
        help="Band for this campaign's serving jobs; production preempts interactive peers.",
    )
    parser.add_argument(
        "--daytona-provisioning",
        choices=("snapshot", "declarative"),
        default="snapshot",
        help="Provision through named snapshots or original-image declarative builds in the DATA account.",
    )
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--action-manifest", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    try:
        run_batch(args)
    except Exception as error:
        record_action_failure(args.ledger, args.action_manifest, error)
        raise


def run_batch(args: argparse.Namespace) -> None:
    """Run a pinned batch and publish its serving and trial evidence."""
    output = StoragePath(args.output_root)
    if args.local_root.exists() or output.exists():
        raise FileExistsError("Preserve prior campaigns and select a new run version")
    sources = load_queue_sources(args.sources, args.sources_sha256, tuple(args.source_id or ()))
    pin = PINS[args.solver]
    action = json.loads(StoragePath(args.action_manifest).read_text())
    campaign = campaign_record(args.ledger)
    if campaign is not None and args.solver_pool is None:
        raise ValueError("The TPU campaign requires its qualified regional pool registry")
    if campaign is not None and args.solver in SELF_HOSTED_SOLVERS and action["vllm_release_tag"] != TPU_SERVING_ID:
        raise ActionFailure(GapReason.SERVING_RELEASE_MISMATCH, "Action does not bind the maintained TPU serving pair")
    release = VLLM_GPU_RELEASE if args.solver_pool is not None else solver_vllm_release(args.solver)
    if campaign is None and args.solver == Solver.SNOWBALL and action["vllm_release_tag"] != release.release_tag:
        raise ActionFailure(
            GapReason.SERVING_RELEASE_MISMATCH,
            f"Snowball action binds {action['vllm_release_tag']}; serving requires {release.release_tag}",
        )
    args.local_root.mkdir(parents=True)
    options = worker_limits(args.local_root)
    # Pulled actor images go to the node's NVMe scratch, never the container's
    # memory-backed root: a cached image also survives later pods on the node.
    options.setdefault("image_cache", LOCAL_IMAGE_CACHE)
    if args.registry_auth_secret is not None:
        authfile = args.local_root / "registry-auth.json"
        authfile.write_text(require_secret(args.registry_auth_secret))
        authfile.chmod(384)
        options["registry_authfile"] = str(authfile)
    prepared = [
        (
            prepare_source(source, args.local_root / "prepared", judge_version=JUDGE_VERSION)
            if campaign
            else prepare_source(source, args.local_root / "prepared")
        )
        for source in sources
    ]
    requested = {source["source_id"]: {task[0] for task in source["tasks"]} for source in action["sources"]}
    prepared = [
        source.model_copy(
            update={"tasks": [task for task in source.tasks if task.task_id in requested[source.sample.source_id]]}
        )
        for source in prepared
    ]
    has_judge = any(
        task.reason is None and attempt_judge_model(task) is not None for source in prepared for task in source.tasks
    )
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
            "campaign": campaign,
        },
    )
    deviations = (
        ["Hosted weight revision is opaque; tokenizer uses the production config pin. Server fingerprint is archived."]
        if args.solver == Solver.HOSTED
        else []
    )
    settings = sampling_settings(args.solver, deviations)
    provider_limits = ProviderLimits(
        max_in_flight=4 if args.solver == Solver.HOSTED and args.solver_pool is None else args.workers,
        attempts=6 if args.solver == Solver.HOSTED else 1,
        request_timeout=MODEL_REQUEST_TIMEOUT,
    )
    judge_limits = (
        ProviderLimits(max_in_flight=min(8, args.workers), attempts=6, request_timeout=MODEL_REQUEST_TIMEOUT)
        if has_judge
        else None
    )
    has_daytona = any(task.backend == "daytona" and task.reason is None for source in prepared for task in source.tasks)
    daytona_api_key = None
    if has_daytona:
        # A LOCAL-planned action omits daytona_account entirely; preparation may have routed
        # its tasks up to Daytona after planning.
        routed_account = action.get("daytona_account")
        if routed_account not in (None, DATA_ACCOUNT):
            raise ValueError("Difficulty Daytona attempts require the DATA account")
        if routed_account is None:
            # Preparation routed these tasks up to Daytona after the action was planned on the
            # LOCAL tier (the grader drops privileges to a fixed uid); they consume the DATA
            # account like any Daytona attempt.
            LOGGER.info(
                "Grader tier routing serves %d task(s) from Daytona on an action planned for the LOCAL tier",
                sum(task.backend == "daytona" and task.reason is None for source in prepared for task in source.tasks),
            )
        daytona_api_key = require_secret("DAYTONA_DATA_API_KEY")
        require_daytona_connection_environment(daytona_api_key)
    pool = (
        solver_pool_record(args.solver, args.solver_pool)
        if args.solver_pool is not None and args.solver in SELF_HOSTED_SOLVERS
        else None
    )
    if pool is not None:
        model_definition = pool["effective_serving"]
        gpu_release = (
            {"platform": "tpu", "campaign_id": campaign["campaign_id"]} if campaign else asdict(VLLM_GPU_RELEASE)
        )
    elif args.solver in SELF_HOSTED_SOLVERS:
        model_definition = asdict(serving_model(args.solver, settings.context_tokens))
        gpu_release = asdict(solver_vllm_release(args.solver))
    else:
        model_definition = gpu_release = None
    if pool is not None:
        connection = solver_pool_connection(pool, args.action_manifest) if campaign else solver_pool_connection(pool)
        route = nullcontext((connection.base_url, connection.model, connection.api_key or "EMPTY", []))
    else:
        route = solver_endpoint(args.solver, settings, priority_band_value(args.priority))
    with route as (
        endpoint,
        api_model,
        api_key,
        jobs,
    ):
        probe = (
            probe_with_transient_retries(args.solver, endpoint, api_model, api_key, settings)
            if args.solver == Solver.HOSTED
            else probe_endpoint(args.solver, endpoint, api_model, api_key, settings)
        )
        judge_tier, shared_judge_model = batch_judge_route(prepared)
        if judge_limits is not None and shared_judge_model in DECLARED_JUDGE_PINS:
            judge_limits = replace(judge_limits, request_timeout=DECLARED_JUDGE_REQUEST_TIMEOUT)
        judge_route = (
            judge_serving(judge_tier, priority_band_value(args.priority), shared_judge_model)
            if has_judge
            else nullcontext((None, None, None, []))
        )
        with judge_route as (judge_endpoint, judge_api_model, judge_api_key, judge_jobs):
            judge_metadata = None
            if has_judge:
                if judge_tier is None:
                    substitute = shared_judge_model == SUBSTITUTE_JUDGE_MODEL
                    declared_pin = None if shared_judge_model is None else DECLARED_JUDGE_PINS.get(shared_judge_model)
                    if substitute:
                        judge_probe = probe_declared_judge_endpoint(
                            judge_endpoint,
                            judge_api_model,
                            judge_api_key,
                            PROTOCOLS[PROTOCOL_ID].temperature,
                            SUBSTITUTE_JUDGE_PIN,
                        )
                    elif declared_pin is not None:
                        judge_probe = probe_declared_judge_endpoint(
                            judge_endpoint,
                            judge_api_model,
                            judge_api_key,
                            PROTOCOLS[PROTOCOL_ID].temperature,
                            declared_pin,
                        )
                    elif args.solver == Solver.HOSTED and shared_judge_model is None and pool is None:
                        judge_probe = probe
                    else:
                        judge_probe = probe_with_transient_retries(
                            Solver.HOSTED,
                            judge_endpoint,
                            judge_api_model,
                            judge_api_key,
                            sampling_settings(Solver.HOSTED, []),
                        )
                    if substitute:
                        judge_metadata = {
                            "model": judge_api_model,
                            "api_model": judge_api_model,
                            "revision": "opaque-provider",
                            "judge_version": JUDGE_VERSION if campaign else "standing-route-repair-20261124",
                            "tokenizer": judge_api_model,
                            "tokenizer_revision": SUBSTITUTE_JUDGE_PIN.revision,
                            "chat_template_sha256": SUBSTITUTE_JUDGE_PIN.template_sha256,
                            "effective_temperature": PROTOCOLS[PROTOCOL_ID].temperature,
                            "recorded_temperature_deviation": (
                                "The source's declared judge is unserved and re-declared to the served "
                                "Together GLM-5.3-Flash route per the standing judge ruling (a protocol "
                                "difference recorded on the source declaration); served at the protocol "
                                "temperature 0.7."
                            ),
                            "token_budget": "original source request; production service enforces its served context",
                            "basis": (
                                "Standing judge ruling: the bundle's declared judge resolves to an unserved "
                                "model and rides its versioned judge-route repair to the served substitute."
                            ),
                            "serving_probe": judge_probe,
                        }
                    elif declared_pin is not None:
                        judge_metadata = {
                            "model": judge_api_model,
                            "api_model": judge_api_model,
                            "revision": declared_pin.revision,
                            "tokenizer": judge_api_model,
                            "tokenizer_revision": declared_pin.revision,
                            "chat_template_sha256": declared_pin.template_sha256,
                            "effective_temperature": PROTOCOLS[PROTOCOL_ID].temperature,
                            "recorded_temperature_deviation": (
                                "Sources whose JudgeSpec declares temperature 0.0 are served at the protocol "
                                "temperature 0.7; the deviation is recorded per attempt as judge_temperature."
                            ),
                            "token_budget": "original source request; production service enforces its served context",
                            "basis": (
                                "Declared judge served on the shared Together route; the bundle's own grader "
                                "model identity."
                            ),
                            "serving_probe": judge_probe,
                        }
                    else:
                        judge_pin = PINS[Solver.HOSTED]
                        judge_metadata = {
                            "model": SOLVER_MODELS[Solver.HOSTED],
                            "api_model": judge_api_model,
                            "revision": (
                                f"unexposed:{judge_probe['tools_thinking_probe']['completion']['system_fingerprint']}"
                            ),
                            "tokenizer": SOLVER_MODELS[Solver.HOSTED],
                            "tokenizer_revision": judge_pin.revision,
                            "chat_template_sha256": judge_pin.template_sha256,
                            "effective_temperature": PROTOCOLS[PROTOCOL_ID].temperature,
                            "recorded_temperature_deviation": (
                                "Sources whose JudgeSpec declares temperature 0.0 are served at the protocol "
                                "temperature 0.7; the deviation is recorded per attempt as judge_temperature."
                            ),
                            "token_budget": "original source request; production service enforces its served context",
                            "basis": "Host override; original TaskSpec and requested judge parameters are retained.",
                            "serving_probe": judge_probe,
                        }
                else:
                    definition = JUDGE_TIERS[judge_tier]
                    judge_probe = probe_judge_endpoint(
                        judge_endpoint, definition, definition.spec.temperature, judge_api_key
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        definition.model_identity, revision=definition.model_revision
                    )
                    judge_metadata = {
                        "model": definition.spec.model,
                        "api_model": judge_api_model,
                        "revision": definition.model_revision,
                        "tokenizer": definition.model_identity,
                        "tokenizer_revision": definition.model_revision,
                        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                        "effective_temperature": definition.spec.temperature,
                        "recorded_temperature_deviation": (
                            "Sources whose JudgeSpec declares temperature 0.0 are served at the tier "
                            "temperature 0.7; the deviation is recorded per attempt as judge_temperature."
                        ),
                        "token_budget": "original source request; production service enforces its served context",
                        "basis": "Pinned judge tier; original TaskSpec and requested judge parameters are retained.",
                        "serving_probe": judge_probe,
                    }
            completion = probe["tools_thinking_probe"]["completion"]
            revision = (
                f"unexposed:{completion['system_fingerprint']}"
                if args.solver == Solver.HOSTED and pool is None
                else pin.revision
            )
            receipt = write_json_evidence(
                output / "serving.json",
                {
                    "protocol": PROTOCOL_ID,
                    "solver": args.solver.value,
                    "model": SOLVER_MODELS[args.solver],
                    "requested_revision": pin.revision,
                    "tokenizer_revision": pin.revision,
                    "chat_template_sha256": pin.template_sha256,
                    "model_definition": model_definition,
                    "serving_jobs": jobs,
                    "solver_pool": pool,
                    "judge_serving_jobs": judge_jobs,
                    "placement": {
                        "root_target_cluster": (
                            "marin" if campaign else "cw-us-east-08a" if args.solver_pool is not None else TARGET_CLUSTER
                        ),
                        "child_region_constraint": "us-central2" if campaign else None,
                        "zone": "us-central2-b" if campaign else None,
                        "model_cache_root": (
                            "gs://marin-us-central2"
                            if campaign
                            else "s3://marin-us-east-08a" if args.solver_pool is not None else MODEL_CACHE_ROOT
                        ),
                        "basis": "The explicitly targeted coordinator and its entire child job tree share one peer.",
                    },
                    "sampling": request_parameters(settings),
                    "provider_limits": asdict(provider_limits),
                    "judge_provider_limits": None if judge_limits is None else asdict(judge_limits),
                    "judge": judge_metadata,
                    "native_client_context_safety_buffer_tokens": NATIVE_CONTEXT_SAFETY_BUFFER,
                    **(
                        {"compilation_requested": True, "cuda_graphs_requested": False}
                        if campaign
                        else {"compilation_and_cuda_graphs_requested": True}
                    ),
                    "vllm_gpu_release": gpu_release,
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
                "daytona_provisioning": args.daytona_provisioning,
                "workers": args.workers,
                # Persistent pool allocation is accounted once by its owner, not by every concurrent batch.
                "gpu_count": 0 if pool is not None else pin.gpus,
                "attempts_per_task": args.attempts_per_task,
                "judge": (
                    None
                    if judge_metadata is None
                    else {
                        "endpoint": judge_endpoint,
                        "api_model": judge_api_model,
                        # The shared judge route serves from Together, so its relay must
                        # authenticate upstream with the same key judge_serving resolves;
                        # the old GLM_BULK_TOKEN name is rejected there (measured 401).
                        "api_key_secret": None if judge_tier is not None else TOGETHER_API_KEY_ENV,
                        "provider_limits": asdict(judge_limits),
                        "identity": {
                            **{
                                key: judge_metadata[key]
                                for key in (
                                    "model",
                                    "revision",
                                    "tokenizer",
                                    "tokenizer_revision",
                                    "chat_template_sha256",
                                )
                            },
                            "serving_receipt": receipt.model_dump(mode="json"),
                        },
                    }
                ),
            }
            config_path = args.local_root / "candidate.json"
            candidate = CandidateRun.model_validate(config)
            config_path.write_text(candidate.model_dump_json() + "\n")
            run_trial_worker(
                prepared_path,
                config_path,
                args.local_root,
                output,
                api_key,
                candidate.judge,
                daytona_api_key=daytona_api_key,
                ledger=args.ledger,
                action_manifest=args.action_manifest,
            )
            completion: dict = {"solver": args.solver, "serving_jobs": jobs, "worker_exit": 0}
            run_seal_path = output / "trials" / "run-seal.json"
            if run_seal_path.exists():
                seal = json.loads(run_seal_path.read_text())
                completion["run_seal"] = {
                    name: seal[name]
                    for name in (
                        "iris_job_id",
                        "started_at",
                        "updated_at",
                        "sources_aggregated",
                        "own_gpu_hours",
                        "own_input_tokens",
                        "own_output_tokens",
                        "judge_own_input_tokens",
                        "judge_own_output_tokens",
                        "gpu_hour_basis",
                        "token_basis",
                    )
                }
            write_json_evidence(output / "completion.json", completion)


if __name__ == "__main__":
    main()
