# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own one bounded difficulty solver campaign and its Iris serving lifetime."""

import argparse
import hashlib
import json
import logging
import os
import subprocess
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType

import httpx
from finestore.rl.curation import ContentEvidenceReference
from finestore.rl.difficulty import PROTOCOL_ID, PROTOCOLS, SOLVER_MODELS, SamplingSettings, Solver
from huggingface_hub import hf_hub_download
from iris.rpc.proto_display import priority_band_value
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
from taskcompendium.pipeline.reviewer_spec import JudgeTier
from transformers import AutoTokenizer

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL
from experiments.post_training.task_curation.difficulty_config import CandidateRun, JudgeRun
from experiments.post_training.task_curation.difficulty_evidence import write_json_evidence
from experiments.post_training.task_curation.difficulty_preparation import PreparedSource, prepare_source
from experiments.post_training.task_curation.difficulty_protocol import (
    PINS,
    TARGET_CLUSTER,
    serving_model,
)
from experiments.post_training.task_curation.difficulty_provider import ProviderLimits
from experiments.post_training.task_curation.difficulty_recovery import CampaignParent
from experiments.post_training.task_curation.difficulty_samples import load_queue_sources
from experiments.post_training.task_curation.difficulty_sampling import request_parameters, sampling_settings
from experiments.post_training.task_curation.difficulty_serving_probe import (
    probe_endpoint,
    probe_judge_endpoint,
)
from experiments.post_training.task_curation.judges import JUDGE_TIERS, LocalJudgeEndpoint

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_API_KEY_ENV = "TOGETHER_API_KEY"
TOGETHER_MODEL = "zai-org/GLM-5.3"


def hosted_endpoint_liveness(base_url: str, api_key: str, model: str) -> None:
    """Fail in seconds when the hosted endpoint is dead, instead of a 30-minute probe timeout.

    One short chat completion with a 60-second timeout separates a live serving
    endpoint from a dead one (a dead relay once sat in the running state with its
    serving pod deleted, and the campaign's serving probe hung for its full window).
    """
    try:
        with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=60) as client:
            response = client.post(
                base_url.rstrip("/") + "/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": "liveness"}], "max_tokens": 1},
            )
            response.raise_for_status()
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
JOB_TIMEOUT = Duration.from_seconds(259200)
NATIVE_CONTEXT_SAFETY_BUFFER = 0
# A hosted endpoint queues probes behind hosted traffic; a retry succeeds once
# the queue drains instead of killing the campaign (hosted-v13c died on a
# single probe ReadTimeout).
SERVING_PROBE_ATTEMPTS = 3


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


@contextmanager
def judge_serving(tier: JudgeTier | None, priority: int):
    """Yield the campaign's judge route: the shared GLM relay, or one locally hosted tier.

    A tier is hosted for the whole campaign lifetime; the endpoint is validated
    against its tier pin before the first judged trial dispatches.
    """
    if tier is None:
        yield TOGETHER_BASE_URL, TOGETHER_MODEL, require_secret(TOGETHER_API_KEY_ENV), []
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


def campaign_judge_tier(prepared: list[PreparedSource]) -> JudgeTier | None:
    """The single judge tier this run must host, when its judged sources need one.

    A run version serves exactly one judge route. Sources whose judge is the GLM
    route share the relay; sources declaring a hosted tier model require that
    tier, and mixing routes inside one queue is a split-the-queue error.
    """
    judged_models = {
        task.judge_model
        for source in prepared
        for task in source.tasks
        if task.judge_model is not None and task.reason is None
    }
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


def run_trial_worker(
    prepared_path: Path,
    config_path: Path,
    local_root: Path,
    output: StoragePath,
    api_key: str,
    recovery_arguments: list[str],
    judge: JudgeRun | None,
    daytona_api_key: str | None = None,
) -> None:
    """Launch the locked Harbor runtime with private host-side secret files."""
    secrets_dir = local_root / "framework-secrets"
    secrets_dir.mkdir(mode=0o700)
    secrets = {**forwarded_secrets(), "ATLAS_DIFFICULTY_CANDIDATE_TOKEN": api_key}
    if judge is not None and judge.api_key_secret is not None:
        secrets[judge.api_key_secret] = require_secret(judge.api_key_secret)
    worker_env = {**os.environ}
    if daytona_api_key is not None:
        # Harbor reads the Daytona connection from DAYTONA_API_KEY; measurement
        # runs map the campaign's resolved credential onto that name without
        # printing it and without touching other campaigns' accounts.
        worker_env["DAYTONA_API_KEY"] = daytona_api_key
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
            **worker_env,
            "PYTHONPATH": str(Path.cwd()),
            SECRETS_DIR_ENV: str(secrets_dir),
            "HARBOR_VLLM_CONTEXT_SAFETY_BUFFER": str(NATIVE_CONTEXT_SAFETY_BUFFER),
        },
    )


def same_run_output(output: StoragePath, args: argparse.Namespace) -> bool:
    """Whether an existing output root is a partial write of this same run version.

    Iris retries a preempted root task with identical arguments; the retry must
    resume from its own sealed output instead of failing the new-version guard.
    The binding is verified against the invocation's frozen source queue, its
    declared parent campaign and the root's solver, so a run version can only
    resume from evidence it (or its declared parent) produced. An existing root
    that fails this check keeps the preserve-prior-campaigns error.
    """
    prepared_path = output / "prepared.json"
    if not prepared_path.exists():
        return False
    prepared = json.loads(prepared_path.read_text())
    if prepared.get("source_queue") != args.sources or prepared.get("source_queue_sha256") != args.sources_sha256:
        return False
    parent_path = output / "trials" / "parent-campaign.json"
    if args.previous_campaign is None:
        if parent_path.exists():
            return False
    else:
        if not parent_path.exists():
            return False
        parent = CampaignParent.model_validate_json(parent_path.read_text())
        if parent.root != args.previous_campaign or parent.serving_receipt.sha256 != args.previous_serving_sha256:
            return False
    serving_path = output / "serving.json"
    if serving_path.exists():
        serving = json.loads(serving_path.read_text())
        if serving.get("solver") != args.solver.value:
            return False
    return True


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
        help=(
            "How Daytona tasks provision machines: snapshot uses pre-created RL-account "
            "snapshots (run the shellbox snapshot preflight first); declarative builds the "
            "original image or Dockerfile at sandbox creation and needs an account that "
            "permits declarative builds."
        ),
    )
    args = parser.parse_args()
    if bool(args.previous_campaign) != bool(args.previous_serving_sha256):
        raise ValueError("Continuation requires both the previous campaign and its serving receipt digest")
    configure_coreweave_s3()
    output = StoragePath(args.output_root)
    resumed = output.exists() and same_run_output(output, args)
    if args.local_root.exists() or (output.exists() and not resumed):
        raise FileExistsError("Preserve prior campaigns and select a new run version")
    sources = load_queue_sources(args.sources, args.sources_sha256, tuple(args.source_id or ()))
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
    if resumed:
        # The retried attempt re-prepares the same frozen sources; the published
        # prepared receipt is already durable and must not be replaced.
        prior_prepared = json.loads((output / "prepared.json").read_text())
        if [row["sample"]["source_id"] for row in prior_prepared["sources"]] != [
            source.sample.source_id for source in prepared
        ]:
            raise ValueError("Resumed root's prepared sources differ from this invocation")
    else:
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
    has_daytona = any(task.backend == "daytona" and task.reason is None for source in prepared for task in source.tasks)
    daytona_api_key = None
    if has_daytona:
        # Harbor reads the Daytona connection from DAYTONA_API_KEY. The
        # launcher environment carries the account's key directly, or the
        # dedicated RL measurement key under its documented name.
        daytona_api_key = os.environ.get("DAYTONA_API_KEY") or os.environ.get("DAYTONA_RL_API_KEY")
        if not daytona_api_key:
            raise ValueError(
                "Daytona tasks require DAYTONA_API_KEY or DAYTONA_RL_API_KEY in the coordinator environment"
            )
    with solver_endpoint(args.solver, settings, priority_band_value(args.priority)) as (
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
        prior_serving_path = output / "serving.json"
        prior_serving = json.loads(prior_serving_path.read_text()) if resumed and prior_serving_path.exists() else None
        if prior_serving is not None:
            # The retried attempt keeps its own published serving receipt, so the
            # live serving identity must still be the one that receipt recorded.
            fresh_revision = (
                f"unexposed:{probe['tools_thinking_probe']['completion']['system_fingerprint']}"
                if args.solver == Solver.HOSTED
                else pin.revision
            )
            prior_revision = (
                f"unexposed:{prior_serving['tools_thinking_probe']['completion']['system_fingerprint']}"
                if args.solver == Solver.HOSTED
                else prior_serving["requested_revision"]
            )
            changed = [
                name
                for name, before, after in (
                    ("protocol", prior_serving.get("protocol"), PROTOCOL_ID),
                    ("solver", prior_serving.get("solver"), args.solver.value),
                    ("model", prior_serving.get("model"), SOLVER_MODELS[args.solver]),
                    ("revision", prior_revision, fresh_revision),
                    ("tokenizer_revision", prior_serving.get("tokenizer_revision"), pin.revision),
                    ("chat_template_sha256", prior_serving.get("chat_template_sha256"), pin.template_sha256),
                    ("sampling", prior_serving.get("sampling"), request_parameters(settings)),
                )
                if before != after
            ]
            if changed:
                raise ValueError("Resumed root's serving identity changed across the retry: " + ", ".join(changed))
        judge_tier = campaign_judge_tier(prepared)
        judge_route = (
            judge_serving(judge_tier, priority_band_value(args.priority))
            if has_judge
            else nullcontext((None, None, None, []))
        )
        with judge_route as (
            judge_endpoint,
            judge_api_model,
            judge_api_key,
            judge_jobs,
        ):
            judge_metadata = None
            if has_judge:
                if judge_tier is None:
                    if args.solver == Solver.HOSTED:
                        judge_probe = probe
                    else:
                        # The shared GLM relay may queue the probe behind hosted traffic; a
                        # retry succeeds once the relay drains instead of killing the campaign.
                        judge_probe = probe_with_transient_retries(
                            Solver.HOSTED,
                            judge_endpoint,
                            judge_api_model,
                            judge_api_key,
                            sampling_settings(Solver.HOSTED, []),
                        )
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
                    # A hosted tier judge is validated against its own pinned
                    # rendering before any judged trial dispatches.
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
                if prior_serving is not None and prior_serving.get("judge") is not None:
                    # Keep the killed attempt's judge identity for the whole run
                    # version; a fresh fingerprint from the same shared relay would
                    # otherwise split the version's judged history across receipts.
                    prior_judge = prior_serving["judge"]
                    changed_judge = [
                        name
                        for name, before, after in (
                            ("model", prior_judge.get("model"), judge_metadata["model"]),
                            ("api_model", prior_judge.get("api_model"), judge_metadata["api_model"]),
                            ("tokenizer", prior_judge.get("tokenizer"), judge_metadata["tokenizer"]),
                            (
                                "tokenizer_revision",
                                prior_judge.get("tokenizer_revision"),
                                judge_metadata["tokenizer_revision"],
                            ),
                            (
                                "chat_template_sha256",
                                prior_judge.get("chat_template_sha256"),
                                judge_metadata["chat_template_sha256"],
                            ),
                            (
                                "effective_temperature",
                                prior_judge.get("effective_temperature"),
                                judge_metadata["effective_temperature"],
                            ),
                        )
                        if before != after
                    ]
                    if changed_judge:
                        raise ValueError(
                            "Resumed root's judge identity changed across the retry: " + ", ".join(changed_judge)
                        )
                    judge_metadata = prior_judge
            completion = probe["tools_thinking_probe"]["completion"]
            revision = f"unexposed:{completion['system_fingerprint']}" if args.solver == Solver.HOSTED else pin.revision
            if prior_serving is not None:
                receipt = ContentEvidenceReference(uri=str(prior_serving_path), sha256=file_sha256(prior_serving_path))
            else:
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
                            None
                            if args.solver == Solver.HOSTED
                            else asdict(serving_model(args.solver, settings.context_tokens))
                        ),
                        "serving_jobs": jobs,
                        "judge_serving_jobs": judge_jobs,
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
                "daytona_provisioning": args.daytona_provisioning,
                "workers": args.workers,
                "gpu_count": pin.gpus,
                "attempts_per_task": args.attempts_per_task,
                "task_limit": args.task_limit,
                "judge": (
                    None
                    if judge_metadata is None
                    else {
                        "endpoint": judge_endpoint,
                        "api_model": judge_api_model,
                        "api_key_secret": None if judge_tier is not None else GLM_BULK_TOKEN_ENV,
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
            recovery_arguments = []
            if resumed and prior_serving is not None:
                # The retried attempt continues from its own sealed output: the root
                # binds itself as the recovery parent, and its parent-campaign.json
                # keeps the original chain intact.
                recovery_arguments = [
                    "--previous-campaign",
                    str(output),
                    "--previous-serving-sha256",
                    receipt.sha256,
                    "--source-queue-sha256",
                    args.sources_sha256,
                ]
            elif args.previous_campaign is not None:
                recovery_arguments = [
                    "--previous-campaign",
                    args.previous_campaign,
                    "--previous-serving-sha256",
                    args.previous_serving_sha256,
                    "--source-queue-sha256",
                    args.sources_sha256,
                ]
            run_trial_worker(
                prepared_path,
                config_path,
                args.local_root,
                output,
                api_key,
                recovery_arguments,
                candidate.judge,
                daytona_api_key=daytona_api_key,
            )
            completion: dict = {"solver": args.solver, "serving_jobs": jobs, "worker_exit": 0}
            run_seal_path = output / "trials" / "run-seal.json"
            if run_seal_path.exists():
                # The worker's checkpointed own-cost seal travels with the completion
                # receipt: per-run GPU-hours and token deltas, not cumulative counters.
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
