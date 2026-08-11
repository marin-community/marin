#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-cluster steps of an evidence-gated Iris controller rollout.

The ``deploy-iris-controllers`` skill drives these subcommands. The restart itself
stays in ``iris cluster controller restart``: this script never restarts a
controller and never walks the cluster list. The skill continues after passed
gates and stops when a gate is blocked.
"""

import functools
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import click
import yaml
from iris.cli.build import get_git_sha
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS, connect_controller, rpc_client
from iris.client import IrisClient, Job, LocalClientConfig
from iris.client.local_client import local_client
from iris.cluster.config import IrisClusterConfig, load_config
from iris.resources.action import ActionState
from iris.resources.execution import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.resources.job import JobQuery, JobSummary, PriorityBand
from iris.resources.log import LogPage
from iris.resources.state import JobState
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.worker_codec import provenance_from_proto
from rigging.config_discovery import list_cluster_configs, resolve_cluster_config
from rigging.filesystem.cluster_config import StoreType, store_config
from rigging.provenance import Provenance
from rigging.secrets import ENV_SCHEME, GCP_SECRET_SCHEME, as_secret_spec, resolve_secret_spec
from rigging.timing import Duration, ExponentialBackoff

logger = logging.getLogger(__name__)

# Clusters that lead every rollout, in this order: the dev cluster proves the tree,
# then production. Both are GCE/IAP clusters that restart over `gcloud compute ssh`.
DEPLOY_LEAD = ("marin-dev", "marin")

# Config stems that are never part of an operator rollout: CI owns them and brings
# them up per run from its own workflow.
CI_CONFIG_PREFIX = "ci-"

# The branch a deploy is expected to ship. Anything else is code no reviewer saw.
DEFAULT_UPSTREAM = "origin/main"

# Uncommitted files named in the warning before it summarizes the remainder.
DIRTY_FILES_SHOWN = 5

# Ceiling for a preflight probe command, so a hung daemon does not hang the gate.
PROBE_TIMEOUT = 30.0

# Where kubectl and the kubernetes client look when no path is pinned and KUBECONFIG is
# unset. A cluster config may name a context without a file and still deploy.
DEFAULT_KUBECONFIG = "~/.kube/config"

# `iris job run` defaults. Smoke jobs ask for nothing more than a real user job's
# floor, so they schedule on any cluster without competing with real work.
SMOKE_CPU = 0.1
SMOKE_MEMORY = "1GB"
SMOKE_DISK = "5GB"
SMOKE_SUCCESS_MARKER = "iris-rollout-smoke-success"
SMOKE_CANCEL_MARKER = "iris-rollout-smoke-cancel-ready"
SMOKE_FOLLOWUP_MARKER = "iris-rollout-smoke-followup"
SMOKE_SUCCESS_COMMAND = ("bash", "-c", f"echo {SMOKE_SUCCESS_MARKER}")
SMOKE_CANCEL_COMMAND = ("bash", "-c", f"echo {SMOKE_CANCEL_MARKER}; while true; do sleep 5; done")
SMOKE_FOLLOWUP_COMMAND = ("bash", "-c", f"echo {SMOKE_FOLLOWUP_MARKER}")

# A cluster with no idle worker must scale one up before the smoke job can run. On
# marin-dev, which sits at 0 workers and is TPU-backed, that takes 15-20 minutes, so
# the wait covers a scale-up rather than only the dispatch of an already-idle worker.
SMOKE_TIMEOUT = 1800

# Once a job has started, log ingestion and action completion should be quick.
# Separate ceilings keep a broken log or cancellation path from consuming the
# full scale-up allowance above.
SMOKE_LOG_TIMEOUT = 60
SMOKE_ACTION_TIMEOUT = 300
SMOKE_POLL_INTERVAL = 5

# Post-restart watch: the skill's gate wants ~5 minutes of steady controller.
WATCH_DURATION = 300
WATCH_INTERVAL = 30

# A small health loss can be normal autoscaler churn during the watch. Always
# permit one execution unit, then permit 5% of larger baselines.
HEALTH_LOSS_TOLERANCE_PERCENT = 5

# Rough counts are enough for a before/after comparison, and a bounded query keeps
# the RPC cheap on a busy controller.
JOB_COUNT_LIMIT = 500

# Snapshot job-count keys. `compare` reads the first two by name, so they are named
# constants rather than literals repeated at both ends.
JOB_RUNNING = "running"
JOB_PENDING = "pending"
JOB_BUILDING = "building"

WATCHED_JOB_STATES = (
    (JOB_RUNNING, job_pb2.JOB_STATE_RUNNING),
    (JOB_PENDING, job_pb2.JOB_STATE_PENDING),
    (JOB_BUILDING, job_pb2.JOB_STATE_BUILDING),
)


class Need(StrEnum):
    """What kind of thing a requirement names, which decides how it is checked."""

    ENV = "env"
    COMMAND = "command"
    COMMAND_RUNS = "command-runs"
    SECRET = "secret"
    KUBE_CONTEXT = "kube-context"


@dataclass(frozen=True)
class Requirement:
    """One operator-side input the deploy consumes, and why it is consumed."""

    kind: Need
    target: str
    why: str
    # For KUBE_CONTEXT: the kubeconfig the cluster config names. An exported
    # KUBECONFIG overrides it at check time, exactly as the deploy resolves it.
    source: str = ""

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.target}"


@dataclass(frozen=True)
class CheckResult:
    requirement: Requirement
    ok: bool
    detail: str


class HealthUnit(StrEnum):
    """The backend resource used for rollout health."""

    WORKER = "workers"
    NODE = "nodes"


@dataclass(frozen=True)
class ExecutionHealth:
    """The healthy execution resources for one backend."""

    backend_id: str
    unit: HealthUnit
    healthy: int
    total: int

    def summary(self) -> str:
        return f"{self.backend_id}={self.healthy}/{self.total} healthy {self.unit.value}"


@dataclass(frozen=True)
class Snapshot:
    """A rough controller state reading, taken before and after a restart."""

    cluster: str
    captured_at: str
    reachable: bool
    tree_hash: str = ""
    version: str = ""
    execution_health: tuple[ExecutionHealth, ...] = ()
    jobs: dict[str, int] = field(default_factory=dict)
    error: str = ""
    # The tree this session would deploy, read when the baseline is taken. `verify`
    # compares against it, so an edit made during the restart cannot move the target.
    working_tree_hash: str = ""

    def summary(self) -> str:
        if not self.reachable:
            return f"{self.cluster}: UNREACHABLE ({self.error})"
        # A count at the cap is a floor, not a total.
        counts = " ".join(
            f"{name}={count}{'+' if count >= JOB_COUNT_LIMIT else ''}" for name, count in sorted(self.jobs.items())
        )
        health = ", ".join(item.summary() for item in self.execution_health)
        identity = f"{self.cluster}: version={self.version} tree={self.tree_hash}"
        return f"{identity} health {health}, jobs {counts}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "Snapshot":
        data = json.loads(raw)
        data["execution_health"] = tuple(
            ExecutionHealth(
                backend_id=item["backend_id"],
                unit=HealthUnit(item["unit"]),
                healthy=item["healthy"],
                total=item["total"],
            )
            for item in data["execution_health"]
        )
        return cls(**data)


class TreeIssue(StrEnum):
    """Why a tree is not a clean checkout of the upstream branch."""

    DIRTY = "dirty"
    BEHIND = "behind"
    AHEAD = "ahead"


@dataclass(frozen=True)
class TreeWarning:
    issue: TreeIssue
    message: str


@dataclass(frozen=True)
class TreeState:
    """What the working tree would ship, relative to the upstream branch.

    A restart builds images from HEAD plus staged and unstaged changes, so a dirty
    or behind tree deploys code that is in no upstream commit.
    """

    tree_hash: str
    branch: str
    base_commit: str
    dirty_files: tuple[str, ...]
    upstream: str
    ahead: int
    behind: int

    def summary(self) -> str:
        position = f"{self.ahead} ahead / {self.behind} behind {self.upstream}"
        state = f"{len(self.dirty_files)} uncommitted file(s)" if self.dirty_files else "clean"
        return f"tree={self.tree_hash} branch={self.branch or '(detached)'} base={self.base_commit} {state}, {position}"


@dataclass(frozen=True)
class Verdict:
    """The post-restart comparison. ``concerns`` is why a human should look."""

    healthy: bool
    concerns: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SmokeSuiteResult:
    """Resource identifiers produced by one successful smoke suite."""

    completed_job: str
    cancelled_job: str
    followup_job: str
    cancel_action_id: str

    def summary(self) -> str:
        return (
            f"completed={self.completed_job}, cancelled={self.cancelled_job}, "
            f"followup={self.followup_job}, cancel_action={self.cancel_action_id}"
        )


def deploy_candidates(configs: Mapping[str, Path]) -> dict[str, Path]:
    """Cluster configs an operator rolls out, i.e. everything CI does not own."""
    return {name: path for name, path in configs.items() if not name.startswith(CI_CONFIG_PREFIX)}


def cluster_capacity(config: IrisClusterConfig) -> int:
    """Summed `max_slices` config cap across scale groups. An ordering key only.

    This is not a slice count the cluster can reach. On a TPU cluster it is far
    above one, because `tpu_pools` expands to a scale group per size *per zone*:
    the sum multiplies each cap by the zone count, and it adds the sizes of one
    pool even though those are tiers over the same quota. Flat scale groups, which
    is what every CoreWeave config declares, do sum honestly.

    Used only to order non-lead clusters smallest-first, so a bad deploy hits the
    least capacity first. Config caps (not live workers) keep the order stable
    before any controller is contacted. `DEPLOY_LEAD` pins the GCE clusters ahead
    of that sort, so their inflated totals never order anything. `plan` does not
    print a cap for them, because the figure would only mislead.
    """
    return sum(group.max_slices for group in config.scale_groups.values())


def rollout_order(capacities: Mapping[str, int]) -> tuple[str, ...]:
    """Order clusters for a rollout: dev, production, then the rest smallest-first."""
    lead = [name for name in DEPLOY_LEAD if name in capacities]
    rest = sorted((name for name in capacities if name not in lead), key=lambda name: (capacities[name], name))
    return tuple(lead + rest)


def parse_dirty_files(porcelain: str) -> tuple[str, ...]:
    """Paths from `git status --porcelain`, which prefixes each with its status code."""
    paths = []
    for line in porcelain.splitlines():
        # Split on whitespace, not at a fixed column: a caller that stripped the
        # output loses the leading status space, and git quotes a path with spaces.
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            paths.append(fields[1])
    return tuple(paths)


def parse_ahead_behind(rev_list_counts: str) -> tuple[int, int]:
    """Split `git rev-list --left-right --count <upstream>...HEAD` into (ahead, behind).

    The left column counts commits only the upstream has (behind), the right column
    commits only HEAD has (ahead).
    """
    parts = rev_list_counts.split()
    if len(parts) != 2:
        raise ValueError(f"unexpected rev-list --count output: {rev_list_counts!r}")
    behind, ahead = (int(part) for part in parts)
    return ahead, behind


def tree_warnings(state: TreeState) -> tuple[TreeWarning, ...]:
    """Reasons a human must confirm this tree before it is deployed."""
    warnings = []
    if state.dirty_files:
        shown = ", ".join(state.dirty_files[:DIRTY_FILES_SHOWN])
        hidden = len(state.dirty_files) - DIRTY_FILES_SHOWN
        more = f" (+{hidden} more)" if hidden > 0 else ""
        warnings.append(
            TreeWarning(
                TreeIssue.DIRTY,
                f"the deploy ships {len(state.dirty_files)} uncommitted file(s): {shown}{more}",
            )
        )
    if state.behind:
        warnings.append(
            TreeWarning(TreeIssue.BEHIND, f"the tree is {state.behind} behind {state.upstream}: it ships stale code")
        )
    if state.ahead:
        warnings.append(
            TreeWarning(TreeIssue.AHEAD, f"the tree is {state.ahead} ahead of {state.upstream}: it ships unmerged code")
        )
    return tuple(warnings)


def _git(args: Sequence[str], *, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def read_tree_state(*, upstream: str = DEFAULT_UPSTREAM, fetch: bool = True) -> TreeState:
    """Read what this checkout would deploy, comparing it against ``upstream``.

    Fetches the upstream ref first (a read-only network call), so "behind" reflects
    the branch as it is now rather than the last local fetch.
    """
    repo = Path(_git(["rev-parse", "--show-toplevel"], cwd=Path.cwd()))
    provenance = Provenance.from_git(repo)
    if fetch:
        remote, _, branch = upstream.partition("/")
        _git(["fetch", "--quiet", remote, branch], cwd=repo)
    ahead, behind = parse_ahead_behind(_git(["rev-list", "--left-right", "--count", f"{upstream}...HEAD"], cwd=repo))
    return TreeState(
        tree_hash=provenance.tree_hash,
        branch=provenance.branch or "",
        base_commit=provenance.base_commit,
        # --untracked-files=all overrides a status.showUntrackedFiles=no in the
        # operator's git config, which would otherwise report a tree with new files as
        # clean. The image build copies them in, and the tree hash does not cover them.
        dirty_files=parse_dirty_files(_git(["status", "--porcelain", "--untracked-files=all"], cwd=repo)),
        upstream=upstream,
        ahead=ahead,
        behind=behind,
    )


def _s3_credential_env() -> tuple[str, str]:
    store = store_config(StoreType.COREWEAVE)
    return store.key_id_env, store.key_secret_env


def requirements(config: IrisClusterConfig, *, s3_env: Sequence[str] | None = None) -> tuple[Requirement, ...]:
    """The operator-side inputs a restart of ``config`` reads from this session.

    Derived from the config so it cannot drift from what the deploy actually
    resolves: ``defaults.inject_env`` (``iris.cluster.inject_env``), the S3
    credentials a Kubernetes controller folds into ``iris-task-env``, and the
    signing-key references a Kubernetes deploy resolves in the operator shell.

    Args:
        config: The cluster config a restart would deploy.
        s3_env: Override for the CoreWeave key env var names. Defaults to the
            names the marin store config declares.
    """
    needs = [
        # `docker` on PATH is not enough: the deploy runs `docker buildx build --push`,
        # which fails without a reachable daemon or without the buildx plugin — and it
        # would fail after the operator already approved the cluster at its gate.
        Requirement(Need.COMMAND_RUNS, "docker info", "the deploy builds the controller image locally"),
        Requirement(Need.COMMAND_RUNS, "docker buildx version", "the image build and push runs through buildx"),
        Requirement(Need.COMMAND, "git", "the image tag is a hash of the working tree content"),
    ]
    for name in config.defaults.inject_env:
        needs.append(Requirement(Need.ENV, name, "defaults.inject_env: the deploy aborts when it is unset"))

    kind = config.controller.controller_kind()
    if kind == "coreweave":
        needs.append(Requirement(Need.COMMAND, "kubectl", "the controller Deployment is driven over the k8s API"))
        platform = config.platform.coreweave
        if platform and platform.kube_context:
            needs.append(
                Requirement(
                    Need.KUBE_CONTEXT,
                    platform.kube_context,
                    "the deploy binds this context, so the kubeconfig must define it",
                    source=platform.kubeconfig_path,
                )
            )
        if config.storage.remote_state_dir.startswith("s3://"):
            for name in s3_env if s3_env is not None else _s3_credential_env():
                needs.append(Requirement(Need.ENV, name, "S3 state auth projected into the iris-task-env Secret"))
        # A Kubernetes controller pod holds no cloud credentials, so the deploy resolves
        # the signing key in the operator's shell and projects it into the Secret. It
        # resolves `persistent or refs`, so an `env:` reference addresses the pod only
        # while a persistent source backs it. Alone, it is what the deploy reads here.
        if config.auth and config.auth.signing_key:
            refs = as_secret_spec(config.auth.signing_key)
            persistent = [ref for ref in refs if not ref.startswith(ENV_SCHEME)]
            for ref in persistent:
                needs.append(Requirement(Need.SECRET, ref, "signing key projected as IRIS_SIGNING_KEY"))
            if not persistent:
                needs.extend(
                    Requirement(
                        Need.ENV,
                        ref.removeprefix(ENV_SCHEME),
                        "the only signing-key source: the deploy reads it from this shell",
                    )
                    for ref in refs
                )
    elif kind == "gcp":
        needs.append(Requirement(Need.COMMAND, "gcloud", "the restart drives the controller VM over IAP SSH"))

    return tuple(needs)


def effective_kubeconfigs(configured: str, *, environ: Mapping[str, str]) -> tuple[Path, ...]:
    """The kubeconfig files the deploy will read, in kubectl's own precedence.

    An exported ``KUBECONFIG`` wins over the configured path — the k8s controller
    manager passes ``kubeconfig_path=None`` when it sees that variable — and kubectl
    merges a path-separated list, so a context may come from any entry. A config that
    pins a context but no path resolves that context against the default kubeconfig.
    """
    override = [part for part in environ.get("KUBECONFIG", "").split(os.pathsep) if part]
    if override:
        return tuple(Path(part).expanduser() for part in override)
    return (Path(configured or DEFAULT_KUBECONFIG).expanduser(),)


def parse_kube_contexts(kubeconfig: str) -> tuple[str, ...]:
    """Context names a kubeconfig document defines."""
    document = yaml.safe_load(kubeconfig) or {}
    contexts = document.get("contexts") or []
    return tuple(entry["name"] for entry in contexts if isinstance(entry, dict) and entry.get("name"))


def check_kube_context(requirement: Requirement, *, environ: Mapping[str, str]) -> CheckResult:
    """Check that a kubeconfig the deploy will read defines the cluster's context."""
    paths = effective_kubeconfigs(requirement.source, environ=environ)
    present = [path for path in paths if path.is_file()]
    if not present:
        listed = ", ".join(str(path) for path in paths)
        return CheckResult(requirement, False, f"kubeconfig is absent: {listed}")

    found: list[str] = []
    for path in present:
        found.extend(parse_kube_contexts(path.read_text()))
    if requirement.target in found:
        return CheckResult(requirement, True, f"defined in {', '.join(str(path) for path in present)}")
    return CheckResult(
        requirement,
        False,
        f"not among the {len(found)} context(s) in {', '.join(str(path) for path in present)}",
    )


def run_probe(requirement: Requirement, *, timeout: float = PROBE_TIMEOUT) -> CheckResult:
    """Run the requirement's command and pass on a zero exit status.

    The target splits on whitespace into argv (no shell), so every probe must be a
    command whose arguments carry no spaces.
    """
    argv = requirement.target.split()
    if shutil.which(argv[0]) is None:
        return CheckResult(requirement, False, f"{argv[0]} is not on PATH")
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return CheckResult(requirement, False, f"{requirement.target!r} timed out after {timeout:.0f}s")
    if result.returncode == 0:
        return CheckResult(requirement, True, "ok")
    reason = (result.stderr or result.stdout).strip().splitlines()
    return CheckResult(requirement, False, f"exit {result.returncode}: {reason[0] if reason else 'no output'}")


def check_requirement(requirement: Requirement, *, environ: Mapping[str, str]) -> CheckResult:
    """Check one requirement. Secret values are resolved but never returned."""
    target = requirement.target
    if requirement.kind is Need.ENV:
        value = environ.get(target, "")
        return CheckResult(requirement, bool(value), "set" if value else "unset")
    if requirement.kind is Need.COMMAND:
        path = shutil.which(target)
        return CheckResult(requirement, path is not None, path or "not on PATH")
    if requirement.kind is Need.COMMAND_RUNS:
        return run_probe(requirement)
    if requirement.kind is Need.KUBE_CONTEXT:
        return check_kube_context(requirement, environ=environ)
    resolved = resolve_secret_spec((target,))
    hint = "" if not target.startswith(GCP_SECRET_SCHEME) else " (needs GCP credentials)"
    return CheckResult(requirement, bool(resolved.value), f"resolved from {resolved.source}{hint}")


def check_requirements(needs: Sequence[Requirement], *, environ: Mapping[str, str]) -> list[CheckResult]:
    """Check every requirement, turning a resolution failure into a failed check."""
    results = []
    for requirement in needs:
        try:
            results.append(check_requirement(requirement, environ=environ))
        except Exception as exc:
            results.append(CheckResult(requirement, False, str(exc)))
    return results


def backend_health(
    backends: Sequence[controller_pb2.Controller.BackendSummary],
) -> tuple[ExecutionHealth, ...]:
    """Read the execution health that each backend type provides."""
    if not backends:
        raise ValueError("controller reported no execution backends")

    health = []
    for backend in backends:
        detail_kind = backend.detail.WhichOneof("detail")
        if detail_kind == "worker":
            detail = backend.detail.worker
            health.append(
                ExecutionHealth(
                    backend_id=backend.backend_id,
                    unit=HealthUnit.WORKER,
                    healthy=detail.healthy_worker_count,
                    total=detail.total_worker_count,
                )
            )
            continue
        if detail_kind == "kubernetes":
            detail = backend.detail.kubernetes
            if detail.total_nodes != len(detail.nodes):
                raise ValueError(
                    f"backend {backend.backend_id!r} reports {detail.total_nodes} nodes "
                    f"but includes {len(detail.nodes)} node records"
                )
            health.append(
                ExecutionHealth(
                    backend_id=backend.backend_id,
                    unit=HealthUnit.NODE,
                    healthy=sum(node.ready and node.schedulable for node in detail.nodes),
                    total=detail.total_nodes,
                )
            )
            continue
        raise ValueError(f"backend {backend.backend_id!r} has no supported health detail")

    return tuple(health)


def compare(baseline: Snapshot, latest: Snapshot, *, expect_tree_hash: str = "") -> Verdict:
    """Judge a post-restart snapshot against its pre-restart baseline.

    Concerns stop a rollout: an unreachable controller, a version that is not the
    tree that was deployed, or backend health loss above the churn tolerance.
    Queue depth and job counts move on their own as work arrives, so they are
    notes — context a human reads at the gate, not a blocker.
    """
    concerns: list[str] = []
    notes: list[str] = []

    if not latest.reachable:
        return Verdict(False, (f"controller is unreachable: {latest.error}",), ())

    if expect_tree_hash and latest.tree_hash != expect_tree_hash:
        concerns.append(f"controller runs tree {latest.tree_hash or '(unknown)'}, expected {expect_tree_hash}")
    if latest.tree_hash and latest.tree_hash == baseline.tree_hash:
        notes.append(f"tree hash is unchanged ({latest.tree_hash}) — the deploy shipped the same tree content")

    baseline_health = {(item.backend_id, item.unit): item for item in baseline.execution_health}
    latest_health = {(item.backend_id, item.unit): item for item in latest.execution_health}
    if baseline_health.keys() != latest_health.keys():
        baseline_targets = ", ".join(f"{backend_id} ({unit.value})" for backend_id, unit in baseline_health)
        latest_targets = ", ".join(f"{backend_id} ({unit.value})" for backend_id, unit in latest_health)
        concerns.append(
            f"execution health targets changed from [{baseline_targets or 'none'}] to [{latest_targets or 'none'}]"
        )

    for key in sorted(baseline_health.keys() & latest_health.keys()):
        before = baseline_health[key]
        after = latest_health[key]
        loss = before.healthy - after.healthy
        tolerance = max(1, before.healthy * HEALTH_LOSS_TOLERANCE_PERCENT // 100)
        subject = f"{before.backend_id} healthy {before.unit.value}"
        if loss > tolerance:
            concerns.append(
                f"{subject} fell from {before.healthy} to {after.healthy} "
                f"(loss {loss} exceeds churn tolerance {tolerance})"
            )
        elif loss > 0:
            notes.append(
                f"{subject} fell from {before.healthy} to {after.healthy} "
                f"(loss {loss} is within churn tolerance {tolerance})"
            )
        if after.total and after.healthy < after.total:
            notes.append(
                f"{after.total - after.healthy} of {after.total} {after.backend_id} {after.unit.value} are unhealthy"
            )

    baseline_running = baseline.jobs.get(JOB_RUNNING, 0)
    latest_running = latest.jobs.get(JOB_RUNNING, 0)
    if latest_running < baseline_running:
        notes.append(f"running jobs fell from {baseline_running} to {latest_running}")
    baseline_pending = baseline.jobs.get(JOB_PENDING, 0)
    latest_pending = latest.jobs.get(JOB_PENDING, 0)
    if latest_pending > baseline_pending:
        notes.append(f"pending jobs grew from {baseline_pending} to {latest_pending}")

    return Verdict(not concerns, tuple(concerns), tuple(notes))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def take_snapshot(cluster: str) -> Snapshot:
    """Read a controller's version, backend health, and rough job counts."""
    try:
        with connect_controller(cluster_name=cluster) as endpoint:
            with rpc_client(endpoint.url, endpoint.credentials) as client:
                info = client.get_process_status(job_pb2.GetProcessStatusRequest(max_log_lines=0)).process_info
                backends = client.list_backends(controller_pb2.Controller.ListBackendsRequest()).backends
                provenance = provenance_from_proto(info.provenance)
            with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as iris:
                jobs = {
                    name: len(iris.list_jobs(JobQuery(states=frozenset({state}), page_size=JOB_COUNT_LIMIT)).items)
                    for name, state in WATCHED_JOB_STATES
                }
        return Snapshot(
            cluster=cluster,
            captured_at=_now(),
            reachable=True,
            tree_hash=provenance.tree_hash,
            version=str(provenance),
            execution_health=backend_health(backends),
            jobs=jobs,
            working_tree_hash=get_git_sha(),
        )
    except Exception as exc:
        logger.debug("Snapshot of %s failed", cluster, exc_info=True)
        return Snapshot(cluster=cluster, captured_at=_now(), reachable=False, error=f"{type(exc).__name__}: {exc}")


def _submit_smoke_job(client: IrisClient, *, suite_id: str, role: str, command: tuple[str, ...]) -> Job:
    """Submit one minimal command job through the resource client."""
    job = client.submit(
        entrypoint=Entrypoint.from_command(*command),
        name=f"deploy-smoke-{role}-{suite_id}",
        resources=ResourceSpec(cpu=SMOKE_CPU, memory=SMOKE_MEMORY, disk=SMOKE_DISK),
        environment=EnvironmentSpec(setup_scripts=[]),
        priority_band=PriorityBand.INTERACTIVE,
    )
    click.echo(f"Submitted {role} job {job.job_id}")
    return job


def _wait_for_log_marker(
    read_logs: Callable[[], LogPage],
    *,
    marker: str,
    subject: str,
    timeout: float,
) -> None:
    """Wait for a marker through one public log-reading boundary."""

    def marker_is_visible() -> bool:
        return any(marker in entry.data for entry in read_logs().entries)

    ExponentialBackoff(initial=0.25, maximum=2).wait_until_or_raise(
        marker_is_visible,
        timeout=Duration.from_seconds(timeout),
        error_message=f"{subject} logs did not contain {marker!r} within {timeout:.0f}s",
    )
    click.echo(f"Read {marker!r} from {subject} logs")


def _verify_job_and_task_logs(job: Job, marker: str, *, timeout: float) -> None:
    """Read one marker through both the aggregate Job and exact Task APIs."""
    log_timeout = min(timeout, SMOKE_LOG_TIMEOUT)
    _wait_for_log_marker(
        lambda: job.logs(max_lines=100),
        marker=marker,
        subject=f"Job {job.job_id}",
        timeout=log_timeout,
    )
    tasks = job.tasks()
    if len(tasks) != 1:
        raise RuntimeError(f"Smoke Job {job.job_id} has {len(tasks)} Tasks, expected 1")
    task = tasks[0]
    _wait_for_log_marker(
        lambda: task.logs(max_lines=100),
        marker=marker,
        subject=f"Task {task.task_id}",
        timeout=log_timeout,
    )


def _require_job_state(status: JobSummary, expected: JobState, *, subject: str) -> None:
    if status.state is expected:
        return
    raise RuntimeError(f"{subject} ended {status.state.name}, expected {expected.name}: {status.error_message}")


def run_smoke_suite(client: IrisClient, *, timeout: float) -> SmokeSuiteResult:
    """Exercise submit, wait, logs, cancellation, action polling, and reuse.

    ``setup_scripts=[]`` keeps this focused on the control plane and task runtime.
    The first Job may pay for autoscaler scale-up. The cancellation and follow-up
    Jobs then reuse that capacity.
    """
    suite_id = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    click.echo(f"Waiting up to {timeout:.0f}s per Job — a cluster at 0 workers must scale one up first.")

    completed = _submit_smoke_job(
        client,
        suite_id=suite_id,
        role="complete",
        command=SMOKE_SUCCESS_COMMAND,
    )
    completed_status = completed.wait(
        timeout=timeout,
        poll_interval=SMOKE_POLL_INTERVAL,
        raise_on_failure=False,
    )
    _require_job_state(completed_status, JobState.SUCCEEDED, subject=f"Job {completed.job_id}")
    _verify_job_and_task_logs(completed, SMOKE_SUCCESS_MARKER, timeout=timeout)

    cancellable = _submit_smoke_job(
        client,
        suite_id=suite_id,
        role="cancel",
        command=SMOKE_CANCEL_COMMAND,
    )
    cancel_key = f"deploy-smoke-cancel-{suite_id}"
    cancellation_finished = False
    try:
        _verify_job_and_task_logs(cancellable, SMOKE_CANCEL_MARKER, timeout=timeout)
        receipt = cancellable.cancel(idempotency_key=cancel_key)
        completed_receipt = client.wait_for_action(
            receipt.action_id,
            timeout=Duration.from_seconds(min(timeout, SMOKE_ACTION_TIMEOUT)),
        )
        if completed_receipt.state is not ActionState.SUCCEEDED:
            raise RuntimeError(
                f"Cancellation {completed_receipt.action_id} ended {completed_receipt.state.value}: "
                f"{completed_receipt.result_message}"
            )
        cancelled_status = cancellable.wait(
            timeout=timeout,
            poll_interval=SMOKE_POLL_INTERVAL,
            raise_on_failure=False,
        )
        _require_job_state(cancelled_status, JobState.KILLED, subject=f"Job {cancellable.job_id}")
        cancellation_finished = True
    finally:
        if not cancellation_finished:
            cleanup = cancellable.cancel(idempotency_key=cancel_key)
            client.wait_for_action(
                cleanup.action_id,
                timeout=Duration.from_seconds(min(timeout, SMOKE_ACTION_TIMEOUT)),
            )

    followup = _submit_smoke_job(
        client,
        suite_id=suite_id,
        role="followup",
        command=SMOKE_FOLLOWUP_COMMAND,
    )
    followup_status = followup.wait(
        timeout=timeout,
        poll_interval=SMOKE_POLL_INTERVAL,
        raise_on_failure=False,
    )
    _require_job_state(followup_status, JobState.SUCCEEDED, subject=f"Job {followup.job_id}")
    _verify_job_and_task_logs(followup, SMOKE_FOLLOWUP_MARKER, timeout=timeout)

    return SmokeSuiteResult(
        completed_job=completed.job_id.to_wire(),
        cancelled_job=cancellable.job_id.to_wire(),
        followup_job=followup.job_id.to_wire(),
        cancel_action_id=completed_receipt.action_id,
    )


def run_smoke_suite_on_cluster(cluster: str, *, timeout: float) -> SmokeSuiteResult:
    """Connect to one configured cluster and run the rollout smoke suite."""
    with connect_controller(cluster_name=cluster) as endpoint:
        # These are command-only Jobs with no setup scripts or Python entrypoint.
        # Supplying a workspace would upload the whole checkout before the first
        # submit without changing what the smoke executes.
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as client:
            return run_smoke_suite(client, timeout=timeout)


@functools.cache
def _resolve_config(cluster: str) -> IrisClusterConfig:
    return load_config(str(resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS)))


def _selected_order(clusters: str | None) -> tuple[str, ...]:
    """Resolve `--clusters` to an ordered tuple, defaulting to the rollout order.

    An explicit list is used verbatim: the operator's chosen order wins over the
    default, which is what lets a human roll out one cluster or reorder a wave.
    """
    if clusters:
        return tuple(name.strip() for name in clusters.split(",") if name.strip())
    candidates = deploy_candidates(list_cluster_configs(IRIS_CLUSTER_CONFIG_DIRS))
    capacities = {name: cluster_capacity(_resolve_config(name)) for name in candidates}
    return rollout_order(capacities)


@click.group()
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Steps of an evidence-gated Iris controller rollout."""
    # Every step reports through click.echo, so the default level keeps iris's own
    # config/tunnel chatter out of a gate the operator has to read.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.WARNING,
    )


@cli.command("plan")
@click.option("--clusters", default=None, help="Comma-separated clusters, in the order to deploy them.")
def plan(clusters: str | None) -> None:
    """Print the deploy order and the image tag the restarts will ship."""
    order = _selected_order(clusters)
    click.echo(f"Working tree image tag: {get_git_sha()}")
    click.echo("Run `preflight` to check this tree against origin/main before deploying.")
    click.echo("Deploy order:")
    for position, cluster in enumerate(order, start=1):
        config = _resolve_config(cluster)
        kind = config.controller.controller_kind() or "unknown"
        # Lead clusters are pinned first, so their cap sorts nothing. Printing one
        # invites reading it as capacity, which it is not (see cluster_capacity).
        if cluster in DEPLOY_LEAD:
            click.echo(f"  {position}. {cluster} (controller={kind}, deploy-lead)")
            continue
        click.echo(f"  {position}. {cluster} (controller={kind}, max_slices cap={cluster_capacity(config)})")


@cli.command("preflight")
@click.option("--clusters", default=None, help="Comma-separated clusters to check.")
@click.option("--upstream", default=DEFAULT_UPSTREAM, show_default=True, help="Branch the tree is compared against.")
@click.option("--no-fetch", is_flag=True, help="Compare against the last fetched upstream instead of fetching it.")
@click.option(
    "--accept-tree-state",
    is_flag=True,
    help="Proceed with a dirty or diverged tree. Set this only after the operator confirms the warnings.",
)
def preflight(clusters: str | None, upstream: str, no_fetch: bool, accept_tree_state: bool) -> None:
    """Check what this tree would ship and every operator-side input the deploys consume.

    Fails when a requirement is absent, and fails on a dirty or diverged tree until
    the operator confirms it — both before any controller is touched.
    """
    order = _selected_order(clusters)

    state = read_tree_state(upstream=upstream, fetch=not no_fetch)
    click.echo(f"Working tree: {state.summary()}")
    warnings = tree_warnings(state)
    for warning in warnings:
        click.echo(f"  [WARN] {warning.issue.value}: {warning.message}")

    failures = 0
    for cluster in order:
        click.echo(f"\n{cluster}:")
        for result in check_requirements(requirements(_resolve_config(cluster)), environ=os.environ):
            mark = "PASS" if result.ok else "FAIL"
            click.echo(f"  [{mark}] {result.requirement} — {result.detail}")
            if not result.ok:
                failures += 1
                click.echo(f"         needed because: {result.requirement.why}")
    if failures:
        raise click.ClickException(f"{failures} requirement(s) are missing. Supply them before deploying.")
    click.echo("\nAll requirements are present.")

    # A dirty tree is the operator's to approve, not the reader of this output.
    if warnings and not accept_tree_state:
        raise click.ClickException(
            "The working tree is not a clean checkout of "
            f"{state.upstream}. Ask the operator whether to deploy this exact tree, "
            "then re-run with --accept-tree-state."
        )
    if warnings:
        click.echo(f"Tree state accepted by the operator: deploying tree {state.tree_hash}.")


@cli.command("snapshot")
@click.option("--cluster", required=True, help="Cluster to snapshot.")
@click.option(
    "--out", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Write the snapshot JSON here."
)
def snapshot(cluster: str, out: Path | None) -> None:
    """Capture a controller's rough state. Run this before a restart."""
    result = take_snapshot(cluster)
    click.echo(result.summary())
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.to_json())
        click.echo(f"Baseline written to {out}")
    if not result.reachable:
        raise click.ClickException("Controller is unreachable — do not restart until you know why.")


@cli.command("verify")
@click.option("--cluster", required=True, help="Cluster to watch.")
@click.option("--baseline", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--duration", default=WATCH_DURATION, show_default=True, help="Seconds to watch the controller.")
@click.option("--interval", default=WATCH_INTERVAL, show_default=True, help="Seconds between samples.")
@click.option("--expect-tree-hash", default=None, help="Tree hash the controller must run. Defaults to the baseline's.")
def verify(cluster: str, baseline: Path, duration: int, interval: int, expect_tree_hash: str | None) -> None:
    """Watch a restarted controller, then compare it against its baseline."""
    before = Snapshot.from_json(baseline.read_text())
    expected = expect_tree_hash if expect_tree_hash is not None else before.working_tree_hash
    if not expected:
        raise click.ClickException("The baseline records no tree hash. Re-take it, or pass --expect-tree-hash.")

    deadline = time.monotonic() + duration
    latest = take_snapshot(cluster)
    click.echo(f"  {latest.captured_at} {latest.summary()}")
    unreachable = 0 if latest.reachable else 1
    while time.monotonic() < deadline:
        time.sleep(min(interval, max(1.0, deadline - time.monotonic())))
        latest = take_snapshot(cluster)
        click.echo(f"  {latest.captured_at} {latest.summary()}")
        if not latest.reachable:
            unreachable += 1

    verdict = compare(before, latest, expect_tree_hash=expected)
    if unreachable:
        click.echo(f"\n{unreachable} sample(s) could not reach the controller.")
    for note in verdict.notes:
        click.echo(f"note: {note}")
    for concern in verdict.concerns:
        click.echo(f"CONCERN: {concern}")
    if not verdict.healthy or unreachable:
        raise click.ClickException("The controller does not look healthy. Stop the rollout and diagnose.")
    click.echo("Controller looks healthy.")


@cli.command("smoke")
@click.option("--cluster", required=True, help="Cluster to run the smoke job on.")
@click.option("--timeout", default=SMOKE_TIMEOUT, show_default=True, help="Seconds to wait for the job to finish.")
def smoke(cluster: str, timeout: int) -> None:
    """Run the resource lifecycle smoke suite through one configured cluster."""
    result = run_smoke_suite_on_cluster(cluster, timeout=timeout)
    click.echo(f"Smoke suite passed: {result.summary()}")


@cli.command("smoke-local")
@click.option("--timeout", default=120, show_default=True, help="Seconds to wait for each Job.")
def smoke_local(timeout: int) -> None:
    """Run the resource lifecycle smoke suite on a scratch local cluster."""
    with local_client(LocalClientConfig(max_workers=1)) as client:
        result = run_smoke_suite(client, timeout=timeout)
    click.echo(f"Local smoke suite passed: {result.summary()}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
