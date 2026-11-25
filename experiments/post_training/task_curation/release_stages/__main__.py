# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``atlas-release``: the single entry point that builds an Atlas release (R40 gate).

One command drives the whole release graph (DESIGN.md §1):

    atlas-release --version <release-id> --plan <release-plan.json> <mode>

In-region runs (an Iris bundle is a copied workspace, not a git checkout) pass
``--source-revision <landed-sha>``: the entry point validates it against the plan and
against the bundled landing log, and every helper that would read the checkout's HEAD
uses it instead.

Modes:

- ``release``    — derive the release plan from the reviewed config and authoritative
                   state (``--dry-run`` prints the derived summary, the lowered graph
                   and the preflight results; the sealing path composes submission once
                   the P0 ``submit/resume --cluster`` mode lands).
- ``plan``      — resolve the plan, lower the graph and print every node; I/O-free.
- ``preflight`` — per-stage type-check + small-slice run on real inputs; seals a
                  receipt per stage under the release prefix. Stages whose real
                  write-mode upstream receipts do not exist yet are reported as
                  blocked (with the reason) instead of failing.
- ``run``       — the submission path: the guard refuses anything not landed on
                  ``atlas-env-audit`` or carrying release literals, every stage gates
                  on its fresh preflight receipt, then the lazy graph builds the
                  release (cluster stages through ``marin.execution.remote``).
- ``status``    — per-node artifact and preflight state.
- ``resume``    — a rerun: completed stages are skipped (matching fingerprints),
                  failed stages restart alone.

Nothing outside the repository is read or executed; the plan document (or the
release config the plan stage derives from) is the only release configuration, and
release-identifying values never appear in stage source (``guard.py`` enforces it).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from marin.execution.lazy import lower
from marin.execution.lazy import run as run_handles
from marin.execution.step_runner import StepRunner
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.step_status import read_step_status
from rigging.timing import Duration

from experiments.post_training.task_curation.atlas_deployment import ATLAS_SERVICE
from experiments.post_training.task_curation.release_stages import guard, preflight_slices
from experiments.post_training.task_curation.release_stages import preflight as preflight_module
from experiments.post_training.task_curation.release_stages.graph import (
    REPO_ROOT,
    STAGE_CHILD_ENV_KEYS,
    PreflightPolicy,
    ReleaseGraph,
    ReleasePlan,
    graph_nodes,
    load_plan,
    release_graph,
)
from experiments.post_training.task_curation.release_stages.guard import assert_submission_allowed, repo_head
from experiments.post_training.task_curation.release_stages.paced_query import RetryControl, open_paced_query
from experiments.post_training.task_curation.release_stages.plan_derivation import (
    derive_release_plan,
    seal_derivation,
)
from experiments.post_training.task_curation.release_stages.preflight_slices import SliceSelection
from experiments.post_training.task_curation.release_stages.preflight_slices import (
    ensure_preflight as ensure_node_preflight,
)
from experiments.post_training.task_curation.release_stages.release_config import load_release_config


def _plan_document(args) -> ReleasePlan:
    if getattr(args, "plan", None) is None or getattr(args, "version", None) is None:
        raise ValueError("this mode requires --version and --plan")
    plan = load_plan(Path(args.plan))
    if args.version is not None and plan.release_id != args.version:
        raise ValueError(f"the plan binds release {plan.release_id!r}, not {args.version!r}")
    return plan


def _attest_revision(args, plan: ReleasePlan) -> None:
    """Validate and export the submitter-attested revision for the in-region bundle mode.

    The bundle is a copied workspace without git: the containment (the plan tip
    contains the plan's pinned revision) is the submitter's pre-submission
    verification, and the guard verifies the attested revision is landed through the
    bundled landing log. Where git is available (the local orchestration), the
    containment is verified here as well.
    """
    if getattr(args, "source_revision", None) is None:
        return
    revision = args.source_revision
    in_bundle = (
        not subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        == 0
    )
    if plan.source_revision != revision and not in_bundle:
        pinned = subprocess.run(
            ["git", "merge-base", "--is-ancestor", plan.source_revision, revision],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if pinned.returncode != 0:
            raise ValueError(
                f"the attested revision {revision} does not contain the plan's pinned revision {plan.source_revision}"
            )
    os.environ[guard.ATTESTED_REVISION_ENV] = revision


def _selected_nodes(graph: ReleaseGraph, stages: list[str] | None) -> list[str]:
    if not stages:
        return sorted(graph.stages)
    unknown = [stage for stage in stages if stage not in graph.stages]
    if unknown:
        raise ValueError(f"unknown graph nodes: {unknown}; known: {sorted(graph.stages)}")
    return stages


def _block_reason(node: str) -> str:
    reason = preflight_slices.BLOCK_REASONS.get(node)
    if reason is None and "[" in node:
        reason = preflight_slices.BLOCK_REASONS.get(node.split("[", 1)[0])
    return reason or "no slice runner"


def mode_plan(args) -> int:
    """Print the lowered graph; I/O-free (the runner lists the full static schedule)."""
    plan = _plan_document(args)
    graph = release_graph(plan)
    nodes = graph_nodes(graph)
    for node in nodes:
        deps = ", ".join(dep.split("atlas-release-", 1)[-1] for dep in node["deps"]) or "-"
        print(f"NODE {node['name']} deps={deps} fp={node['fingerprint'][:12]}")
    StepRunner().run([lower(graph.plan_handle), lower(graph.terminal)], dry_run=True)
    print(f"PLAN {plan.release_id} {len(nodes)} nodes, dry-run schedule listed; nothing executed")
    return 0


def mode_preflight(args) -> int:
    """Type-check and small-slice every requested stage on real inputs."""
    configure_coreweave_s3()
    plan = _plan_document(args)
    return _run_preflight(plan, source=args.source, max_files=args.max_files, stages=args.stage)


def _run_preflight(plan: ReleasePlan, *, source: str | None, max_files: int, stages: list[str] | None) -> int:
    """The preflight sweep over one plan; shared by the preflight and release modes."""
    graph = release_graph(plan)
    selection = SliceSelection(source=source, max_files=max_files)
    nodes = _selected_nodes(graph, stages)
    blocked: list[str] = []
    failed: list[str] = []
    for node in sorted(nodes):
        if node not in preflight_slices.RUNNABLE_NODES:
            blocked.append(node)
            print(f"BLOCKED {node}: {_block_reason(node)}")
            continue
        try:
            receipt = ensure_node_preflight(node, plan, selection=selection)
        except Exception as error:
            failed.append(node)
            print(f"FAIL {node}: {str(error)[:400]}")
            continue
        evidence = json.dumps(receipt.slice_evidence, sort_keys=True)[:300]
        print(f"OK {node}: {evidence}")
    if failed:
        print(f"PREFLIGHT {plan.release_id}: {len(failed)} failed, {len(blocked)} blocked")
        return 1
    print(
        f"PREFLIGHT {plan.release_id}: {len(nodes) - len(blocked)} slices ok,"
        f" {len(blocked)} blocked until their upstream stages run"
    )
    return 0


def mode_release(args) -> int:
    """Derive the release plan from the reviewed config and authoritative state.

    ``--dry-run`` derives, prints the derived plan summary, the lowered graph and
    the preflight results, and writes nothing. Without it the derivation seals the
    plan and its receipt write-once under the release prefix and then refuses the
    submission until the P0 ``submit/resume --cluster`` mode lands (the R40 lane
    owns that plumbing; this command composes it rather than duplicating it).
    """
    configure_coreweave_s3()
    config_path = args.config or Path(__file__).parent / "plans" / f"{args.release}.json"
    config = load_release_config(config_path)
    if config.release_id != args.release:
        raise ValueError(f"the config binds release {config.release_id!r}, not {args.release!r}")
    if args.cluster is not None and args.cluster != config.cluster:
        raise ValueError(f"the config pins cluster {config.cluster!r}, not {args.cluster!r}")

    def query(sql: str, parameters: dict):
        return open_paced_query(ATLAS_SERVICE.service_url, ATLAS_SERVICE.applet_id, RetryControl())(sql, parameters)

    derived = derive_release_plan(config, repo_root=REPO_ROOT, query=query)
    measured = derived.receipt["measured"]
    print(
        f"DERIVED {derived.release_id} at freeze {config.freeze_source_revision[:12]}"
        f" artifact_version={derived.receipt['artifact_version']}: sources={measured['declared_source_count']}"
        f" prior_sources={measured['prior_source_count']} targets={len(measured['targets'])}"
        f" population_tasks={measured['expected_population_tasks']} panels={measured['expected_panels']}"
        f" contract_tasks={measured['expected_contract_tasks']}"
    )
    for source_id, count in sorted(measured["population_task_counts"].items()):
        print(f"POPULATION {source_id} tasks={count} evidence={measured['population_evidence_kinds'][source_id]}")
    if not args.dry_run:
        release_root = derived.plan_document["release_root"]
        plan_ref = seal_derivation(release_root, derived)
        print(f"SEALED {plan_ref['uri']} sha256={plan_ref['sha256']}")
        raise ValueError(
            "release submission rides on the P0 submit/resume --cluster mode (the R40 lane owns"
            " that plumbing); the derived plan is sealed, and this command composes submission"
            " once that mode lands"
        )
    plan = ReleasePlan.model_validate(derived.plan_document)
    plan.validate_shape()
    for node in graph_nodes(release_graph(plan)):
        deps = ", ".join(dep.split("atlas-release-", 1)[-1] for dep in node["deps"]) or "-"
        print(f"NODE {node['name']} deps={deps} fp={node['fingerprint'][:12]}")
    print(f"RELEASE {plan.release_id}: plan derived from state; {len(plan.populations)} populations pinned")
    return _run_preflight(plan, source=args.source, max_files=args.max_files, stages=None)


def mode_run(args, *, force_run_failed: bool) -> int:
    """The submission path: guard, per-stage preflight gate, then the lazy graph."""
    configure_coreweave_s3()
    plan = _plan_document(args)
    _attest_revision(args, plan)
    assert_submission_allowed(REPO_ROOT)
    selection = SliceSelection(source=args.source, max_files=args.max_files)
    policy = PreflightPolicy(ensure=lambda node: ensure_node_preflight(node, plan, selection=selection))
    graph = release_graph(plan, preflight=policy)
    stages = _selected_nodes(graph, args.stage)
    handles: list = [graph.plan_handle]
    if args.stage:
        handles += [graph.stages[node] for node in stages]
    else:
        # An unrestricted run builds the terminal stage, which transitively
        # builds every node; the plan document is its first dependency.
        handles.append(graph.terminal)
        # The main-source sync is a plan-gated leaf outside the terminal's edges:
        # a plan that opts in runs it on every build; other plans never touch it.
        if plan.main_source_sync is not None:
            handles.append(graph.stages["main_source_sync"])
    run_handles(*handles, force_run_failed=force_run_failed, max_concurrent=args.max_concurrent)
    verb = "RESUME" if force_run_failed else "RUN"
    print(f"{verb} {plan.release_id}: {len(handles) - 1} terminal handle(s) built")
    return 0


def mode_status(args) -> int:
    """Per-node artifact and preflight state under the release prefix."""
    configure_coreweave_s3()
    plan = _plan_document(args)
    _attest_revision(args, plan)
    graph = release_graph(plan)
    landed = repo_head(REPO_ROOT)
    selection = SliceSelection(source=args.source)
    rows = []
    for node in sorted(graph.stages):
        preflight_ref = preflight_module.read_receipt(plan.release_root, node)
        state = "missing"
        if preflight_ref is not None:
            digests = preflight_module.input_digests(preflight_slices.slice_inputs(node, plan, selection))
            state = "fresh" if preflight_module.receipt_is_fresh(preflight_ref, landed, digests) else "stale"
            if not preflight_ref.ok:
                state = "failed"
        artifact = "absent"
        try:
            record = _artifact_status(graph.stages[node])
            artifact = record
        except Exception:
            artifact = "unknown-prefix"
        rows.append({"node": node, "artifact": artifact, "preflight": state})
    if args.json:
        print(json.dumps({"release_id": plan.release_id, "nodes": rows}, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(f"STATUS {row['node']} artifact={row['artifact']} preflight={row['preflight']}")
    return 0


def _artifact_status(handle) -> str:
    status = read_step_status(handle.path(marin_prefix()))
    return status or "absent"


def mode_resume(args) -> int:
    """A rerun: completed stages are skipped; failed stages restart alone.

    With ``--cluster`` the resume submits as an in-region root job (the sealed
    steps skip by fingerprint there too); without it the graph runs locally.
    """
    if getattr(args, "cluster", None):
        return _submit_root(args, verb="resume")
    return mode_run(args, force_run_failed=True)


def mode_submit(args) -> int:
    """Submit the release's root job: the graph runs in-region, auto-resuming."""
    return _submit_root(args, verb="run")


# ---------------------------------------------------------------------------
# In-region submission (the root job submits itself; no out-of-repo scripts)

_ROOT_JOB_CPU = 8
_ROOT_JOB_MEMORY_GB = 32
_ROOT_JOB_DISK_GB = 512
# Auto-resume caps: interruption retries (preemption, node loss, a killed hung
# root) rerun `resume` at the same version, so sealed steps skip; a deterministic
# code failure burns the cap and the root fails with the node's error.
_ROOT_JOB_MAX_RETRIES_FAILURE = 2
_ROOT_JOB_MAX_RETRIES_PREEMPTION = 8
# The cumulative failed-attempt cap: interruption retries (preemption, node loss, a
# killed hung root) burn it before the root fails with the node's error.
_ROOT_JOB_MAX_TASK_FAILURES = 6
_ROOT_JOB_TIMEOUT_HOURS = 24


def _plan_relative_path(plan_path: Path) -> str:
    """The plan's repo-relative path (the bundle ships the repo tree)."""
    return str(plan_path.resolve().relative_to(REPO_ROOT))


def _bundle_revision(plan: ReleasePlan, plan_path: Path) -> str:
    """The newest landed tip whose plan document is the pinned plan.

    A fresh cut bundles at its re-pin landing (the plan document inside the bundle
    carries the current artifact version); a fix-forward resume bundles at the newest
    landed fix. Both are the same rule: the newest origin tip whose plan file is
    unchanged. Git runs here, on the submitting checkout — never inside the bundle.
    """
    subprocess.run(["git", "fetch", "--quiet", "origin", "atlas-env-audit"], cwd=REPO_ROOT, check=True)
    tip = subprocess.run(
        ["git", "log", "--format=%H", "-1", "origin/atlas-env-audit"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    relative = _plan_relative_path(plan_path)
    document = json.loads(
        subprocess.run(
            ["git", "show", f"{tip}:{relative}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    if document != json.loads(plan_path.read_text()):
        raise ValueError(
            f"origin's newest tip {tip[:12]} carries a different plan document;"
            " re-pin the plan (version bump + fresh prefix) before submitting"
        )
    pinned = subprocess.run(
        ["git", "merge-base", "--is-ancestor", plan.source_revision, tip],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if pinned.returncode != 0:
        raise ValueError(f"the plan's pinned revision {plan.source_revision} is not landed on atlas-env-audit")
    return tip


def _submit_environment() -> dict[str, str]:
    """The root job's environment: the named allowlist and the credential record."""
    env = {}
    for key in STAGE_CHILD_ENV_KEYS:
        if key == "MARIN_IAP_CREDENTIAL_JSON":
            continue  # delivered from the credential record below, not the caller's env
        value = os.environ.get(key)
        if value:
            env[key] = value
    record = Path.home() / ".config/marin/credentials/marin.json"
    if record.is_file():
        env["MARIN_IAP_CREDENTIAL_JSON"] = record.read_text()
    elif os.environ.get("MARIN_IAP_CREDENTIAL_JSON"):
        env["MARIN_IAP_CREDENTIAL_JSON"] = os.environ["MARIN_IAP_CREDENTIAL_JSON"]
    return env


def _root_command(plan: ReleasePlan, plan_relative: str) -> list[str]:
    """The root job's entrypoint: the attempt-aware run/resume verb.

    Attempt 0 runs the graph; an iris-level retry (preemption, node loss, a killed
    hung root) reruns ``resume`` at the same version, so sealed steps skip and only
    failed or unstarted nodes re-run.
    """
    attempt_check = (
        "from iris.cluster.client.job_info import get_job_info;"
        " info = get_job_info(); print(getattr(info, 'attempt_id', 0) or 0)"
    )
    script = (
        'set -euo pipefail; cd "$IRIS_WORKDIR";'
        " export ATLAS_PUSH_LOG=$IRIS_WORKDIR/atlas-landing-log.jsonl;"
        " export ATLAS_WATCHDOG_FEED=$IRIS_WORKDIR/.atlas-alerts.jsonl;"
        " export UV_CACHE_DIR=$IRIS_WORKDIR/.uv-cache;"
        ' if [ -n "${MARIN_IAP_CREDENTIAL_JSON:-}" ]; then'
        " mkdir -p ~/.config/marin/credentials;"
        " printf '%s' \"$MARIN_IAP_CREDENTIAL_JSON\" > ~/.config/marin/credentials/marin.json; fi;"
        ' ATTEMPT=$(uv run --frozen python -c "' + attempt_check + '" 2>/dev/null || echo 0);'
        ' VERB=run; if [ "$ATTEMPT" -gt 0 ]; then VERB=resume; fi;'
        ' echo "root attempt $ATTEMPT -> $VERB";'
        " exec uv run --frozen atlas-release --version "
        + plan.release_id
        + " --plan "
        + plan_relative
        + " --source-revision "
        + plan.source_revision
        + ' "$VERB"'
    )
    return ["bash", "-lc", script]


def _submit_root(args, *, verb: str) -> int:
    """Submit the in-region root job that runs (or resumes) the plan's graph."""
    if getattr(args, "source_revision", None) and args.source_revision != _plan_document(args).source_revision:
        plan = _plan_document(args)
        raise ValueError(
            f"--source-revision {args.source_revision} disagrees with the plan's pinned revision"
            f" {plan.source_revision}; the pin travels with the plan"
        )
    plan = _plan_document(args)
    plan_path = Path(args.plan)
    revision = _bundle_revision(plan, plan_path)
    relative = _plan_relative_path(plan_path)

    landing_log = guard.landing_log_path()
    if not landing_log.is_file():
        raise ValueError(f"the atlas-push landing log is missing at {landing_log}; the guard verifies against it")

    snapshot = Path(tempfile.mkdtemp(prefix="atlas-release-bundle-"))
    subprocess.run(["git", "worktree", "add", "--detach", str(snapshot), revision], cwd=REPO_ROOT, check=True)
    try:
        (snapshot / "atlas-landing-log.jsonl").write_bytes(landing_log.read_bytes())
        command = _root_command(plan, relative)
        name = f"atlas-release-{plan.release_id}-{plan.artifact_version}-{verb}-{uuid.uuid4().hex[:8]}"
        with connect_controller(cluster_name=args.cluster) as endpoint:
            with IrisClient.remote(
                endpoint.url,
                credentials=endpoint.credentials,
                workspace=snapshot,
                extra_bundle_includes=["atlas-landing-log.jsonl"],
            ) as client:
                job = client.submit(
                    entrypoint=Entrypoint.from_command(*command),
                    name=name,
                    resources=ResourceSpec(
                        cpu=_ROOT_JOB_CPU, memory=f"{_ROOT_JOB_MEMORY_GB}GB", disk=f"{_ROOT_JOB_DISK_GB}GB"
                    ),
                    environment=EnvironmentSpec(env_vars=_submit_environment()),
                    priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
                    scheduling_timeout=Duration.from_hours(1),
                    timeout=Duration.from_hours(_ROOT_JOB_TIMEOUT_HOURS),
                    max_retries_failure=_ROOT_JOB_MAX_RETRIES_FAILURE,
                    max_retries_preemption=_ROOT_JOB_MAX_RETRIES_PREEMPTION,
                    max_task_failures=_ROOT_JOB_MAX_TASK_FAILURES,
                )
        job_id = str(job.job_id)
        print(f"SUBMITTED {verb} {plan.release_id}@{plan.artifact_version} bundle={revision[:12]} job={job_id}")
        return 0
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(snapshot)], cwd=REPO_ROOT, check=False)
        shutil.rmtree(snapshot, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas-release",
        description=__doc__.splitlines()[0] if __doc__ else "atlas-release",
    )
    parser.add_argument("--version", help="The release id the plan must bind")
    parser.add_argument("--plan", type=Path, help="The release plan document (JSON)")
    parser.add_argument(
        "--source-revision",
        help=(
            "The landed revision the run's workspace was bundled at (required for run/resume"
            " from a copied workspace, where the checkout revision cannot be read from git)"
        ),
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("plan", help="Print the lowered graph; I/O-free")

    release_parser = modes.add_parser("release", help="Derive the release plan from the reviewed config and dry-run it")
    release_parser.add_argument("--release", required=True, help="The release id the config must bind")
    release_parser.add_argument(
        "--cluster", help="The cluster the config must pin (the config's cluster is the default)"
    )
    release_parser.add_argument(
        "--config",
        type=Path,
        help="The release config document (default: plans/<release>.json in this package)",
    )
    release_parser.add_argument("--dry-run", action="store_true", help="Derive and print; write nothing")
    release_parser.add_argument("--source", help="The census/corpus slice source for the preflight sweep")
    release_parser.add_argument("--max-files", type=int, default=preflight_slices.DEFAULT_MAX_FILES)

    preflight_parser = modes.add_parser("preflight", help="Per-stage type-check + small-slice on real inputs")
    preflight_parser.add_argument("--stage", action="append", help="Restrict to one graph node (repeatable)")
    preflight_parser.add_argument("--source", help="The census/corpus slice source (default: first declared)")
    preflight_parser.add_argument("--max-files", type=int, default=preflight_slices.DEFAULT_MAX_FILES)

    run_parser = modes.add_parser("run", help="Guard, then build the release through the lazy graph")
    run_parser.add_argument("--stage", action="append", help="Restrict to one graph node's subtree (repeatable)")
    run_parser.add_argument("--source", help="The census/corpus slice source for auto-preflight")
    run_parser.add_argument("--max-files", type=int, default=preflight_slices.DEFAULT_MAX_FILES)
    run_parser.add_argument("--max-concurrent", type=int, default=8)

    status_parser = modes.add_parser("status", help="Per-node artifact and preflight state")
    status_parser.add_argument("--json", action="store_true")
    status_parser.add_argument("--source", help="The census/corpus slice source the receipts were sealed with")

    resume_parser = modes.add_parser("resume", help="Rerun: skip completed, restart failed alone")
    resume_parser.add_argument("--stage", action="append")
    resume_parser.add_argument("--source")
    resume_parser.add_argument("--max-files", type=int, default=preflight_slices.DEFAULT_MAX_FILES)
    resume_parser.add_argument("--max-concurrent", type=int, default=8)
    resume_parser.add_argument(
        "--cluster",
        help="Submit the resume as an in-region root job on this cluster instead of running locally",
    )

    submit_parser = modes.add_parser("submit", help="Submit the release's root job to an in-region cluster")
    submit_parser.add_argument("--cluster", required=True, help="The cluster to submit to (e.g. cw-us-east-02a)")

    args = parser.parse_args(argv)
    handlers = {
        "plan": mode_plan,
        "preflight": mode_preflight,
        "release": mode_release,
        "status": mode_status,
        "resume": mode_resume,
        "submit": mode_submit,
    }
    try:
        if args.mode == "run":
            # run threads the explicit force flag; resume sets it itself.
            return mode_run(args, force_run_failed=False)
        return handlers[args.mode](args)
    except Exception as error:
        print(f"ATLAS-RELEASE REFUSED: {str(error)[:600]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
