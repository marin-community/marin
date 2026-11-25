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

Nothing outside the repository is read or executed; the plan document is the only
release configuration, and release-identifying values never appear in stage source
(``guard.py`` enforces it).
"""

import argparse
import json
import os
import sys
from pathlib import Path

from marin.execution.lazy import lower
from marin.execution.lazy import run as run_handles
from marin.execution.step_runner import StepRunner
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.step_status import read_step_status

from experiments.post_training.task_curation.release_stages import guard, preflight_slices
from experiments.post_training.task_curation.release_stages import preflight as preflight_module
from experiments.post_training.task_curation.release_stages.graph import (
    REPO_ROOT,
    PreflightPolicy,
    ReleaseGraph,
    ReleasePlan,
    graph_nodes,
    load_plan,
    release_graph,
)
from experiments.post_training.task_curation.release_stages.guard import assert_submission_allowed, repo_head
from experiments.post_training.task_curation.release_stages.preflight_slices import SliceSelection
from experiments.post_training.task_curation.release_stages.preflight_slices import (
    ensure_preflight as ensure_node_preflight,
)


def _plan_document(args) -> ReleasePlan:
    plan = load_plan(Path(args.plan))
    if args.version is not None and plan.release_id != args.version:
        raise ValueError(f"the plan binds release {plan.release_id!r}, not {args.version!r}")
    return plan


def _attest_revision(args, plan: ReleasePlan) -> None:
    """Validate and export the submitter-attested revision for the in-region bundle mode."""
    if getattr(args, "source_revision", None) is None:
        return
    revision = args.source_revision
    if plan.source_revision != revision:
        import subprocess  # noqa: PLC0415 — the local (git) validation path

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
    graph = release_graph(plan)
    selection = SliceSelection(source=args.source, max_files=args.max_files)
    nodes = _selected_nodes(graph, args.stage)
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
    """A rerun: completed stages are skipped; failed stages restart alone."""
    return mode_run(args, force_run_failed=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas-release",
        description=__doc__.splitlines()[0] if __doc__ else "atlas-release",
    )
    parser.add_argument("--version", required=True, help="The release id the plan must bind")
    parser.add_argument("--plan", required=True, type=Path, help="The release plan document (JSON)")
    parser.add_argument(
        "--source-revision",
        help=(
            "The landed revision the run's workspace was bundled at (required for run/resume"
            " from a copied workspace, where the checkout revision cannot be read from git)"
        ),
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("plan", help="Print the lowered graph; I/O-free")

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

    args = parser.parse_args(argv)
    handlers = {
        "plan": mode_plan,
        "preflight": mode_preflight,
        "status": mode_status,
        "resume": mode_resume,
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
