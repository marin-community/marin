# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preflight for an unattended queue run: six checks with the real run config, failing fast.

Run it inside an Iris task on the target cluster, with the run's own config (``run.json``, the
committed example with ``relay_job`` set) and the shipped libraries (no in-process patch). It
writes ``probe.json`` into ``$IRIS_OUTPUT_DIR/queue_probe`` and exits non-zero at the first failed
check. Probes drive validation themselves, so they run on the interactive pool: ``--pool high``
replaces the config's pool for every request the probe sends, while check 1 also requires workers in
the config's own pool::

    lib/taskforge/.venv/bin/iris --cluster=marin job run --no-wait --no-sync \\
      --job-name taskforge-queue-probe-NN --target-cluster cw-rno2a --priority interactive \\
      --cpu 4 --memory 8GB --timeout 7200 -e GLM_API_TOKEN "$GLM_API_TOKEN" -e PARALLEL_KEY "$PARALLEL_KEY" -- \\
      bash -c 'cd lib/taskforge && uv run --frozen python scripts/cluster_queue_probe.py \\
        run.json --pool high --items 256 \\
        --image docker.io/library/python@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d'

``GLM_API_TOKEN`` must then hold the interactive (``high``) token. With a ``host: laptop`` config the
same checks rehearse on ShellSim and the laptop's Docker; the Finelog check reports that there is no
Iris context.

Checks:

1. ``glm``: the endpoint resolves (through the relay on Iris) and ``/health`` reports workers for the
   probe's pool and the config's pool.
2. ``machines``: each factory the run has creates and closes a machine, and no credential-shaped
   environment variable reaches a docker sandbox.
3. ``width``: ``--items`` concurrent validation rounds of a null-environment math task (controls,
   then one solver trial each) complete with ``Complete`` evidence; the TRIAL ledger rows show the
   concurrency reached and the wall time, and the solver attempt files show the requests sent, the
   retried attempts by outcome and HTTP status (429s among them) and the time held in retried attempts.
4. ``finelog``: the ledger mirror connected and confirmed its flush of the EVENT and TRIAL rows that
   checks 3 and 5 wrote, so it runs after check 5. Read the rows back after the job ends with the
   ``query-finelog`` SQL printed as ``FINELOG_QUERY``.
5. ``resume``: ``--items`` seeded items run through ``queue.run.run_queue`` (a rubric that rejects
   without a model call) at the config's width; ``derive_state`` over the ledger reproduces the
   summary's terminal map, and with ``restore_from`` set the restored terminal items are skipped (and
   check 3 loads its settled trials from the restored attempt files instead of running them).
6. ``image``: ``--image`` is pullable by a sandbox on this host (reported, not fixed).
"""

import argparse
import asyncio
import json
import os
import time
import traceback
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any

import httpx
from shellbox.image import RegistryImage as ShellboxRegistryImage
from shellbox.machine import Command, MachineSpec, NetworkPolicy, ShellSimBuiltins
from taskcompendium.environment import EnvironmentKind
from taskcompendium.execution import TaskExecution
from taskcompendium.grading import numeric_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, Source
from taskcompendium.submission import PlainText

from taskforge.build.run import Provenance, TaskDraft
from taskforge.ledger.finelog import LEDGER_NAMESPACE, CompositeLedger, FlushResult, connect_finelog_ledger
from taskforge.ledger.jsonl import JsonlLedger, ledger_files, read_entries
from taskforge.ledger.records import EntryKind, Ledger
from taskforge.llm.client import AttemptOutcome, FinishReason, GlmClient, GlmEndpoint, Pool, Usage
from taskforge.llm.endpoint import API_ROOT
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.loop.program import LEDGER_DIR
from taskforge.proposal.model import TaskProposal, parse
from taskforge.proposal.source import ProposalBatch, SlotProposal
from taskforge.queue.config import RunConfig, load_run_config
from taskforge.queue.job import (
    IRIS_OUTPUT_DIR_ENV,
    HostSecrets,
    RunInputs,
    controller_url,
    glm_endpoint,
    host_secrets,
    loop_services,
    prepare_root,
    secret_names,
)
from taskforge.queue.run import FailedItems, item_terminal, run_queue
from taskforge.sandbox.factories import MachineHost, factory_capabilities, machine_factories
from taskforge.spec.controls import Control, ControlCategory, ControlConcern, ControlKind, Expectation, Transcript, reply
from taskforge.spec.draft import assemble, environment
from taskforge.triage.checks import ALL_COMBINATIONS, CheckContext
from taskforge.triage.program import RubricAssessment
from taskforge.triage.verdict import ModelCall, RubricAxis, RubricResult, TriageDecision
from taskforge.validate.attempts import load_outcome
from taskforge.validate.controls import ServerTokenizer
from taskforge.validate.evidence import Complete, Evidence
from taskforge.validate.outcome import TrialKind
from taskforge.validate.run import controls_passed, replay_controls
from taskforge.validate.solver import ValidationSite, run_solver

MATH_ANSWER = "395"
# Prints names only; `env | cut` would leak fragments of multi-line values into evidence.
ENV_NAMES_SCRIPT = "awk 'BEGIN{for(k in ENVIRON) print k}' | sort | tr '\\n' ' '"
HEALTH_TIMEOUT = 20.0
PROBE_DIR = "queue_probe"
PROBE_PROPOSAL = """---
id: "probe/SLOT"
source: {kind: capability, ref: "probe", hash: "probe"}
environment: reasoning
verification: simple
grounding: unverified
research: []
build:
  - "grader": "checks the number"
resources: []
null_reason: null
---
## Task
Compute 17 * 23 + 4.

## Realism and workflow
A probe item; the probe's rubric rejects it without a model call.

## Research plan
None.

## Build plan
None.

## Grader design and controls
Exact number.

## Risks and null conditions
None.
"""
REJECT_SAMPLE = RubricResult(tuple((axis, 1) for axis in RubricAxis), ("probe item",), (), (), TriageDecision.REJECT)
NO_CALL = ModelCall(Usage(0, 0, 0, 0), wall_time=0.0, finish_reason=FinishReason.STOP)


class CheckFailed(Exception):
    """A probe check's condition does not hold; the message says which."""


@dataclass
class Probe:
    config: RunConfig
    pool: Pool
    items: int
    image: str
    report: dict[str, Any] = field(default_factory=dict)


def math_draft(index: int) -> TaskDraft:
    task = assemble(
        f"probe-math-{index}",
        "What is 17 * 23 + 4? Reply with only the number, nothing else.",
        AnswerType.NUMBER,
        environment(EnvironmentKind.NULL),
        numeric_answer(MATH_ANSWER, tolerance_abs=0, tolerance_rel=0),
        Source(dataset="taskforge-queue-probe", revision="1", row=str(index), importer_revision="1"),
        execution=TaskExecution(),
    )
    correct = Expectation(status=Outcome.GRADED, reward_min=1.0)
    wrong = Expectation(status=Outcome.GRADED, reward_max=0.0)
    controls = (
        Control(
            "correct",
            ControlKind.POSITIVE,
            ControlCategory.KNOWN_CORRECT,
            ControlConcern.REFERENCE,
            "probe",
            Transcript((reply("395"),)),
            correct,
        ),
        Control(
            "wrong",
            ControlKind.NEGATIVE,
            ControlCategory.PLAUSIBLE_WRONG,
            ControlConcern.ACCEPTANCE,
            "probe",
            Transcript((reply("391"),)),
            wrong,
        ),
        Control(
            "two-answers",
            ControlKind.NEGATIVE,
            ControlCategory.TASK_SPECIFIC_SHORTCUT,
            ControlConcern.SHORTCUT,
            "probe",
            Transcript((reply("395 or 391"),)),
            wrong,
        ),
        Control(
            "empty",
            ControlKind.MALFORMED,
            ControlCategory.EMPTY_OR_MALFORMED,
            ControlConcern.EXTRACTION,
            "probe",
            Transcript((reply(""),)),
            Expectation(status=Outcome.SUBMISSION_FAILURE),
        ),
    )
    provenance = Provenance(task.id, "probe", "probe", "probe", "probe", "probe", 0, (), ())
    return TaskDraft(task, TaskExecution(), PlainText(id="plain_text"), controls, provenance)


def max_concurrency(spans: list[tuple[float, float]]) -> int:
    edges = sorted([(start, 1) for start, _ in spans] + [(end, -1) for _, end in spans], key=lambda e: (e[0], e[1]))
    running = peak = 0
    for _, step in edges:
        running += step
        peak = max(peak, running)
    return peak


def request_attempts(evidence_root: Path) -> dict[str, Any]:
    """The GLM request attempts recorded in the solver rollouts' turn metadata under ``evidence_root``."""
    attempts = [
        attempt
        for path in sorted(evidence_root.rglob("attempt-*.json"))
        if (rollout := load_outcome(path).rollout) is not None
        for step in rollout.steps
        for attempt in step.turn.metadata.get("attempts", ())
    ]
    retried = [a for a in attempts if a["outcome"] != AttemptOutcome.COMPLETED]
    return {
        "completed_attempts": sum(1 for a in attempts if a["outcome"] == AttemptOutcome.COMPLETED),
        "retried_attempts": len(retried),
        "retried_by_outcome": dict(Counter(a["outcome"] for a in retried)),
        "retried_by_status": {str(k): v for k, v in Counter(a["http_status"] for a in retried).items()},
        "status_429": sum(1 for a in retried if a["http_status"] == 429),
        "retry_hold_time": sum(a["duration"] for a in retried),
    }


async def check_glm(probe: Probe, endpoint: GlmEndpoint) -> dict[str, Any]:
    async with httpx.AsyncClient() as http:
        response = await http.get(f"{endpoint.base_url.removesuffix(API_ROOT)}/health", timeout=HEALTH_TIMEOUT)
    body = response.json()
    workers = body.get("workers") or {}
    report = {"base_url": endpoint.base_url, "status": response.status_code, "workers": workers}
    for pool in {probe.pool, probe.config.glm.pool}:
        if not workers.get(str(pool)):
            raise CheckFailed(f"/health reports no workers in the {pool} pool: {body}")
    return report


async def check_machines(probe: Probe) -> dict[str, Any]:
    factories = machine_factories(probe.config.host, controller_url(probe.config.host), probe.config.image_cache)
    report: dict[str, Any] = {}
    for kind, factory in factories.items():
        started = time.monotonic()
        if kind is EnvironmentKind.SHELLSIM:
            machine = await factory.create(MachineSpec(source=ShellSimBuiltins()))
            await machine.close()
            report[str(kind)] = {"created_and_closed": True, "seconds": time.monotonic() - started}
            continue
        spec = MachineSpec(source=ShellboxRegistryImage(probe.image), network=NetworkPolicy.ALLOW)
        machine = await factory.create(spec)
        try:
            result = await machine.run(Command(argv=("sh", "-c", ENV_NAMES_SCRIPT), timeout=60))
        finally:
            await machine.close()
        names = result.stdout.decode().split()
        leaked = secret_names(names)
        report[str(kind)] = {"created_and_closed": True, "seconds": time.monotonic() - started, "secret_names": leaked}
        if leaked:
            raise CheckFailed(f"SANDBOX_SECRETS credential-like names reach {kind} sandboxes: {leaked}")
    return report


async def check_width(probe: Probe, endpoint: GlmEndpoint, ledger: Ledger, root: Path) -> dict[str, Any]:
    config = probe.config
    validation = replace(config.policy.validation, k=1)
    factories = machine_factories(config.host, controller_url(config.host), config.image_cache)
    settings = config.engine.settings(factories, factory_capabilities(config.host))
    slots = asyncio.Semaphore(config.width)
    async with GlmClient(endpoint) as client:
        models = partial(GlmRolloutModel, client, validation.sampling)
        tokenize = ServerTokenizer(client, validation.sampling)

        async def item(index: int) -> tuple[bool, Evidence]:
            draft = math_draft(index)
            site = ValidationSite(f"probe-width-{index}", 0, root / "width" / str(index), ledger)
            async with slots:
                controls = await replay_controls(draft, validation, site, settings, tokenize)
                solver = await run_solver(draft, validation, site, settings, models)
            evidence = Evidence({TrialKind.CONTROL: tuple(c.outcome for c in controls), TrialKind.SOLVER: solver})
            return controls_passed(controls), evidence

        started = time.monotonic()
        rounds = await asyncio.gather(*(item(index) for index in range(probe.items)))
        wall = time.monotonic() - started
    trials = [
        entry
        for path in ledger_files(root / LEDGER_DIR)
        for entry in read_entries(path)
        if entry.kind is EntryKind.TRIAL and entry.item_id.startswith("probe-width-")
    ]
    incomplete = [index for index, (_, evidence) in enumerate(rounds) if not isinstance(evidence.status, Complete)]
    violated = [index for index, (passed, _) in enumerate(rounds) if not passed]
    report = {
        "items": probe.items,
        "width": config.width,
        "wall_time": wall,
        "trial_rows": len(trials),
        "concurrency_reached": max_concurrency([(entry.started, entry.ended) for entry in trials]),
        "incomplete": incomplete,
        "controls_not_met": violated,
        "solved": sum(evidence.reward_stats(TrialKind.SOLVER).solved for _, evidence in rounds),
        **request_attempts(root / "width"),
    }
    if incomplete:
        raise CheckFailed(f"{len(incomplete)} of {probe.items} rounds left trials ungraded: {incomplete[:20]}")
    return report


@dataclass
class RejectingRubric:
    """Rejects every proposal without a model call and records which it saw."""

    assessed: list[str] = field(default_factory=list)

    async def assess(self, p: TaskProposal, structural: object) -> RubricAssessment:
        self.assessed.append(p.header.id)
        return RubricAssessment((REJECT_SAMPLE,), (NO_CALL,))

    async def repair(self, p: TaskProposal, verdict: object) -> Any:
        raise AssertionError("the probe rubric rejects; it never repairs")


@dataclass(frozen=True)
class SeedSource:
    """``n`` copies of the probe proposal for the one idea, without a model call."""

    async def propose(self, idea: str, n: int) -> ProposalBatch:
        slots = tuple(
            SlotProposal(slot, parse(PROBE_PROPOSAL.replace("SLOT", str(slot))), (), (), None) for slot in range(n)
        )
        return ProposalBatch((), (), slots)


def describe_probe(idea: str) -> dict[str, object]:
    return {"idea": idea}


def no_adversary_context(proposal: TaskProposal) -> str:
    """The probe adds nothing to the adversary brief; its rubric rejects every item before trials."""
    return ""


async def check_resume(
    probe: Probe, endpoint: GlmEndpoint, secrets: HostSecrets, ledger: Ledger, root: Path
) -> dict[str, Any]:
    ledger_dir = root / LEDGER_DIR
    restored = {f"probe--{slot}": item_terminal(ledger_dir, f"probe--{slot}") for slot in range(probe.items)}
    rubric = RejectingRubric()

    def inputs(client: GlmClient, run_root: Path) -> RunInputs[str]:
        return RunInputs(
            ideas={"probe": "probe"},
            source=SeedSource(),
            describe_idea=describe_probe,
            adversary_context=no_adversary_context,
            checks=(),
            rubric=rubric,
            check_context=CheckContext(ALL_COMBINATIONS),
        )

    policy = replace(probe.config.policy, proposals_per_idea=probe.items)
    config = replace(probe.config, policy=policy)
    async with loop_services(config, endpoint, secrets, root, ledger, inputs) as (services, ideas):
        summary = await run_queue(ideas, policy, services, FailedItems.SKIP)
    derived = {item: item_terminal(ledger_dir, item) for item in summary.items}
    skipped = [item for item, terminal in restored.items() if terminal is not None]
    rerun = [item for item in skipped if item.replace("--", "/") in rubric.assessed]
    report = {
        "summary": summary.summary_json(),
        "derived_matches_summary": derived == dict(summary.items),
        "restored_terminal_items": len(skipped),
        "restored_items_rerun": rerun,
    }
    if derived != dict(summary.items):
        raise CheckFailed("derive_state over the ledger does not reproduce the run's terminal map")
    if rerun:
        raise CheckFailed(f"restored terminal items ran again: {rerun[:20]}")
    if summary.failed:
        raise CheckFailed(f"items failed: {summary.failed}")
    return report


async def check_finelog(remote_flush: FlushResult | None, run_id: str) -> dict[str, Any]:
    if remote_flush is None:
        return {"applicable": False, "reason": "no in-cluster Iris context; the ledger stayed local"}
    query = (
        f"SELECT kind, count(*) FROM \"{LEDGER_NAMESPACE}\" WHERE run_id = '{run_id}' "
        "AND kind IN ('event', 'trial') GROUP BY kind"
    )
    print(f"FINELOG_QUERY {query}", flush=True)
    if remote_flush is not FlushResult.SUCCEEDED:
        raise CheckFailed(f"the Finelog mirror did not confirm its flush: {remote_flush.value}")
    return {"applicable": True, "flush": remote_flush.value, "query": query}


async def check_image(probe: Probe) -> dict[str, Any]:
    factories = machine_factories(probe.config.host, controller_url(probe.config.host), probe.config.image_cache)
    factory = factories.get(EnvironmentKind.DOCKER)
    if factory is None:
        raise CheckFailed(f"{probe.config.host} has no docker factory to pull {probe.image}")
    started = time.monotonic()
    machine = await factory.create(MachineSpec(source=ShellboxRegistryImage(probe.image), network=NetworkPolicy.ALLOW))
    await machine.close()
    return {"image": probe.image, "pulled": True, "seconds": time.monotonic() - started}


async def run_check(probe: Probe, name: str, check: Callable[[], Awaitable[dict[str, Any]]]) -> None:
    print(f"CHECK_START {name}", flush=True)
    started = time.monotonic()
    try:
        result = await check()
    except Exception as error:
        probe.report[name] = {"ok": False, "error": repr(error)[:4000], "trace": traceback.format_exc()[-4000:]}
        raise
    probe.report[name] = {"ok": True, "seconds": time.monotonic() - started, **result}
    print(f"CHECK_OK {name} {json.dumps(probe.report[name], default=str)[:2000]}", flush=True)


async def run_probe(probe: Probe) -> None:
    probe_glm = replace(probe.config.glm, pool=probe.pool)
    secrets = host_secrets(probe.config)
    endpoint = glm_endpoint(replace(probe.config, glm=probe_glm), secrets.glm_token)
    root = prepare_root(probe.config)
    local = JsonlLedger(root / LEDGER_DIR)
    remote = connect_finelog_ledger(probe.config.run_id) if probe.config.host is MachineHost.IRIS else None
    ledger: Ledger = local if remote is None else CompositeLedger(local, remote)
    flushed: FlushResult | None = None
    try:
        await run_check(probe, "glm", lambda: check_glm(probe, endpoint))
        await run_check(probe, "machines", lambda: check_machines(probe))
        await run_check(probe, "width", lambda: check_width(probe, endpoint, ledger, root))
        await run_check(probe, "resume", lambda: check_resume(probe, endpoint, secrets, ledger, root))
    finally:
        if remote is not None:
            flushed = remote.close()
    await run_check(probe, "finelog", lambda: check_finelog(flushed, probe.config.run_id))
    await run_check(probe, "image", lambda: check_image(probe))


def results_dir(config: RunConfig) -> Path:
    if config.host is MachineHost.LAPTOP:
        return probe_root(config)
    output_dir = os.environ.get(IRIS_OUTPUT_DIR_ENV)
    if not output_dir:
        raise SystemExit(f"{IRIS_OUTPUT_DIR_ENV} is unset; the probe writes into the attempt's output dir")
    return Path(output_dir) / PROBE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--pool", type=Pool, choices=list(Pool), required=True, help="the pool the probe's requests use")
    parser.add_argument("--items", type=int, required=True, help="concurrent probe items (the run's width)")
    parser.add_argument("--image", required=True, help="a registry image the run will use")
    args = parser.parse_args()
    config = load_run_config(args.config)
    results = results_dir(config)
    results.mkdir(parents=True, exist_ok=True)
    probe = Probe(
        config=replace(config, root=probe_root(config)),
        pool=args.pool,
        items=args.items,
        image=args.image,
    )
    started = time.monotonic()
    probe.report["ok"] = False
    try:
        asyncio.run(run_probe(probe))
        probe.report["ok"] = True
    except Exception as error:
        probe.report["error"] = {"error": repr(error)[:4000], "trace": traceback.format_exc()[-4000:]}
        raise
    finally:
        probe.report["wall_time"] = time.monotonic() - started
        (results / "probe.json").write_text(json.dumps(probe.report, indent=1, default=str))
        print("PROBE_REPORT " + json.dumps(probe.report, default=str), flush=True)


def probe_root(config: RunConfig) -> Path:
    """The probe's run root beside the run's own, so a probe never writes into a real run."""
    return config.root.with_name(f"{config.root.name}-{PROBE_DIR}")


if __name__ == "__main__":
    main()
