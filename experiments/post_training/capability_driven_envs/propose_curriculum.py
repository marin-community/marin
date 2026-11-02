# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Propose tasks for every capability of a catalog with ``CapabilitySource``, and triage each with the preflight.

One process runs one shard: the catalog's capability ids, sorted, taken every ``--shards``-th from
``--shard``. Capabilities run through a window and every model request through a limit on in-flight
requests. The limit starts at ``--concurrency`` and follows ``OUTPUT_ROOT/control.json``, polled every
``CONTROL_POLL`` seconds; the window follows the limit (``capability_window``). A lowered limit drains;
no request is cancelled. The control file is
``{"concurrency_per_shard": N, "written_at": "<ISO 8601 with zone>", "reason": "..."}``, or
``{"pause": true, "written_at": ...}``, which lets no new request start. A missing, unreadable, refused
(below 1) or stale (older than ``CONTROL_STALE``) control file holds the current limit and is logged.

Outputs go under ``OUTPUT_ROOT`` (a local path or an ``s3://`` URL), per capability:

* ``plan.request.json``, ``plan.completions.json`` and ``plan.json``, once the plan is valid;
* per slot, the moment it finishes, ``slot-NN.request.json`` and ``slot-NN.completions.json``, then
  ``slot-NN.proposal.json`` and ``slot-NN.verdict.json`` for a proposal or ``slot-NN.failure.txt``;
* ``result.json``, written last: each slot's decision or final failure (cut off, or no valid document
  after ``MAX_REPAIRS`` repairs), the token usage and the run's provenance. A capability with
  ``result.json`` is done and a rerun skips it. At startup the shard refuses to run when any
  ``result.json`` under the root records a provenance other than this run's;
* ``failure.json`` for a capability whose plan failed, whose slot ran out of model retries, or whose
  relay or storage calls failed, with the error kind. A rerun retries it.

The shard rewrites ``shards/progress-NNNN.json`` every ``PROGRESS_INTERVAL`` seconds and writes
``shards/shard-NNNN.json`` when its queue drains. When capability failures outnumber completions in the
last ``BREAKER_WINDOW`` seconds, with at least ``BREAKER_MIN_FAILURES`` failures, the shard takes no new
capability and records why under ``halted``. It exits 1 if any capability failed or it halted, so an
Iris retry resumes the rest. Inside an Iris task::

    python -m experiments.post_training.capability_driven_envs.propose_curriculum \\
        --catalog s3://.../curriculum.json --output-root s3://.../run --shard 0 --shards 8 \\
        --concurrency 40 --relay-job /muchanem/glm53-relay-rno2a-g-20261027T0414Z --git-sha <sha>
"""

import asyncio
import hashlib
import json
import logging
import math
import tempfile
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import click
import fsspec
from pydantic import BaseModel
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from taskforge.capability import FILE_SAFE_ID, CapabilityIdea
from taskforge.content_hash import pretty_json
from taskforge.llm.client import Completion, ModelClient, ModelRequestRejected, ModelUnavailable, StructuredResult, Usage
from taskforge.llm.glm_client import EndpointSource, GlmClient, Pool
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError, response_format
from taskforge.stage10_proposal.model import (
    Environment,
    ResearchNeed,
    SolverMachine,
    Verification,
    document_request_fields,
    render,
)
from taskforge.stage10_proposal.source import Portfolio, SlotFailure, SlotOutcome, SlotProposal, proposal_batch
from taskforge.stage20_triage.preflight import preflight
from taskforge.stage20_triage.verdict import Verdict

from experiments.post_training.capability_driven_envs.catalog import load_capability_ideas
from experiments.post_training.capability_driven_envs.generate_tasks import GLM_PROVIDER, endpoint_source
from experiments.post_training.capability_driven_envs.proposal_source import (
    CUT_OFF,
    VERIFICATION_RULE,
    CapabilitySource,
    PlannedSlot,
    SlotPlan,
    SlotStatus,
    plan_prompt,
    proposal_prompt,
    slot_plan_type,
    source_ref,
    system_prompt,
)

logger = logging.getLogger(__name__)

CONTROL_FILE = "control.json"
CONTROL_KEY = "concurrency_per_shard"
CONTROL_POLL = 15.0
CONTROL_STALE = 300.0
MISSING_CONTROL_LOG_INTERVAL = 60.0
PROGRESS_INTERVAL = 60.0
STORAGE_TIMEOUT = 60.0
STORAGE_READ_BATCH = 256
MAX_REPAIRS = 4
BREAKER_WINDOW = 1800.0
BREAKER_MIN_FAILURES = 10
DEFAULT_MAX_TOKENS = 131072
STACK_POLICY = "stack_verification_rule"
RESULT_FILE = "result.json"
FAILURE_FILE = "failure.json"
SHARDS_DIR = "shards"
REPAIRS_EXHAUSTED = f"no valid document after {MAX_REPAIRS} repairs"


class RepairsExhausted(ModelUnavailable):
    """A slot's document stayed invalid through ``MAX_REPAIRS`` repairs; ``author_proposal`` makes it a slot failure."""

    def __init__(self) -> None:
        super().__init__(REPAIRS_EXHAUSTED, ())


class SlotModelErrors(Exception):
    """Slots of a capability ran out of model retries or were rejected, so the capability is not final."""

    def __init__(self, failures: Sequence[SlotFailure]):
        super().__init__("; ".join(f"slot {f.slot}: {f.error}" for f in failures))


class ProvenanceMismatch(Exception):
    """The output root holds results of a different run configuration."""


CAPABILITY_ERRORS = (
    StructuredOutputError,
    ModelUnavailable,
    ModelRequestRejected,
    SlotModelErrors,
    RuntimeError,
    OSError,
)
"""What fails one capability rather than the shard: a plan that stayed invalid or was cut off, model calls out of
retries or rejected, a relay that cannot be resolved (``RuntimeError``), and storage errors and timeouts."""


class SlotFailureKind(StrEnum):
    CUT_OFF = "cut_off"
    REPAIRS_EXHAUSTED = "repairs_exhausted"
    MODEL_ERROR = "model_error"


FINAL_SLOT_FAILURES = frozenset({SlotFailureKind.CUT_OFF, SlotFailureKind.REPAIRS_EXHAUSTED})
"""Slot failures of the content, which a rerun would repeat; a model error is infrastructure and is retried."""


def slot_failure_kind(failure: SlotFailure) -> SlotFailureKind:
    if failure.error == CUT_OFF:
        return SlotFailureKind.CUT_OFF
    if failure.error.endswith(REPAIRS_EXHAUSTED):
        return SlotFailureKind.REPAIRS_EXHAUSTED
    return SlotFailureKind.MODEL_ERROR


class RequestLimit:
    """A limit on how many holders are inside at once that can change while holders wait or run."""

    def __init__(self, limit: int):
        self.limit = limit
        self.in_flight = 0
        self.waiting = 0
        self._changed = asyncio.Condition()

    async def set(self, limit: int) -> None:
        async with self._changed:
            self.limit = limit
            self._changed.notify_all()

    async def __aenter__(self) -> None:
        async with self._changed:
            self.waiting += 1
            try:
                await self._changed.wait_for(lambda: self.in_flight < self.limit)
            finally:
                self.waiting -= 1
            self.in_flight += 1

    async def __aexit__(self, *exc: object) -> None:
        async with self._changed:
            self.in_flight -= 1
            self._changed.notify_all()


def capability_window(concurrency: int, slots: int) -> int:
    """How many capabilities run at once: enough slot requests to fill ``concurrency`` twice over."""
    return 2 * math.ceil(concurrency / slots) + 2


@dataclass
class Meter:
    """Counts of finished requests and their token usage, and of the outcomes of the shard."""

    requests: int = 0
    usage: Usage = field(default_factory=lambda: Usage(0, 0, 0, 0))
    finish_reasons: Counter[str] = field(default_factory=Counter)
    outcomes: Counter[str] = field(default_factory=Counter)
    slot_failures: Counter[str] = field(default_factory=Counter)
    repairs: Counter[str] = field(default_factory=Counter)
    last_result_at: float | None = None

    def add(self, completions: Sequence[Completion]) -> None:
        for c in completions:
            self.requests += 1
            self.usage = self.usage + c.usage
            self.finish_reasons[str(c.finish_reason)] += 1


class LimitedModel:
    """A ``ModelClient`` that holds each logical call under ``limit``, meters it, and caps document repairs.

    A document request carries one assistant turn per repair (``author_proposal`` appends the invalid reply
    and the parse error), so a request with more than ``MAX_REPAIRS`` raises ``RepairsExhausted`` unsent.
    """

    def __init__(self, model: ModelClient, limit: RequestLimit, meter: Meter):
        self._model = model
        self._limit = limit
        self._meter = meter

    @property
    def model(self) -> str:
        return self._model.model

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        if sum(m["role"] == "assistant" for m in messages) > MAX_REPAIRS:
            raise RepairsExhausted()
        async with self._limit:
            completion = await self._model.complete(messages, policy, request_fields)
        self._meter.add((completion,))
        return completion

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        async with self._limit:
            try:
                result = await self._model.complete_structured(messages, output_type, policy)
            except StructuredOutputError as error:
                self._meter.add(error.completions)
                raise
        self._meter.add(result.completions)
        return result


def shard_ids(ideas: Mapping[str, object], shard: int, shards: int) -> list[str]:
    """The capability ids of ``shard``: every ``shards``-th id of the sorted catalog, from index ``shard``."""
    return sorted(ideas)[shard::shards]


class OutputRoot:
    """The run's output tree, a local directory or an object-store prefix; every call is bounded by
    ``STORAGE_TIMEOUT`` and raises ``TimeoutError`` (an ``OSError``) past it."""

    def __init__(self, url: str):
        self.fs, self.root = fsspec.core.url_to_fs(url.rstrip("/"))

    def path(self, *parts: str) -> str:
        return "/".join((self.root, *parts))

    async def _bounded[T](self, call: Callable[..., T], *args: object) -> T:
        return await asyncio.wait_for(asyncio.to_thread(call, *args), STORAGE_TIMEOUT)

    async def exists(self, *parts: str) -> bool:
        return await self._bounded(self.fs.exists, self.path(*parts))

    async def read(self, *parts: str) -> str:
        return await self._bounded(lambda: self.fs.cat_file(self.path(*parts)).decode())

    async def glob(self, *parts: str) -> list[str]:
        return sorted(await self._bounded(self.fs.glob, self.path(*parts)))

    async def read_many(self, paths: Sequence[str]) -> dict[str, str]:
        """Read ``paths`` (as ``glob`` returns them), ``STORAGE_READ_BATCH`` per bounded call."""
        texts: dict[str, str] = {}
        for start in range(0, len(paths), STORAGE_READ_BATCH):
            batch = list(paths[start : start + STORAGE_READ_BATCH])
            texts.update({p: b.decode() for p, b in (await self._bounded(self.fs.cat, batch)).items()})
        return texts

    async def write(self, files: Mapping[str, str]) -> None:
        await self._bounded(self._write, files)

    def _write(self, files: Mapping[str, str]) -> None:
        for name, text in files.items():
            path = self.path(name)
            self.fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
            with self.fs.open(path, "w") as handle:
                handle.write(text)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


PROBE_IDEA = CapabilityIdea(
    capability_id="probe",
    subject_id="probe",
    subject_name="probe",
    catalog_version="probe",
    capability={},
    prerequisite_edges=(),
    capability_hash="probe",
)
PROBE_SLOT = PlannedSlot(
    slot=1,
    title="probe",
    workflow="probe",
    environment=Environment.REASONING,
    solver_machine=SolverMachine.NONE,
    verification=Verification.CODE,
    distinctive_challenge="probe",
    research_need=ResearchNeed.NOT_NEEDED,
    research_reason="probe",
    status=SlotStatus.PROPOSE,
    reason=None,
)
PROBE_PLAN = SlotPlan(coverage_rationale="probe", excluded_combinations=[], slots=[PROBE_SLOT], research_priorities=[])
"""A fixed capability and plan the prompt templates are rendered against, so their hashes track the template text."""


@dataclass(frozen=True)
class Provenance:
    """What a result depends on besides its capability; ``result.json`` records it with the capability hash.

    The template hashes cover each prompt rendered against ``PROBE_IDEA`` together with its request schema.
    """

    slots: int
    catalog: str
    environments: tuple[str, ...]
    policy: str
    system_prompt_sha256: str
    verification_rule_sha256: str
    plan_template_sha256: str
    proposal_template_sha256: str
    git_sha: str
    max_tokens: int
    max_repairs: int

    def record(self, idea: CapabilityIdea) -> dict[str, object]:
        return json.loads(json.dumps({**asdict(self), "capability_hash": idea.capability_hash}))


@dataclass(frozen=True)
class ShardSpec:
    """One shard's configuration: which shard, how it proposes, and the provenance its results record."""

    shard: int
    slots: int
    concurrency: int
    environments: frozenset[Environment]
    max_tokens: int
    verification_rule: str
    policy_name: str
    catalog: str
    git_sha: str

    def provenance(self) -> Provenance:
        environments = frozenset(self.environments)
        probe_id = f"{PROBE_IDEA.capability_id}/1"
        plan_template = plan_prompt(PROBE_IDEA, self.slots, environments) + json.dumps(
            response_format(slot_plan_type(self.slots, environments)), sort_keys=True
        )
        proposal_template = proposal_prompt(PROBE_IDEA, PROBE_PLAN, PROBE_SLOT, probe_id) + json.dumps(
            document_request_fields(probe_id, source_ref(PROBE_IDEA)), sort_keys=True
        )
        return Provenance(
            slots=self.slots,
            catalog=self.catalog,
            environments=tuple(str(e) for e in sorted(environments)),
            policy=self.policy_name,
            system_prompt_sha256=sha256(system_prompt(self.verification_rule)),
            verification_rule_sha256=sha256(self.verification_rule),
            plan_template_sha256=sha256(plan_template),
            proposal_template_sha256=sha256(proposal_template),
            git_sha=self.git_sha,
            max_tokens=self.max_tokens,
            max_repairs=MAX_REPAIRS,
        )


async def done_capabilities(out: OutputRoot, ideas: Mapping[str, CapabilityIdea], provenance: Provenance) -> set[str]:
    """The capability ids with a ``result.json`` under ``out``.

    Raises:
        ProvenanceMismatch: a ``result.json`` records another provenance, or a capability outside the catalog.
    """
    texts = await out.read_many(await out.glob("*", RESULT_FILE))
    done: set[str] = set()
    mismatched: list[str] = []
    for path, text in texts.items():
        capability_id = path.rsplit("/", 2)[-2]
        recorded = json.loads(text).get("provenance")
        idea = ideas.get(capability_id)
        if idea is None or recorded != provenance.record(idea):
            mismatched.append(f"{capability_id}: {recorded}")
        done.add(capability_id)
    if mismatched:
        raise ProvenanceMismatch(
            f"{len(mismatched)} results under {out.root} differ from this run's provenance "
            f"{provenance.record(PROBE_IDEA)} (capability_hash per capability); first: {mismatched[:3]}"
        )
    return done


@dataclass(frozen=True)
class Control:
    """A read of the control file: the per-shard limit (0 when paused) and when the controller wrote it."""

    limit: int
    paused: bool
    written_at: float


def parse_control(text: str) -> Control:
    """Parse ``control.json``; raise ``ValueError``, ``KeyError`` or ``TypeError`` when it is malformed."""
    raw = json.loads(text)
    written_at = datetime.fromisoformat(raw["written_at"])
    if written_at.tzinfo is None:
        raise ValueError(f"written_at {raw['written_at']!r} has no time zone")
    paused = raw.get("pause", False)
    if not isinstance(paused, bool):
        raise ValueError(f"pause must be true or false, got {paused!r}")
    limit = 0 if paused else int(raw[CONTROL_KEY])
    return Control(limit, paused, written_at.timestamp())


@dataclass
class ControlState:
    written_at: float | None = None
    missing_logged_at: float = -math.inf

    def age(self, now: float) -> float | None:
        return None if self.written_at is None else now - self.written_at


async def apply_control(
    out: OutputRoot, limit: RequestLimit, window: RequestLimit, slots: int, state: ControlState
) -> None:
    """Read ``control.json`` once and set ``limit`` and ``window`` from it, or hold them and say why."""
    now = time.time()
    try:
        if not await out.exists(CONTROL_FILE):
            if now - state.missing_logged_at >= MISSING_CONTROL_LOG_INTERVAL:
                logger.warning("no control file, holding %d", limit.limit)
                state.missing_logged_at = now
            return
        control = parse_control(await out.read(CONTROL_FILE))
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.warning("control file unreadable, holding %d: %s", limit.limit, error)
        return
    state.written_at = control.written_at
    age = now - control.written_at
    if age > CONTROL_STALE:
        logger.error("control age %.0fs: controller dead, holding %d", age, limit.limit)
        return
    if not control.paused and control.limit < 1:
        logger.error("control %s %d refused (below 1), holding %d", CONTROL_KEY, control.limit, limit.limit)
        return
    logger.info("control age %.0fs: limit %d%s", age, control.limit, " (paused)" if control.paused else "")
    if control.limit != limit.limit:
        logger.info("control: concurrency %d -> %d", limit.limit, control.limit)
        await limit.set(control.limit)
        await window.set(capability_window(control.limit, slots))


async def wait_or_stop(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), seconds)
    except TimeoutError:
        pass


async def follow_control(
    out: OutputRoot, limit: RequestLimit, window: RequestLimit, slots: int, state: ControlState, stop: asyncio.Event
) -> None:
    while not stop.is_set():
        await wait_or_stop(stop, CONTROL_POLL)
        await apply_control(out, limit, window, slots, state)


class CircuitBreaker:
    """Halts a shard whose recent capability failures outnumber its completions."""

    def __init__(self) -> None:
        self.events: deque[tuple[float, bool]] = deque()
        self.halted: str | None = None

    def record(self, finished: bool, now: float) -> None:
        self.events.append((now, finished))
        while self.events[0][0] < now - BREAKER_WINDOW:
            self.events.popleft()
        failures = sum(not ok for _, ok in self.events)
        completions = len(self.events) - failures
        if self.halted is None and failures >= BREAKER_MIN_FAILURES and failures > completions:
            self.halted = f"{failures} capability failures against {completions} completions in {BREAKER_WINDOW:.0f}s"
            logger.error("circuit breaker: %s; taking no new capability", self.halted)


@dataclass(frozen=True)
class Progress:
    """The shard's state, rewritten to ``shards/progress-NNNN.json`` while it runs."""

    shard: int
    at: float
    elapsed: float
    in_flight: int
    limit: int
    window: int
    capabilities_in_flight: int
    waiting_on_limit: int
    capabilities_done: int
    capabilities_failed: int
    capabilities_pending: int
    requests: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    finish_reasons: dict[str, int]
    outcomes: dict[str, int]
    slot_failures: dict[str, int]
    repairs: dict[str, int]
    control_age: float | None
    last_result_at: float | None
    halted: str | None

    def line(self) -> str:
        return (
            f"progress t={self.elapsed:.0f}s caps {self.capabilities_done} done {self.capabilities_failed} failed "
            f"{self.capabilities_pending} pending | in_flight {self.in_flight}/{self.limit} waiting "
            f"{self.waiting_on_limit} | window {self.capabilities_in_flight}/{self.window} | requests {self.requests} "
            f"finish {self.finish_reasons} | tokens in {self.prompt_tokens} cached {self.cached_tokens} out "
            f"{self.completion_tokens} reasoning {self.reasoning_tokens} | slots {self.outcomes} failures "
            f"{self.slot_failures} repairs {self.repairs} | control_age {self.control_age} halted {self.halted}"
        )


@dataclass
class Shard:
    """A running shard's live state."""

    spec: ShardSpec
    provenance: Provenance
    out: OutputRoot
    meter: Meter
    limit: RequestLimit
    window: RequestLimit
    control: ControlState
    breaker: CircuitBreaker
    total: int
    started: float = field(default_factory=time.monotonic)

    def progress(self) -> Progress:
        done, failed = self.meter.outcomes["capability_done"], self.meter.outcomes["capability_failed"]
        usage = self.meter.usage
        return Progress(
            shard=self.spec.shard,
            at=time.time(),
            elapsed=time.monotonic() - self.started,
            in_flight=self.limit.in_flight,
            limit=self.limit.limit,
            window=self.window.limit,
            capabilities_in_flight=self.window.in_flight,
            waiting_on_limit=self.limit.waiting,
            capabilities_done=done,
            capabilities_failed=failed,
            capabilities_pending=self.total - done - failed,
            requests=self.meter.requests,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            finish_reasons=dict(self.meter.finish_reasons),
            outcomes={k: v for k, v in self.meter.outcomes.items() if not k.startswith("capability_")},
            slot_failures=dict(self.meter.slot_failures),
            repairs=dict(self.meter.repairs),
            control_age=self.control.age(time.time()),
            last_result_at=self.meter.last_result_at,
            halted=self.breaker.halted,
        )

    async def write_progress(self) -> Progress:
        """Log the progress line and rewrite the progress file; a storage error is logged, not raised."""
        progress = self.progress()
        logger.info("%s", progress.line())
        name = f"{SHARDS_DIR}/progress-{self.spec.shard:04d}.json"
        try:
            await self.out.write({name: json.dumps(asdict(progress), indent=2)})
        except OSError as error:
            logger.error("progress file not written: %r", error)
        return progress


async def report_progress(shard: Shard, stop: asyncio.Event) -> None:
    while not stop.is_set():
        await wait_or_stop(stop, PROGRESS_INTERVAL)
        await shard.write_progress()


def plan_files(capability_id: str, portfolio: Portfolio[SlotPlan]) -> dict[str, str]:
    return {
        f"{capability_id}/plan.request.json": json.dumps(list(portfolio.planning_request), indent=2),
        f"{capability_id}/plan.completions.json": pretty_json(portfolio.planning),
        f"{capability_id}/plan.json": json.dumps(portfolio.plan.model_dump(mode="json"), indent=2),
    }


def slot_files(capability_id: str, outcome: SlotOutcome, verdict: Verdict | None) -> dict[str, str]:
    name = f"{capability_id}/slot-{outcome.slot:02d}"
    files = {
        f"{name}.request.json": json.dumps(list(outcome.request), indent=2),
        f"{name}.completions.json": pretty_json(outcome.completions),
    }
    if isinstance(outcome, SlotProposal):
        assert verdict is not None
        files[f"{name}.proposal.json"] = render(outcome.proposal)
        files[f"{name}.verdict.json"] = verdict.to_json()
    else:
        files[f"{name}.failure.txt"] = outcome.error
    return files


@dataclass(frozen=True)
class WrittenSlot:
    outcome: SlotOutcome
    verdict: Verdict | None


async def write_slot(
    source: CapabilitySource, idea: CapabilityIdea, portfolio: Portfolio[SlotPlan], slot: int, shard: Shard
) -> WrittenSlot:
    """Write ``slot``'s document, triage it, and store its files at once."""
    outcome = await source.write(idea, portfolio, slot)
    verdict = await preflight(outcome.proposal) if isinstance(outcome, SlotProposal) else None
    shard.meter.repairs["slot"] += max(len(outcome.completions) - 1, 0)
    if isinstance(outcome, SlotFailure):
        shard.meter.slot_failures[slot_failure_kind(outcome)] += 1
    else:
        assert verdict is not None
        shard.meter.outcomes[str(verdict.decision)] += 1
    await shard.out.write(slot_files(idea.capability_id, outcome, verdict))
    return WrittenSlot(outcome, verdict)


async def propose_capability(source: CapabilitySource, idea: CapabilityIdea, shard: Shard) -> None:
    """Plan ``idea``, write and triage every slot concurrently, then write ``result.json``.

    Raises:
        SlotModelErrors: a slot ran out of model retries or was rejected; no ``result.json`` is written.
    """
    started = time.monotonic()
    portfolio = await source.plan(idea, shard.spec.slots)
    shard.meter.repairs["plan"] += len(portfolio.planning) - 1
    await shard.out.write(plan_files(idea.capability_id, portfolio))
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(write_slot(source, idea, portfolio, s.slot, shard)) for s in portfolio.slots]
    written = [task.result() for task in tasks]
    batch = proposal_batch(portfolio, (w.outcome for w in written))
    unfinished = [f for f in batch.failures if slot_failure_kind(f) not in FINAL_SLOT_FAILURES]
    if unfinished:
        raise SlotModelErrors(unfinished)
    verdicts = {w.outcome.slot: w.verdict for w in written if w.verdict is not None}
    completions = [*batch.planning, *(c for s in batch.slots for c in s.completions)]
    result = {
        "slots": {
            str(s.slot): str(verdicts[s.slot].decision) if isinstance(s, SlotProposal) else f"failure: {s.error}"
            for s in batch.slots
        },
        "requests": len(completions),
        "usage": asdict(sum((c.usage for c in completions), Usage(0, 0, 0, 0))),
        "wall_time": time.monotonic() - started,
        "provenance": shard.provenance.record(idea),
    }
    await shard.out.write({f"{idea.capability_id}/{RESULT_FILE}": json.dumps(result, indent=2)})


def leaf_errors(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for e in error.exceptions for leaf in leaf_errors(e)]
    return [error]


async def propose_one(source: CapabilitySource, idea: CapabilityIdea, shard: Shard) -> bool:
    """Propose and triage ``idea`` and return whether it finished; a ``CAPABILITY_ERRORS`` error writes
    ``failure.json`` with its kind instead. Cancellation and any other error propagate."""
    failure: dict[str, object] | None = None
    try:
        await propose_capability(source, idea, shard)
    except* CAPABILITY_ERRORS as group:
        errors = leaf_errors(group)
        failure = {
            "kind": sorted({type(e).__name__ for e in errors}),
            "error": "; ".join(f"{type(e).__name__}: {e}" for e in errors)[:4000],
            "at": time.time(),
        }
    if failure is None:
        shard.meter.outcomes["capability_done"] += 1
        shard.meter.last_result_at = time.time()
        return True
    logger.warning("%s failed: %s", idea.capability_id, str(failure["error"])[:500])
    shard.meter.outcomes["capability_failed"] += 1
    try:
        await shard.out.write({f"{idea.capability_id}/{FAILURE_FILE}": json.dumps(failure, indent=2)})
    except OSError as error:
        logger.error("%s: %s not written: %r", idea.capability_id, FAILURE_FILE, error)
    return False


@dataclass(frozen=True)
class ShardSummary:
    capabilities: int
    ran: int
    failed: list[str]
    halted: str | None
    progress: Progress


async def run_shard(
    model: ModelClient,
    ideas: Mapping[str, CapabilityIdea],
    capability_ids: Sequence[str],
    out: OutputRoot,
    spec: ShardSpec,
) -> ShardSummary:
    """Propose every capability of ``capability_ids`` that has no ``result.json`` yet; return the shard summary.

    Raises:
        ProvenanceMismatch: the output root holds results of another provenance.
    """
    provenance = spec.provenance()
    done = await done_capabilities(out, ideas, provenance)
    pending = [c for c in capability_ids if c not in done]
    logger.info(
        "shard: %d capabilities, %d already done, %d to run; provenance %s",
        len(capability_ids),
        len(capability_ids) - len(pending),
        len(pending),
        provenance,
    )
    meter = Meter()
    limit = RequestLimit(spec.concurrency)
    window = RequestLimit(capability_window(spec.concurrency, spec.slots))
    shard = Shard(spec, provenance, out, meter, limit, window, ControlState(), CircuitBreaker(), len(pending))
    await apply_control(out, limit, window, spec.slots, shard.control)
    source = CapabilitySource(
        LimitedModel(model, limit, meter),
        LLMPolicy(max_tokens=spec.max_tokens),
        spec.environments,
        verification_rule=spec.verification_rule,
    )

    async def windowed(capability_id: str) -> bool | None:
        async with window:
            if shard.breaker.halted is not None:
                return None
            finished = await propose_one(source, ideas[capability_id], shard)
        shard.breaker.record(finished, time.monotonic())
        return finished

    stop = asyncio.Event()
    watchers = [
        asyncio.create_task(follow_control(out, limit, window, spec.slots, shard.control, stop)),
        asyncio.create_task(report_progress(shard, stop)),
    ]
    try:
        finished = await asyncio.gather(*(windowed(c) for c in pending))
    finally:
        stop.set()
        await asyncio.gather(*watchers)
    progress = await shard.write_progress()
    failed = [c for c, ok in zip(pending, finished, strict=True) if ok is False]
    return ShardSummary(len(capability_ids), len(pending), failed, shard.breaker.halted, progress)


def read_catalog(url: str) -> dict[str, CapabilityIdea]:
    """Load the catalog at ``url`` (a local path or an object-store URL) as ideas by capability id."""
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "catalog.json"
        with fsspec.open(url, "rb") as handle:
            local.write_bytes(handle.read())
        return load_capability_ideas(local)


async def _run(endpoint: EndpointSource, **run: Any) -> ShardSummary:
    async with GlmClient(endpoint, replace(GLM_PROVIDER, structured_repairs=MAX_REPAIRS)) as client:
        return await run_shard(client, **run)


@click.command()
@click.option("--catalog", required=True, help="The capability catalog JSON, a local path or an s3:// URL.")
@click.option("--output-root", required=True, help="The run's output tree, a local path or an s3:// URL.")
@click.option("--shard", type=click.IntRange(min=0), required=True)
@click.option("--shards", type=click.IntRange(min=1), required=True)
@click.option("--slots", type=click.IntRange(min=1), default=25, show_default=True, help="Proposals per capability.")
@click.option(
    "--concurrency",
    type=click.IntRange(min=1),
    required=True,
    help=f"In-flight requests for this shard until {CONTROL_FILE} sets {CONTROL_KEY}.",
)
@click.option(
    "--environment",
    "environments",
    type=click.Choice([e.value for e in Environment]),
    multiple=True,
    default=tuple(e.value for e in Environment),
    show_default=True,
    help="A task environment the plan may propose in; repeatable.",
)
@click.option(
    "--max-tokens",
    type=click.IntRange(min=1),
    default=DEFAULT_MAX_TOKENS,
    show_default=True,
    help="Output budget per request, reasoning included.",
)
@click.option(
    "--verification-rule-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A file whose text, less its final newline, replaces the stack's VERIFICATION_RULE in the system "
    "prompt; its stem names the policy.",
)
@click.option("--git-sha", required=True, help="The commit the bundle was built from, recorded in the provenance.")
@click.option("--capability", "capabilities", multiple=True, help="Only this capability id; repeatable (pilot runs).")
@click.option("--base-url", help="The GLM router's OpenAI-compatible base URL, ending in /v1.")
@click.option("--relay-job", help="The exact Iris GLM relay job id (inside Iris only).")
def main(
    catalog: str,
    output_root: str,
    shard: int,
    shards: int,
    slots: int,
    concurrency: int,
    environments: tuple[str, ...],
    max_tokens: int,
    verification_rule_file: Path | None,
    git_sha: str,
    capabilities: tuple[str, ...],
    base_url: str | None,
    relay_job: str | None,
) -> None:
    """Propose ``--slots`` tasks for each capability of this shard and triage them with the preflight."""
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    if shard >= shards:
        raise click.BadParameter(f"--shard {shard} is not below --shards {shards}")
    endpoint = endpoint_source(base_url, relay_job, Pool.BULK)
    ideas = read_catalog(catalog)
    unsafe = [c for c in ideas if FILE_SAFE_ID.match(c) is None]
    if unsafe:
        raise click.ClickException(f"capability ids unsafe as directory names: {unsafe[:10]}")
    unknown = sorted(set(capabilities) - set(ideas))
    if unknown:
        raise click.ClickException(f"unknown capability ids: {unknown}")
    selected = {c: ideas[c] for c in capabilities} if capabilities else ideas
    capability_ids = shard_ids(selected, shard, shards)
    logger.info(
        "catalog %s: %d capabilities; shard %d/%d has %d", catalog, len(ideas), shard, shards, len(capability_ids)
    )
    spec = ShardSpec(
        shard=shard,
        slots=slots,
        concurrency=concurrency,
        environments=frozenset(Environment(e) for e in environments),
        max_tokens=max_tokens,
        verification_rule=(
            VERIFICATION_RULE
            if verification_rule_file is None
            else verification_rule_file.read_text().removesuffix("\n")
        ),
        policy_name=STACK_POLICY if verification_rule_file is None else verification_rule_file.stem,
        catalog=catalog,
        git_sha=git_sha,
    )
    out = OutputRoot(output_root)
    summary = asyncio.run(_run(endpoint, ideas=ideas, capability_ids=capability_ids, out=out, spec=spec))
    asyncio.run(out.write({f"{SHARDS_DIR}/shard-{shard:04d}.json": json.dumps(asdict(summary), indent=2)}))
    logger.info("shard %d done: failed %d halted %s", shard, len(summary.failed), summary.halted)
    if summary.failed or summary.halted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
