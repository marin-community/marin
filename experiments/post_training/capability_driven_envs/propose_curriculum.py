# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Propose ``--slots`` tasks for each capability of one catalog shard with ``CapabilitySource``, triaging each.

A shard is the sorted capability ids taken every ``--shards``-th from ``--shard``. Model requests run under a limit
that starts at ``--concurrency`` and follows ``OUTPUT_ROOT/control.json`` (``apply_control``). Each capability's
directory gets the plan's and each slot's request, completions and proposal with its verdict (or failure), then
``result.json`` with each slot's decision or final failure, usage and the provenance; a capability that failed
gets ``failure.json`` instead. A rerun skips every capability with a ``result.json``, reading only the shard's own
result files, and refuses a root whose results record another provenance (``git_sha`` aside). The process exits
without joining threads, 1 when a capability failed.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import fsspec
from pydantic import BaseModel
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from taskforge.capability import FILE_SAFE_ID, CapabilityIdea
from taskforge.content_hash import pretty_json
from taskforge.defaults import DEFAULT_POLICY
from taskforge.llm import client as llm
from taskforge.llm.glm_client import EndpointSource, GlmClient, Pool
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError, response_format
from taskforge.stage10_proposal.model import Environment, document_request_fields, render
from taskforge.stage10_proposal.source import SlotProposal, propose_all
from taskforge.stage20_triage.preflight import preflight

from experiments.post_training.capability_driven_envs import proposal_source as ps
from experiments.post_training.capability_driven_envs.catalog import load_capability_ideas
from experiments.post_training.capability_driven_envs.generate_tasks import GLM_PROVIDER, endpoint_source

logger = logging.getLogger(__name__)

CONTROL_FILE = "control.json"
CONTROL_KEY = "concurrency_per_shard"
CONTROL_POLL = 15.0
CONTROL_STALE = 300.0
STORAGE_TIMEOUT = 60.0
STORAGE_READ_BATCH = 256
MAX_REPAIRS = 4
WINDOW_FACTOR = 4
"""Capabilities open per ``concurrency / slots``: one averages a quarter of its slots in flight (5-9 of 25 on
2026-11-02), so a factor of 2 left the request limit unreachable."""
RESULT_FILE = "result.json"
FAILURE_FILE = "failure.json"
REPAIRS_EXHAUSTED = f"no valid document after {MAX_REPAIRS} repairs"
CAPABILITY_ERRORS = (StructuredOutputError, llm.ModelUnavailable, llm.ModelRequestRejected, RuntimeError, OSError)
"""What fails one capability rather than the shard; ``RuntimeError`` is a relay that cannot be resolved."""
PROBE_IDEA = CapabilityIdea.model_validate(
    dict.fromkeys(("capability_id", "subject_id", "subject_name", "catalog_version", "capability_hash"), "probe")
    | {"capability": {}, "prerequisite_edges": ()}
)
PROBE_SLOT = ps.PlannedSlot.model_validate(
    dict.fromkeys(("title", "workflow", "distinctive_challenge", "research_reason"), "probe")
    | {"slot": 1, "environment": "reasoning", "solver_machine": "none", "verification": "code"}
    | {"research_need": "not_needed", "status": "propose", "reason": None}
)
"""What the plan and document prompts are rendered against, so the provenance's hashes track their templates."""


class RequestLimit:
    """How many holders may be inside at once; ``set`` changes it while holders wait or run."""

    def __init__(self, limit: int):
        self.limit, self.in_flight, self._changed = limit, 0, asyncio.Condition()

    async def set(self, limit: int) -> None:
        async with self._changed:
            self.limit = limit
            self._changed.notify_all()

    async def __aenter__(self) -> None:
        async with self._changed:
            await self._changed.wait_for(lambda: self.in_flight < self.limit)
            self.in_flight += 1

    async def __aexit__(self, *exc: object) -> None:
        async with self._changed:
            self.in_flight -= 1
            self._changed.notify_all()


def capability_window(concurrency: int, slots: int) -> int:
    return WINDOW_FACTOR * math.ceil(concurrency / slots) + 2


class LimitedModel:
    """Holds each call of ``model`` under ``limit``. A document request after ``MAX_REPAIRS`` repairs (one assistant
    turn each) raises ``ModelUnavailable(REPAIRS_EXHAUSTED)`` unsent, which ``CapabilitySource`` makes a slot failure."""

    def __init__(self, model: llm.ModelClient, limit: RequestLimit):
        self._model, self._limit = model, limit

    @property
    def model(self) -> str:
        return self._model.model

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> llm.Completion:
        if sum(m["role"] == "assistant" for m in messages) > MAX_REPAIRS:
            raise llm.ModelUnavailable(REPAIRS_EXHAUSTED, ())
        async with self._limit:
            return await self._model.complete(messages, policy, request_fields)

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> llm.StructuredResult[T]:
        async with self._limit:
            return await self._model.complete_structured(messages, output_type, policy)


class OutputRoot:
    """A local directory or object-store prefix; each call is bounded by ``STORAGE_TIMEOUT`` (a ``TimeoutError``)."""

    def __init__(self, url: str):
        self.fs, self.root = fsspec.core.url_to_fs(url.rstrip("/"))

    async def _bounded[T](self, call: Callable[[], T]) -> T:
        return await asyncio.wait_for(asyncio.to_thread(call), STORAGE_TIMEOUT)

    async def read_present(self, names: Sequence[str]) -> dict[str, str]:
        """The files of ``names`` that exist, by name, read ``STORAGE_READ_BATCH`` at a time and never listed."""
        texts: dict[str, str] = {}
        for start in range(0, len(names), STORAGE_READ_BATCH):
            paths = [f"{self.root}/{n}" for n in names[start : start + STORAGE_READ_BATCH]]
            found = await self._bounded(lambda paths=paths: self.fs.cat(paths, on_error="omit"))
            texts |= {p[len(self.root) + 1 :]: b.decode() for p, b in found.items()}
        return texts

    async def write(self, files: Mapping[str, str]) -> None:
        """Write ``files`` in order, each call bounded on its own."""
        for name, text in files.items():
            path = f"{self.root}/{name}"
            await self._bounded(lambda path=path: self.fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True))
            await self._bounded(lambda path=path, text=text: self.fs.pipe_file(path, text.encode()))


async def apply_control(root: OutputRoot, limit: RequestLimit, window: RequestLimit, slots: int) -> None:
    """Set ``limit`` and ``window`` from ``control.json``, ``{"concurrency_per_shard": N, "written_at": ...}`` or
    ``{"pause": true, "written_at": ...}``; hold them, logging an error, on a missing, malformed, zoneless, stale
    (``CONTROL_STALE``) or below-1 file."""
    try:
        raw = json.loads((await root.read_present([CONTROL_FILE]))[CONTROL_FILE])
        written_at = datetime.fromisoformat(raw["written_at"])
        age, paused = time.time() - written_at.timestamp(), raw.get("pause") is True
        value = 0 if paused else int(raw[CONTROL_KEY])
        if written_at.tzinfo is None or age > CONTROL_STALE or (value < 1 and not paused):
            raise ValueError(f"{raw} is zoneless, older than {CONTROL_STALE:.0f}s (controller dead) or below 1")
        if value != limit.limit:
            logger.info("control: concurrency %d -> %d (age %.0fs)", limit.limit, value, age)
            await limit.set(value)
            await window.set(capability_window(max(value, 1), slots))
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.error("control file unusable, holding %d: %r", limit.limit, error)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class ShardSpec:
    slots: int
    concurrency: int
    environments: frozenset[Environment]
    max_tokens: int
    catalog: str
    git_sha: str
    rule: str = ps.VERIFICATION_RULE
    """The policy key of the verification rule; its name after ``proposer.`` is the provenance's ``policy``."""

    def provenance(self) -> dict[str, Any]:
        """What a result depends on besides its capability; ``result.json`` adds the capability hash."""
        proposer, rule, envs = DEFAULT_POLICY.proposer, DEFAULT_POLICY.prompt(self.rule), self.environments
        plan = ps.SlotPlan(
            coverage_rationale="probe", excluded_combinations=[], slots=[PROBE_SLOT], research_priorities=[]
        )
        plan_schema = response_format(ps.slot_plan_type(self.slots, envs))
        document_schema = document_request_fields("probe/1", ps.source_ref(PROBE_IDEA))
        return {
            "slots": self.slots,
            "catalog": self.catalog,
            "environments": sorted(str(e) for e in envs),
            "policy": self.rule.removeprefix("proposer."),
            "system_prompt_sha256": sha256(ps.system_prompt(proposer, rule)),
            "verification_rule_sha256": rule.sha256,
            "plan_template_sha256": sha256(
                ps.plan_prompt(proposer, PROBE_IDEA, self.slots, envs) + json.dumps(plan_schema, sort_keys=True)
            ),
            "proposal_template_sha256": sha256(
                ps.proposal_prompt(proposer, PROBE_IDEA, plan, PROBE_SLOT, "probe/1")
                + json.dumps(document_schema, sort_keys=True)
            ),
            "git_sha": self.git_sha,
            "max_tokens": self.max_tokens,
            "max_repairs": MAX_REPAIRS,
        }


class ProvenanceMismatch(Exception):
    """The output root holds results of another configuration."""


def resume_key(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """All of a provenance but ``git_sha``, so a driver fix resumes a run whose prompts are unchanged."""
    return {k: v for k, v in provenance.items() if k != "git_sha"}


async def done_capabilities(root: OutputRoot, ideas: Mapping[str, CapabilityIdea], spec: ShardSpec) -> set[str]:
    """The ids of ``ideas`` with a ``result.json``; raise ``ProvenanceMismatch`` if one records other provenance."""
    texts = await root.read_present([f"{c}/{RESULT_FILE}" for c in ideas])
    recorded = {name.rsplit("/", 1)[0]: json.loads(text).get("provenance") or {} for name, text in texts.items()}
    key = resume_key(spec.provenance())
    mismatched = [c for c, p in recorded.items() if resume_key(p) != key | {"capability_hash": ideas[c].capability_hash}]
    if mismatched:
        raise ProvenanceMismatch(f"{len(mismatched)} results under {root.root} differ from {key}: {mismatched[:3]}")
    return set(recorded)


async def propose_capability(source: ps.CapabilitySource, idea: CapabilityIdea, spec: ShardSpec) -> dict[str, str]:
    """Plan, write and triage every slot of ``idea``; return its files, ``result.json`` included.

    Raises:
        ModelUnavailable: a slot's model call ran out of retries or was rejected.
    """
    started, cid = time.monotonic(), idea.capability_id
    batch = await propose_all(source, idea, spec.slots)
    retryable = [s for s in batch.failures if not s.error.endswith((ps.CUT_OFF, REPAIRS_EXHAUSTED))]
    if retryable:
        raise llm.ModelUnavailable("; ".join(f"slot {s.slot}: {s.error}" for s in retryable), ())
    files = {
        f"{cid}/plan.request.json": json.dumps(list(batch.planning_request), indent=2),
        f"{cid}/plan.completions.json": pretty_json(batch.planning),
        f"{cid}/plan.json": json.dumps(json.loads(batch.planning[-1].content), indent=2),
    }
    decisions = {}
    for outcome in batch.slots:
        name = f"{cid}/slot-{outcome.slot:02d}"
        files[f"{name}.request.json"] = json.dumps(list(outcome.request), indent=2)
        files[f"{name}.completions.json"] = pretty_json(outcome.completions)
        if isinstance(outcome, SlotProposal):
            verdict = await preflight(outcome.proposal)
            files |= {f"{name}.proposal.json": render(outcome.proposal), f"{name}.verdict.json": verdict.to_json()}
            decisions[str(outcome.slot)] = str(verdict.decision)
        else:
            files[f"{name}.failure.txt"] = outcome.error
            decisions[str(outcome.slot)] = f"failure: {outcome.error}"
    completions = [*batch.planning, *(c for s in batch.slots for c in s.completions)]
    files[f"{cid}/{RESULT_FILE}"] = json.dumps(
        {
            "slots": decisions,
            "requests": len(completions),
            "usage": asdict(sum((c.usage for c in completions), llm.Usage(0, 0, 0, 0))),
            "wall_time": time.monotonic() - started,
            "provenance": spec.provenance() | {"capability_hash": idea.capability_hash},
        },
        indent=2,
    )
    return files


async def run_shard(
    model: llm.ModelClient, ideas: Mapping[str, CapabilityIdea], root: OutputRoot, spec: ShardSpec
) -> list[str]:
    """Propose every capability of ``ideas`` without a ``result.json``; return the ids that failed.

    Raises:
        ProvenanceMismatch: the root holds results of another configuration.
    """
    done = await done_capabilities(root, ideas, spec)
    logger.info("%d capabilities, %d done; provenance %s", len(ideas), len(done), spec.provenance())
    limit, window = RequestLimit(spec.concurrency), RequestLimit(capability_window(spec.concurrency, spec.slots))
    policy, rule = LLMPolicy(max_tokens=spec.max_tokens), DEFAULT_POLICY.prompt(spec.rule)
    source = ps.CapabilitySource(LimitedModel(model, limit), policy, DEFAULT_POLICY.proposer, rule, spec.environments)
    failed: list[str] = []

    async def one(cid: str) -> None:
        async with window:
            try:
                await root.write(await propose_capability(source, ideas[cid], spec))
            except* CAPABILITY_ERRORS as group:
                failed.append(cid)
                error = "; ".join(map(repr, group.exceptions))[:4000]
                logger.warning("%s failed: %s", cid, error[:500])
                try:
                    await root.write({f"{cid}/{FAILURE_FILE}": json.dumps({"error": error, "at": time.time()})})
                except OSError as write_error:
                    logger.error("%s: %s not written: %r", cid, FAILURE_FILE, write_error)
        logger.info("%s done; in flight %d/%d, %d failed", cid, limit.in_flight, limit.limit, len(failed))

    async def follow_control() -> None:
        while True:
            await asyncio.sleep(CONTROL_POLL)
            await apply_control(root, limit, window, spec.slots)

    await apply_control(root, limit, window, spec.slots)
    control = asyncio.create_task(follow_control())
    try:
        await asyncio.gather(*(one(c) for c in ideas if c not in done))
    finally:
        control.cancel()
    return failed


def read_catalog(url: str) -> dict[str, CapabilityIdea]:
    fs, path = fsspec.core.url_to_fs(url)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "catalog.json").write_bytes(fs.cat_file(path))
        return load_capability_ideas(Path(tmp) / "catalog.json")


async def _run(endpoint: EndpointSource, ideas: Mapping[str, CapabilityIdea], root: OutputRoot, spec: ShardSpec):
    async with GlmClient(endpoint, replace(GLM_PROVIDER, structured_repairs=MAX_REPAIRS)) as client:
        return await run_shard(client, ideas, root, spec)


@click.command()
@click.option("--catalog", required=True, help="The capability catalog JSON, a local path or an s3:// URL.")
@click.option("--output-root", required=True, help="The run's output tree, a local path or an s3:// URL.")
@click.option("--shard", type=click.IntRange(min=0), required=True)
@click.option("--shards", type=click.IntRange(min=1), required=True)
@click.option("--slots", type=click.IntRange(min=1), default=25, show_default=True)
@click.option("--concurrency", type=click.IntRange(min=1), required=True, help=f"Until {CONTROL_FILE} sets one.")
@click.option("--environment", "environments", type=click.Choice(list(map(str, Environment))), multiple=True)
@click.option("--max-tokens", type=click.IntRange(min=1), default=131072, show_default=True)
@click.option("--git-sha", required=True, help="The commit the bundle was built from, for the provenance.")
@click.option("--capability", "capabilities", multiple=True, help="Only this capability id; repeatable.")
@click.option("--base-url", help="The GLM router's OpenAI-compatible base URL, ending in /v1.")
@click.option("--relay-job", help="The exact Iris GLM relay job id (inside Iris only).")
def main(catalog: str, output_root: str, shard: int, shards: int, capabilities: tuple[str, ...], **options: Any) -> None:
    """Propose tasks for one shard of the catalog, in every environment unless ``--environment`` names some."""
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    if shard >= shards:
        raise click.BadParameter(f"--shard {shard} is not below --shards {shards}")
    endpoint = endpoint_source(options.pop("base_url"), options.pop("relay_job"), Pool.BULK)
    ideas = read_catalog(catalog)
    bad = sorted(c for c in ideas if FILE_SAFE_ID.match(c) is None) + sorted(set(capabilities) - set(ideas))
    if bad:
        raise click.ClickException(f"unsafe or unknown capability ids: {bad[:10]}")
    selected = {c: ideas[c] for c in sorted(capabilities or ideas)[shard::shards]}
    environments = frozenset(map(Environment, options.pop("environments") or Environment))
    spec = ShardSpec(environments=environments, catalog=catalog, **options)
    failed = asyncio.run(_run(endpoint, selected, OutputRoot(output_root), spec))
    logger.info("shard %d done: %d of %d failed", shard, len(failed), len(selected))
    # A storage call that outlived STORAGE_TIMEOUT still holds an executor thread, which interpreter exit joins:
    # four shards of the 2026-11-02 run hung there. Exit without joining, with the shard's status.
    logging.shutdown()
    sys.stdout.flush()
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
