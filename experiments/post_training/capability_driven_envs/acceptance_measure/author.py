# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The stack's agentic author with an arm's treatment applied: ``TreatedAgenticAuthor``.

It is ``AgenticAuthor`` with five seams, each driven by ``AuthorTreatment`` (resolved from an arm file):

* ``guidance``: text appended to the system turn of every fresh, handoff and revision session (L1);
* ``reference_files``: read-only workspace files under ``reference/`` (the d3-v2 kit pages and ``grade_kit.py``);
* ``tools``: ``embed_file`` (d3-v2's embed.py as a tool) and ``run_probe_battery`` (L2), which grades the
  code-generated battery of ``probes.py`` against the last draft, with earlier rounds' witnesses when
  ``witness_replay`` is on. A battery run spends no revision: it is a session tool;
* ``enforce_battery``: ``submit`` refuses a program whose last build has not met the battery, at most
  ``battery_max_refusals`` times, then accepts it and records ``battery_unmet`` (a broken battery cannot trap a
  session);
* in every arm (section 3.1 prerequisites): ``submit`` refuses, without charging, a program byte-identical to the
  one the previous round submitted after a review ``Modify`` (``unchanged_max_refusals`` times); and every tool
  result passes through an identical-error detector that adds a hint after ``hint_after`` identical consecutive
  error results and ends the call as the infrastructure failure ``non_converging_internal_loop`` after
  ``hold_after`` (the coordinator holds and retries it; it never becomes repair feedback).

Every refusal, battery run and detector event is appended to ``acceptance.jsonl`` in the session directory, which
``records.py`` reads. ``_session`` mirrors ``AgenticAuthor._session`` (stack ``5be05096b8``) with the treated
plan and session substituted; keep the two in step when the stack's changes.
"""

import ast
import asyncio
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from taskcompendium.grading_result import Outcome
from taskcompendium.runtime.resources import resource_bytes
from taskforge.llm.agent import AgentInterrupted, AgentTool
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage30_author.author import ROUNDS_DIR, AuthorContext, Revision, RevisionKind
from taskforge.stage30_author.loader import GRADER_STEP
from taskforge.stage40_build_with_program.agentic_author import (
    AGENTIC_DIR,
    AUTHOR_KIND,
    AUTHOR_RECORD,
    CONVERSATION_FILE,
    PATH_ARGUMENT,
    SESSION_FILE,
    SUBMIT_FILE,
    TRANSCRIPT_FILE,
    AgenticAuthor,
    AuthorSession,
    AuthorStatus,
    SessionBuild,
    SessionMode,
    SessionOutcome,
    SessionPlan,
    ToolRefusal,
    _numbered,
    _read_json,
    _round_dir,
    _schema,
    _tool,
    _turn_entry,
    _write_json,
    _write_text,
    candidate_file,
    drive,
    host_retried,
    plan_session,
    session_record,
)
from taskforge.stage40_build_with_program.context import BuildInfrastructureFailure
from taskforge.stage40_build_with_program.glm_author import RECORD_FILE
from taskforge.stage40_build_with_program.outputs import Grader
from taskforge.stage40_build_with_program.run import TaskDraft, step_output
from taskforge.stage40_build_with_program.step_store import SDK_VERSION
from taskforge.stage50_validate.adversary import adversary_record
from taskforge.stage50_validate.round import VALIDATION_FILE, load_validation

from experiments.post_training.capability_driven_envs.acceptance_measure.probes import (
    Contract,
    Probe,
    Witness,
    battery,
    met,
)

EVENTS_FILE = "acceptance.jsonl"
NON_CONVERGING = "non_converging_internal_loop"
BATTERY_CONCURRENCY = 4
ERROR_PREFIXES = ("error:", "build failed", "refused by the loader", "not handed in")
EMBED_START = re.compile(r"^(?P<indent>[ \t]*)# embed: (?P<path>\S+)[ \t]*$")
EMBED_END = re.compile(r"^[ \t]*# end embed[ \t]*$")
EMBED_ASSIGN = re.compile(r"^[ \t]*(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]*=")


@dataclass(frozen=True)
class AuthorTreatment:
    """What an arm changes in the author, and the prerequisite settings every arm shares."""

    guidance: tuple[str, ...]
    reference_files: Mapping[str, str]
    tools: frozenset[str]
    enforce_battery: bool
    witness_replay: bool
    battery_max_refusals: int
    unchanged_text: str
    unchanged_max_refusals: int
    repeated_error_text: str
    hint_after: int
    hold_after: int

    @property
    def system_addendum(self) -> str:
        return "".join(f"\n\n{text.strip()}" for text in self.guidance)


def treated_plan(plan: SessionPlan, treatment: AuthorTreatment) -> SessionPlan:
    """``plan`` with the treatment's guidance after the system turn; a same-session revision keeps its saved turns."""
    if plan.mode is SessionMode.SAME_SESSION or not treatment.guidance:
        return plan
    first, *rest = plan.messages
    if first["role"] != "system":
        raise ValueError("an opening conversation starts with the system turn")
    system = {**first, "content": f"{first['content']}{treatment.system_addendum}"}
    return replace(plan, messages=(system, *rest))


def embedded(source: str, path: str, text: str) -> str:
    """``source`` with the assignment between ``# embed: <path>`` and ``# end embed`` set to ``text``.

    Raises:
        ToolRefusal: the marker or its assignment is missing, or the literal would not evaluate to ``text``.
    """
    lines = source.splitlines(keepends=True)
    for index, line in enumerate(lines):
        start = EMBED_START.match(line)
        if not start or start.group("path") != path:
            continue
        end = next((j for j in range(index + 1, len(lines)) if EMBED_END.match(lines[j])), None)
        if end is None:
            raise ToolRefusal(f"'# embed: {path}' at line {index + 1} has no '# end embed'")
        assign = EMBED_ASSIGN.match(lines[index + 1]) if index + 1 < end else None
        if assign is None:
            raise ToolRefusal(f"the line after '# embed: {path}' must assign the text: name = ...")
        literal = (
            "r'''" + text + "'''"
            if "'''" not in text and not text.endswith(("\\", "'")) and "\r" not in text
            else repr(text)
        )
        updated = "".join(
            [*lines[: index + 1], f"{start.group('indent')}{assign.group('name')} = {literal}\n", *lines[end:]]
        )
        node = next(
            (n for n in ast.walk(ast.parse(updated)) if isinstance(n, ast.Assign) and n.lineno == index + 2), None
        )
        if node is None or ast.literal_eval(node.value) != text:
            raise ToolRefusal("the rewritten literal does not evaluate to the file's text; nothing written")
        return updated
    raise ToolRefusal(f"the program has no '# embed: {path}' marker")


def previous_submission(workpiece_dir: Path, round_no: int) -> str | None:
    """The program digest the latest earlier round's sessions handed in, if any."""
    for earlier in range(round_no - 1, -1, -1):
        sessions = _round_dir(workpiece_dir, earlier) / AGENTIC_DIR
        for k in reversed(_numbered(sessions)):
            submit = sessions / str(k) / SUBMIT_FILE
            if submit.exists() and _read_json(submit).get("accepted"):
                return _read_json(submit)["digest"]
    return None


def earlier_witnesses(workpiece_dir: Path, round_no: int) -> list[Witness]:
    """Every defect submission review asked the author to address in rounds before ``round_no``."""
    witnesses = []
    for earlier in range(round_no):
        round_dir = workpiece_dir / ROUNDS_DIR / str(earlier)
        if not (round_dir / VALIDATION_FILE).exists():
            continue
        for trial in load_validation(round_dir).adversary:
            if trial.report is None or trial.role is None:
                continue
            wanted = {i for d in trial.report.defects if d.needs_addressing for i in d.submission_ids}
            if not wanted:
                continue
            record = adversary_record((round_dir / trial.attempt_file).read_bytes())
            for submission in record.submissions:
                if submission.ordinal not in wanted or any(f.encoding != "utf-8" for f in submission.files):
                    continue
                files = {f.path: f.content for f in submission.files}
                witnesses.append(
                    Witness(f"r{earlier}-{trial.trial}-s{submission.ordinal}", str(trial.role), submission.reply, files)
                )
    return witnesses


class TreatedAuthorSession(AuthorSession):
    """``AuthorSession`` with the arm's workspace files, tools, submit rules and the identical-error detector."""

    def __init__(
        self,
        *args,
        treatment: AuthorTreatment,
        revision: Revision | None,
        previous_digest: str | None,
        witnesses: Sequence[Witness],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.treatment = treatment
        self.revision = revision
        self.previous_digest = previous_digest
        self.witnesses = tuple(witnesses)
        self.battery_by_program: dict[str, bool] = {}
        self.last_built_digest: str | None = None
        self.battery_refusals = 0
        self.unchanged_refusals = 0
        self.last_error: tuple[str, str] | None = None
        self.repeats = 0

    def event(self, kind: str, **fields: object) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with (self.session_dir / EVENTS_FILE).open("a") as handle:
            handle.write(json.dumps({"kind": kind, "at": time.time(), **fields}) + "\n")

    def prepare(self, seed: Path | None) -> None:
        super().prepare(seed)
        for path, text in self.treatment.reference_files.items():
            _write_text(self.workspace / path, text)

    async def run_build(self, arguments: Mapping[str, object]) -> str:
        result = await super().run_build(arguments)
        if self.builds and self.builds[-1]["status"] == "built":
            self.last_built_digest = self.builds[-1]["program_digest"]
        return result

    async def embed_file(self, arguments: Mapping[str, object]) -> str:
        program, path = str(arguments["program"]), str(arguments["file"])
        target = self.path(program, write=True)
        text = self.read(path)
        target.write_text(embedded(target.read_text(), path, text))
        return f"{path}: embedded ({len(text)} characters) in {program}"

    async def run_probe_battery(self, arguments: Mapping[str, object]) -> str:
        draft = self.last_draft
        if draft is None or self.last_built_digest is None:
            raise ToolRefusal("no draft built in this session yet; run_build the program first")
        try:
            contract = Contract.model_validate(dict(arguments))
        except ValueError as error:
            raise ToolRefusal(f"the contract arguments are invalid: {error}") from error
        grader = step_output(draft, self.build.store_root, GRADER_STEP, Grader)
        reference_files = {f"/{r.path}": resource_bytes(r).decode(errors="replace") for r in grader.reference_files}
        witnesses = self.witnesses if self.treatment.witness_replay else ()
        try:
            probes = battery(grader.reference_reply, reference_files, contract, witnesses)
        except ValueError as error:
            raise ToolRefusal(f"the battery cannot read the reference under this contract: {error}") from error
        slots = asyncio.Semaphore(BATTERY_CONCURRENCY)

        async def one(probe: Probe) -> dict:
            files = [candidate_file(path.lstrip("/"), content) for path, content in probe.files.items()]
            async with slots:
                grade = await host_retried(lambda: self._grade(draft, grader, probe.reply, files))
            if grade.status in (Outcome.INFRA_ERROR, Outcome.UNAVAILABLE):
                raise BuildInfrastructureFailure("probe_ungraded", f"probe {probe.name}: {grade.status}")
            ok = met(probe.expect, grade.status, grade.reward, grader.rejection_ceiling)
            return {
                "name": probe.name,
                "class": probe.probe_class,
                "expect": str(probe.expect),
                "status": str(grade.status),
                "reward": grade.reward,
                "met": ok,
            }

        rows = await asyncio.gather(*(one(probe) for probe in probes))
        all_met = all(row["met"] for row in rows)
        self.battery_by_program[self.last_built_digest] = all_met
        self.event(
            "probe_battery",
            program_digest=self.last_built_digest,
            contract=contract.model_dump(mode="json"),
            ceiling=grader.rejection_ceiling,
            probes=rows,
            all_met=all_met,
        )
        lines = [
            f"{'ok  ' if r['met'] else 'MISS'} {r['name']} [{r['class']}] expect {r['expect']}: {r['status']} "
            f"reward {r['reward']}"
            for r in rows
        ]
        missed = sum(not r["met"] for r in rows)
        head = "every probe met its expectation" if all_met else f"{missed} of {len(rows)} probes missed"
        return f"{head} (ceiling {grader.rejection_ceiling}; program {self.last_built_digest[:16]})\n" + "\n".join(lines)

    async def submit(self, arguments: Mapping[str, object]) -> str:
        refusal = await self._refusal(str(arguments["path"]))
        if refusal is not None:
            return refusal
        return await super().submit(arguments)

    async def _refusal(self, raw: str) -> str | None:
        """Why this submit is refused before the stack's own checks, or None."""
        try:
            program = await self.build.loader(self.read(raw))
        except (ToolRefusal, ValueError):
            return None  # the stack's submit reports it
        unchanged = (
            self.revision is not None
            and self.revision.kind is RevisionKind.MODIFY
            and program.digest == self.previous_digest
        )
        if unchanged and self.unchanged_refusals < self.treatment.unchanged_max_refusals:
            self.unchanged_refusals += 1
            self.event("unchanged_resubmit_refused", program_digest=program.digest, refusals=self.unchanged_refusals)
            return f"not handed in: {self.treatment.unchanged_text.strip()}"
        if unchanged:
            self.event("unchanged_resubmit_accepted", program_digest=program.digest)
        if not self.treatment.enforce_battery:
            return None
        passed = self.battery_by_program.get(program.digest)
        if passed:
            return None
        if self.battery_refusals < self.treatment.battery_max_refusals:
            self.battery_refusals += 1
            self.event("battery_refused", program_digest=program.digest, ran=passed is not None)
            why = "missed probes" if passed is False else "has not been run on this program's build"
            return (
                f"not handed in: the probe battery {why}. Build this program, call run_probe_battery with the "
                "contract the instruction states, and fix every MISS (no revision is spent)."
            )
        self.event("battery_unmet", program_digest=program.digest, ran=passed is not None)
        return None

    def tools(self, web_tools: Sequence[AgentTool]) -> list[AgentTool]:
        tools = super().tools(web_tools)
        if "embed_file" in self.treatment.tools:
            tools.append(
                _tool(
                    "embed_file",
                    "Copy a workspace file's exact text into program.py: rewrites the assignment between "
                    "'# embed: <file>' and '# end embed' and checks the literal. program: the program file; file: "
                    "the workspace path written in the marker (for example reference/helpers/grade_kit.py).",
                    _schema({"program": PATH_ARGUMENT, "file": PATH_ARGUMENT}, ["program", "file"]),
                    self.embed_file,
                )
            )
        if "run_probe_battery" in self.treatment.tools:
            properties = {
                name: {k: v for k, v in schema.items() if k in ("type", "enum", "items", "anyOf")}
                for name, schema in Contract.model_json_schema()["properties"].items()
            }
            tools.append(
                _tool(
                    "run_probe_battery",
                    "Grade the code-generated probe battery (reference/patterns/probe-battery.md) against the task "
                    "the last successful run_build built. The arguments are the answer contract the instruction "
                    "states; omitted fields take the defaults. Spends no revision.",
                    _schema(properties, []),
                    self.run_probe_battery,
                )
            )
        return [self._detected(tool) for tool in tools]

    def _detected(self, tool: AgentTool) -> AgentTool:
        async def handler(arguments: Mapping[str, object]) -> str:
            result = await tool.handler(arguments)
            if not result.startswith(ERROR_PREFIXES):
                self.last_error, self.repeats = None, 0
                return result
            signature = (tool.name, hashlib.sha256(result.split("\n", 1)[0].encode()).hexdigest())
            self.repeats = self.repeats + 1 if signature == self.last_error else 1
            self.last_error = signature
            if self.repeats >= self.treatment.hold_after:
                self.event("zero_progress_escalation", tool=tool.name, signature=signature[1], count=self.repeats)
                raise BuildInfrastructureFailure(
                    NON_CONVERGING, f"{tool.name} returned the same error {self.repeats} times in a row"
                )
            if self.repeats >= self.treatment.hint_after:
                self.event("identical_error_hint", tool=tool.name, signature=signature[1], count=self.repeats)
                hint = self.treatment.repeated_error_text.format(count=self.repeats, tool=tool.name)
                return f"{result}\n\n{hint.strip()}"
            return result

        return replace(tool, handler=handler)


@dataclass(frozen=True)
class TreatedAgenticAuthor(AgenticAuthor):
    """``AgenticAuthor`` whose sessions carry ``treatment`` (see the module docstring)."""

    treatment: AuthorTreatment = field(kw_only=True)

    async def _session(
        self,
        build: SessionBuild,
        context: AuthorContext,
        proposal: TaskProposal,
        revision: Revision | None,
    ) -> SessionOutcome:
        plan = treated_plan(
            plan_session(context.workpiece_dir, context.round, proposal, revision, self.config), self.treatment
        )
        sessions = _round_dir(context.workpiece_dir, context.round) / AGENTIC_DIR
        k = len(_numbered(sessions))
        session_dir = sessions / str(k)
        session_id = f"{context.round}.{k}"
        session = TreatedAuthorSession(
            session_dir,
            context.workpiece_id,
            context.round,
            proposal,
            build,
            session_id,
            treatment=self.treatment,
            revision=revision,
            previous_digest=previous_submission(context.workpiece_dir, context.round),
            witnesses=earlier_witnesses(context.workpiece_dir, context.round) if self.treatment.witness_replay else (),
        )
        session.prepare(plan.seed)
        session.event("session_start", mode=str(plan.mode), system_sha256=_system_sha(plan))
        try:
            result = await drive(session, plan, self.config, self.client, self.policy, self.ledger, session_id)
        except AgentInterrupted as error:
            _write_json(session_dir / CONVERSATION_FILE, list(error.run.messages))
            _write_json(session_dir / TRANSCRIPT_FILE, [_turn_entry(i, turn) for i, turn in enumerate(error.run.turns)])
            _write_json(
                session_dir / AUTHOR_RECORD,
                {"status": AuthorStatus.HANDED_OFF, "stop_reason": error.run.stop, "error": str(error)},
            )
            raise
        _write_json(session_dir / CONVERSATION_FILE, list(result.messages))
        _write_json(session_dir / TRANSCRIPT_FILE, list(result.transcript))
        _write_json(session_dir / SESSION_FILE, session_record(session_id, plan, session, result))
        record: dict = {
            "author_kind": AUTHOR_KIND,
            "revision": None if revision is None else {"kind": str(revision.kind), "detail": revision.detail},
            "mode": plan.mode,
            "stop_reason": result.stop,
            "wall": result.wall,
        }
        submission = result.submission
        calls = {"calls": session.submissions}
        if submission is None:
            _write_json(session_dir / SUBMIT_FILE, {"accepted": False, **calls})
            error = f"no submission ({result.stop})"
            _write_json(session_dir / AUTHOR_RECORD, record | {"status": AuthorStatus.HANDED_OFF, "error": error})
            return SessionOutcome(AuthorStatus.HANDED_OFF, None, result.stop)
        _write_json(
            session_dir / SUBMIT_FILE,
            {
                "accepted": True,
                "path": submission.path,
                "digest": submission.manifest.digest,
                "notes": submission.notes,
                "tested": submission.tested,
                "submitted_by": session_id,
                "unchanged_from_previous": submission.manifest.digest == session.previous_digest,
                "battery_met": session.battery_by_program.get(submission.manifest.digest),
                **calls,
            },
        )
        _write_json(
            session_dir / RECORD_FILE,
            {
                "author_kind": AUTHOR_KIND,
                "program_digest": submission.manifest.digest,
                "sdk_version": SDK_VERSION,
                "model": self.client.model,
                "revision": None if revision is None else str(revision.kind),
                "notes": submission.notes,
                "turns": len(result.turns),
            },
        )
        accepted = {"status": AuthorStatus.ACCEPTED, "program_digest": submission.manifest.digest}
        _write_json(session_dir / AUTHOR_RECORD, record | accepted | {"tested": submission.tested})
        return SessionOutcome(AuthorStatus.ACCEPTED, submission.source.decode(), result.stop)


def _system_sha(plan: SessionPlan) -> str | None:
    first = plan.messages[0] if plan.messages else None
    if first is None or first.get("role") != "system":
        return None
    return hashlib.sha256(str(first["content"]).encode()).hexdigest()


def draft_reference(draft: TaskDraft, store_root: str) -> Grader:
    """The grader step's output for ``draft``, as the battery and the shadows read it."""
    return step_output(draft, store_root, GRADER_STEP, Grader)
