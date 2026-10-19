# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The agentic author offline: a scripted model drives whole sessions against the real loader and ``run_build``.

``ScriptedModel`` is an in-process ``ModelClient`` that answers each request with the next queued
``Completion`` and keeps every request, so a test sets each turn's tool calls and token usage exactly.
The session's ``ProgramBuilder`` runs ``run_build`` in process on a step store under ``tmp_path``,
recording its step spans to a ``JsonlLedger``; the toy ``PROGRAM`` grades in process and calls no model.
"""

import inspect
import itertools
import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from pydantic import BaseModel
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind
from taskforge.llm.client import Completion, FinishReason, StructuredResult, ToolCall, Usage
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.recording import CallLedger
from taskforge.llm.structured import complete_structured, response_format
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage30_author import load_process
from taskforge.stage30_author.author import Revision, RevisionKind, program_manifest
from taskforge.stage30_author.load_process import ProgramManifest, load_program_in_process
from taskforge.stage30_author.loader import load_program_source
from taskforge.stage40_build_with_program import agentic_author as aa
from taskforge.stage40_build_with_program.agentic_author import (
    AgenticAuthorConfig,
    AuthorOutcome,
    AuthorStatus,
    ProgramBuilder,
    SameSessionBounds,
    SessionBuild,
    SessionMode,
    author_round,
    plan_session,
)
from taskforge.stage40_build_with_program.context import BuildExpired, BuildInfrastructureFailure, ExpiredClock
from taskforge.stage40_build_with_program.glm_author import AUTHOR_DIR, GlmAuthor
from taskforge.stage40_build_with_program.infrastructure import InfrastructureCause
from taskforge.stage40_build_with_program.run import ResearchRequirement, TaskDraft, run_build
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage40_build_with_program.template import standard
from tests.stage40_build_with_program.conftest import WORKPIECE, SAMPLING, declaring

PROGRAM = '''"""Grades 6 * 7 in process."""

from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, Source, TaskSpec
from verifyit.spec import ExactSpec

from taskforge.spec import draft
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage30_author.program import rl_env_build_program, rl_env_build_sink, rl_env_build_step
from taskforge.stage40_build_with_program.context import BuildFailure, RLEnvBuildContext
from taskforge.stage40_build_with_program.outputs import Grader, Sources
from taskforge.stage40_build_with_program.sandbox_effects import grade_candidate
from taskforge.stage40_build_with_program.sdk import ANSWER_FORMAT, graded_candidate


@rl_env_build_step
async def sources(context: RLEnvBuildContext, proposal: TaskProposal) -> Sources:
    return Sources(notes="", findings=())


@rl_env_build_step
async def grader(context: RLEnvBuildContext, sources: Sources) -> Grader:
    package = draft.answer_grader(ExactSpec(expected=("42",)))
    reference = await grade_candidate(context, None, package, AnswerType.TEXT, ANSWER_FORMAT, "(instruction)", "42")
    if reference.status != Outcome.GRADED:
        raise BuildFailure(f"the reference was graded {reference.status}")
    return Grader(
        package, "Reply with the number.", "42", (), ("42",), 0.0, (graded_candidate("42", (), reference),)
    )


@rl_env_build_sink
async def assemble(context: RLEnvBuildContext, grader: Grader) -> TaskSpec:
    return draft.assemble(
        context.workpiece_id,
        "What is 6 * 7?",
        AnswerType.TEXT,
        ANSWER_FORMAT,
        grader.package,
        Source(dataset="test", revision="r", row="0", importer_revision="test"),
        environment=draft.requirements(image=None),
    )


PROGRAM = rl_env_build_program(sources, grader, sink=assemble)
'''
CONFIG = AgenticAuthorConfig(
    turns=48,
    turn_max_tokens=49_152,
    session_token_cap=262_144,
    revision_same_session=SameSessionBounds(150_000, 36),
    max_continuations=0,
)
CALL_IDS = itertools.count()
STRUCTURED_REPAIRS = 1
FAILURE = Revision(RevisionKind.BUILD_FAILURE, "fixtures: KeyError 'rows'")
SCHEDULING = BuildInfrastructureFailure(InfrastructureCause.SCHEDULING_TIMEOUT, "the machine factory gave up waiting")
NO_FACTORY = BuildInfrastructureFailure(InfrastructureCause.NO_FACTORY, "this host has no docker machine factory")
ALWAYS = 1_000
"""A failure count no session reaches: the host fails every build."""


@dataclass(frozen=True)
class ScriptedRequest:
    messages: list[Message]
    policy: LLMPolicy
    fields: dict[str, object]


class ScriptedModel:
    """Answers each request with the next queued completion; structured calls go through ``complete``."""

    model = "scripted"

    def __init__(self, *replies: Completion) -> None:
        self.replies: deque[Completion] = deque(replies)
        self.requests: list[ScriptedRequest] = []

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        self.requests.append(ScriptedRequest(list(messages), policy, dict(request_fields or {})))
        return self.replies.popleft()

    async def complete_schema(
        self, messages: Sequence[Message], output_type: type[BaseModel], policy: LLMPolicy
    ) -> Completion:
        return await self.complete(messages, policy, {"response_format": response_format(output_type)})

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        return await complete_structured(self, messages, output_type, policy, STRUCTURED_REPAIRS)


def turn(*calls: tuple[str, Mapping[str, object]], prompt: int = 30_000, out: int = 2_000) -> Completion:
    """An assistant turn making ``calls`` (tool name, arguments)."""
    return Completion(
        content="",
        reasoning="thinking",
        tool_calls=tuple(ToolCall(f"call-{next(CALL_IDS)}", name, json.dumps(args)) for name, args in calls),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(prompt_tokens=prompt, completion_tokens=out, reasoning_tokens=out // 2, cached_tokens=prompt - 1000),
        wall_time=2.0,
        ttft=0.1,
        decode_time=1.9,
        continuations=0,
        attempts=(),
    )


def answer(text: str) -> Completion:
    return Completion(text, "", (), FinishReason.STOP, Usage(30_000, 100, 50, 29_000), 1.0, 0.1, 0.9, 0, ())


def cut(reasoning_tokens: int = 49_152) -> Completion:
    """A turn cut at the output limit while reasoning, before any tool call."""
    usage = Usage(30_000, reasoning_tokens, reasoning_tokens, 29_000)
    return Completion("", "r" * 100, (), FinishReason.LENGTH, usage, 900.0, 0.1, 899.0, 0, ())


def session_ledger(root: Path) -> JsonlLedger:
    return JsonlLedger(root / "ledger" / "author_session")


async def in_process_loader(source: str) -> ProgramManifest:
    """Loads in this process: the tests check the session, and the child loader has its own tests."""
    return ProgramManifest.model_validate(program_manifest(load_program_source(source)))


def session_build(
    root: Path, model: ScriptedModel, services: RLEnvBuildServices, proposal: TaskProposal, round_no: int
) -> SessionBuild:
    """In-process builds on ``root/store``, recording their spans to the session ledger under ``root``."""
    store = str(root / "store")
    ledger = session_ledger(root)
    build_services = replace(services, ledger=ledger)

    async def builder(source: str, run_dir: Path, *, invalidate: frozenset[str]) -> TaskDraft:
        program = load_program_source(source)
        return await run_build(
            program,
            proposal,
            WORKPIECE,
            run_dir,
            store,
            model,
            SAMPLING,
            build_services,
            round_no,
            research=ResearchRequirement.OFF,
            invalidate=invalidate,
        )

    return SessionBuild(builder, store, model, SAMPLING, services, ledger.path_for(WORKPIECE), in_process_loader)


@pytest.fixture
def author(services, proposal, ledger):
    """``author(workpiece_dir, model, round_no=0, revision=None)``: one ``author_round`` under ``workpiece_dir``'s parent."""

    async def run(
        workpiece_dir: Path, model: ScriptedModel, round_no: int = 0, revision: Revision | None = None
    ) -> AuthorOutcome:
        build = session_build(workpiece_dir.parent, model, services, proposal, round_no)
        return await author_round(
            CONFIG, model, LLMPolicy(), ledger, build, workpiece_dir, WORKPIECE, round_no, proposal, revision
        )

    return run


def authored(workpiece: Path, round_no: int, k: int) -> Path:
    return workpiece / "rounds" / str(round_no) / "authored" / str(k)


def agentic(workpiece: Path, round_no: int, k: int) -> Path:
    return workpiece / "rounds" / str(round_no) / "agentic" / str(k)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def tool_outputs(model: ScriptedModel) -> list[str]:
    """The tool results the model saw, in order, from the last request's conversation."""
    return [str(m["content"]) for m in model.requests[-1].messages if m["role"] == "tool"]


def tool_texts(model: ScriptedModel) -> list[str]:
    """Every tool result any request of the session carried."""
    return [str(m["content"]) for request in model.requests for m in request.messages if m["role"] == "tool"]


@dataclass
class HostFailingBuilder:
    """A ``ProgramBuilder`` whose first ``failures`` builds the host fails with ``failure``; ``calls`` counts builds."""

    builder: ProgramBuilder
    failure: BuildInfrastructureFailure
    failures: int
    calls: int = 0

    async def __call__(self, source: str, run_dir: Path, *, invalidate: frozenset[str]) -> TaskDraft:
        self.calls += 1
        if self.calls <= self.failures:
            raise BuildInfrastructureFailure(self.failure.cause, self.failure.message)
        return await self.builder(source, run_dir, invalidate=invalidate)


def step_statuses(path: Path) -> list[tuple[str, str]]:
    return [(e.step, e.attrs["cache"]) for e in read_entries(path) if e.kind is EntryKind.STEP and "tool" not in e.attrs]


async def test_session_builds_grades_and_submits_and_a_later_build_hits_every_step(tmp_path, author, proposal, services):
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_loader", {"path": "program.py"}), ("submit", {"path": "program.py", "notes": "first try"})),
        turn(("run_build", {"path": "program.py"})),
        turn(("run_grade_candidate", {"reply": "41"})),
        turn(("submit", {"path": "program.py", "notes": "grades 6 * 7"})),
    )
    outcome = await author(workpiece, model)

    assert outcome.status is AuthorStatus.ACCEPTED and outcome.source == PROGRAM
    wrote, loader, refused, built, graded = tool_outputs(model)
    assert wrote == f"wrote {len(PROGRAM.encode())} bytes to program.py"
    assert loader.startswith("loads: program ")
    assert refused.startswith("not handed in: the loader accepts it") and "never built in this session" in refused
    assert built.startswith("built: program ") and "sources MISS" in built and "grader MISS" in built
    assert "reference: graded reward 1.0" in built
    assert "empty answer: submission_failure reward 0.0 (graded by the library; ceiling 0.0)" in built
    assert graded.startswith("status graded, reward 0.0")
    assert not model.replies, "the session stopped at the accepted submit"

    target = authored(workpiece, 0, 0)
    assert (target / "program.py").read_bytes() == (agentic(workpiece, 0, 0) / "workspace" / "program.py").read_bytes()
    program = load_program_source(PROGRAM)
    assert read(target / "program.json") == json.loads(json.dumps(program_manifest(program)))
    record = read(target / "author.json")
    assert record["author_kind"] == "agentic" and record["status"] == "accepted" and record["tested"] is True
    assert record["program_digest"] == program.digest and record["stop_reason"] == "submitted"
    assert read(target / "record.json")["author_kind"] == "agentic"
    submit = read(target / "submit.json")
    calls = submit.pop("calls")
    assert submit == {
        "accepted": True,
        "path": "program.py",
        "digest": program.digest,
        "notes": "grades 6 * 7",
        "tested": True,
        "submitted_by": "0.0",
    }
    assert [(c["outcome"], c["loaded"], c["digest"]) for c in calls] == [
        ("refused_untested", True, program.digest),
        ("accepted", True, program.digest),
    ]

    session = read(target / "session.json")
    assert session["turns"] == 5 and session["stop_reason"] == "submitted" and session["mode"] == "fresh"
    assert session["tool_calls"] == {
        "write_file": 1,
        "run_loader": 1,
        "submit": 2,
        "run_build": 1,
        "run_grade_candidate": 1,
    }
    assert session["first_submission_loaded"] is True and session["builds"] == 1 and session["builds_ok"] == 1
    assert session["tokens"] == {"in": 150_000, "out": 10_000, "reasoning": 5_000, "cached": 145_000}
    assert session["cached_share"] == pytest.approx(145 / 150)
    assert [s["outcome"] for s in session["submissions"]] == ["refused_untested", "accepted"]

    transcript = read(target / "transcript.json")
    assert [e["role"] for e in transcript] == ["system", "user", *["assistant"] * 5]
    write = transcript[2]
    assert write["tool_calls"][0]["name"] == "write_file" and write["finish"] == "tool_calls"
    assert write["tool_results"][0]["output"] == wrote
    assert write["tokens"] == {"in": 30_000, "out": 2_000, "reasoning": 1_000, "cached": 29_000}

    # The in-session build ran on the shared store under the workpiece's id, recording to the session ledger; the
    # build stage's build of the submitted program loads every step from the store.
    run_dir = agentic(workpiece, 0, 0) / "runs" / "0"
    assert {p.name for p in run_dir.iterdir()} == {"draft", "run.json"}
    assert read(run_dir / "run.json")["status"] == "built"
    misses = [("sources", "miss"), ("grader", "miss"), ("assemble", "miss")]
    assert step_statuses(session_ledger(tmp_path).path_for(WORKPIECE)) == misses
    stage_ledger = JsonlLedger(tmp_path / "ledger")
    await run_build(
        load_program_source(outcome.source or ""),
        proposal,
        WORKPIECE,
        workpiece / "rounds" / "0",
        str(tmp_path / "store"),
        model,
        SAMPLING,
        replace(services, ledger=stage_ledger),
        0,
        research=ResearchRequirement.OFF,
    )
    assert step_statuses(stage_ledger.path_for(WORKPIECE)) == [("sources", "hit"), ("grader", "hit"), ("assemble", "hit")]


async def test_a_grader_crediting_the_empty_answer_fails_the_build_in_the_grader_step(tmp_path, author, monkeypatch):
    """The program grades nothing but its reference; the library's empty-answer grade still fails the build."""
    graded = (Outcome.GRADED, 1.0)

    async def credit_everything(context, requirements, package, answer_type, answer_format, instruction, reply, **kw):
        status, reward = graded if reply == "" else (Outcome.GRADED, 1.0 if reply == "42" else 0.0)
        return GradeResult(status=status, reward=reward)

    monkeypatch.setattr(aa, "grade_candidate", credit_everything)
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "ships anyway"})),
        turn(("submit", {"path": "program.py", "notes": "ships anyway"})),
    )
    outcome = await author(workpiece, model)

    _, built, refused = tool_outputs(model)
    assert built.startswith("build failed (build_failure in step grader)")
    assert "the empty answer is graded with reward 1.0, above the grader's declared rejection_ceiling 0.0" in built
    assert "never built in this session" in refused
    assert outcome.status is AuthorStatus.ACCEPTED
    assert read(authored(workpiece, 0, 0) / "submit.json")["tested"] is False
    run = read(agentic(workpiece, 0, 0) / "runs" / "0" / "run.json")
    assert (run["status"], run["step"]) == ("build_failure", "grader")


async def test_agentic_record_has_the_one_shot_record_shape(tmp_path, author, proposal, ledger):
    one_shot_workpiece, agentic_workpiece = tmp_path / "one_shot" / "workpiece", tmp_path / "agentic" / "workpiece"
    one_shot_model = ScriptedModel(answer(json.dumps({"notes": "n", "source": PROGRAM})))
    await GlmAuthor(one_shot_model, SAMPLING, one_shot_workpiece, CallLedger(ledger, WORKPIECE, 0, "author"))(proposal, None)
    await author(
        agentic_workpiece,
        ScriptedModel(
            turn(("write_file", {"path": "program.py", "content": PROGRAM})),
            turn(("submit", {"path": "program.py", "notes": "n"})),
            turn(("submit", {"path": "program.py", "notes": "n"})),
        ),
    )

    target = authored(agentic_workpiece, 0, 0)
    assert (one_shot_workpiece / "program.py").read_bytes() == (target / "program.py").read_bytes()
    one_shot, agentic_record = read(one_shot_workpiece / AUTHOR_DIR / "00" / "record.json"), read(target / "record.json")
    shared = {"program_digest", "sdk_version", "model", "revision", "notes"}
    assert {k: one_shot[k] for k in shared} == {k: agentic_record[k] for k in shared}
    assert agentic_record["author_kind"] == "agentic"
    assert read(target / "author.json")["tested"] is False, "the second submit hands in an untested program"


SPINNING_CLASS = "class Spin:\n    while True:\n        pass\n\n\n"
"""An output model whose class body never returns; the template with it added is a program that spins on load."""


async def test_a_module_body_that_never_returns_is_refused_by_the_loader_tool(tmp_path, services, proposal, ledger):
    """The default loader runs in a child process, so a spinning module body costs the session one refusal."""

    async def bounded(source: str) -> ProgramManifest:
        return await load_program_in_process(source, wall_timeout=2)

    spinning = inspect.getsource(standard).replace("PROGRAM = ", f"{SPINNING_CLASS}PROGRAM = ")
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": spinning})),
        turn(("run_loader", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": ""})),
        answer("giving up"),
        answer("still nothing"),
        answer("done"),
    )
    build = replace(session_build(workpiece.parent, model, services, proposal, 0), loader=bounded)

    outcome = await author_round(CONFIG, model, LLMPolicy(), ledger, build, workpiece, WORKPIECE, 0, proposal, None)

    assert outcome.status is not AuthorStatus.ACCEPTED
    _, loader, submitted = tool_outputs(model)[:3]
    cut = "the program's module body ran for more than 2 seconds without returning"
    assert loader.startswith(f"refused by the loader: {cut}")
    assert submitted.startswith(f"refused by the loader, nothing handed in: {cut}")


async def test_a_loader_child_the_host_killed_ends_the_round_as_infrastructure(
    tmp_path, services, proposal, ledger, monkeypatch
):
    """A loader child that dies before it writes a result, here because its module is missing, is the host's."""
    monkeypatch.setattr(load_process, "MODULE", "taskforge.no_such_loader")
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(turn(("run_loader", {"path": "program.py"})), answer("done"))
    build = replace(session_build(tmp_path, model, services, proposal, 0), loader=aa.load_in_child)

    with pytest.raises(BuildInfrastructureFailure) as raised:
        await author_round(CONFIG, model, LLMPolicy(), ledger, build, workpiece, WORKPIECE, 0, proposal, None)

    assert raised.value.cause == InfrastructureCause.BUILD_PROCESS
    assert "the program loader process exited with status 1" in raised.value.message
    assert len(model.replies) == 1 and not tool_texts(model)


async def test_a_build_the_host_failed_once_is_built_again_inside_the_tool(
    tmp_path, services, proposal, ledger, monkeypatch
):
    monkeypatch.setattr(aa, "HOST_RETRY_BACKOFF", 0.0)
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "n"})),
    )
    build = session_build(tmp_path, model, services, proposal, 0)
    flaky = HostFailingBuilder(build.builder, SCHEDULING, failures=1)

    outcome = await author_round(
        CONFIG, model, LLMPolicy(), ledger, replace(build, builder=flaky), workpiece, WORKPIECE, 0, proposal, None
    )

    assert outcome.status is AuthorStatus.ACCEPTED and flaky.calls == 2
    assert tool_outputs(model)[1].startswith("built: program ")
    assert read(agentic(workpiece, 0, 0) / "runs" / "0" / "run.json")["status"] == "built"


async def test_a_container_proposal_on_a_host_without_an_image_builder_ends_before_the_author_runs(
    tmp_path, services, proposal, ledger
):
    container = declaring(proposal, None)
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel()
    build = session_build(tmp_path, model, services, container, 0)

    with pytest.raises(BuildInfrastructureFailure) as raised:
        await author_round(CONFIG, model, LLMPolicy(), ledger, build, workpiece, WORKPIECE, 0, container, None)

    assert raised.value.cause == InfrastructureCause.NO_IMAGE_BUILDER
    assert not model.requests, "the author never ran"
    assert not workpiece.exists(), "the round wrote no record and spent no revision"


@pytest.mark.parametrize(
    ("failure", "builds"), [(SCHEDULING, 1 + aa.HOST_RETRIES), (NO_FACTORY, 1)], ids=["retried", "host-rejection"]
)
async def test_a_build_the_host_keeps_failing_ends_the_round_unseen_by_the_author(
    tmp_path, services, proposal, ledger, monkeypatch, failure, builds
):
    monkeypatch.setattr(aa, "HOST_RETRY_BACKOFF", 0.0)
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "n"})),
    )
    build = session_build(tmp_path, model, services, proposal, 0)
    failing = HostFailingBuilder(build.builder, failure, failures=ALWAYS)

    with pytest.raises(BuildInfrastructureFailure) as raised:
        await author_round(
            CONFIG, model, LLMPolicy(), ledger, replace(build, builder=failing), workpiece, WORKPIECE, 0, proposal, None
        )

    assert raised.value.cause == failure.cause and failing.calls == builds
    assert read(agentic(workpiece, 0, 0) / "runs" / "0" / "run.json")["status"] == "infrastructure"
    assert len(model.replies) == 1, "the session ended at the build"
    assert not any("infrastructure" in text for text in tool_texts(model))
    assert not (authored(workpiece, 0, 0) / "author.json").exists(), "the round wrote no record and spent no revision"


async def test_a_grade_the_host_keeps_failing_ends_the_round_unseen_by_the_author(
    tmp_path, services, proposal, ledger, monkeypatch
):
    monkeypatch.setattr(aa, "HOST_RETRY_BACKOFF", 0.0)
    graded: list[str] = []
    grade_candidate = aa.grade_candidate

    async def host_down_for_candidates(
        context, requirements, grader, answer_type, answer_format, instruction, reply, **kw
    ):
        if not reply:  # the library's own empty-answer grade after the build
            return await grade_candidate(
                context, requirements, grader, answer_type, answer_format, instruction, reply, **kw
            )
        graded.append(reply)
        raise BuildInfrastructureFailure(InfrastructureCause.HOST_UNREACHABLE, "machine controller RPC failed")

    monkeypatch.setattr(aa, "grade_candidate", host_down_for_candidates)
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("run_grade_candidate", {"reply": "41"})),
        turn(("submit", {"path": "program.py", "notes": "n"})),
    )
    build = session_build(tmp_path, model, services, proposal, 0)

    with pytest.raises(BuildInfrastructureFailure) as raised:
        await author_round(CONFIG, model, LLMPolicy(), ledger, build, tmp_path / "workpiece", WORKPIECE, 0, proposal, None)

    assert raised.value.cause == InfrastructureCause.HOST_UNREACHABLE
    assert graded == ["41"] * (1 + aa.HOST_RETRIES)
    assert tool_outputs(model)[1].startswith("built: program ")
    assert not any("grading failed" in text or "infrastructure" in text for text in tool_texts(model))


async def test_a_build_past_its_clock_ends_the_round_unseen_by_the_author_and_unretried(
    tmp_path, services, proposal, ledger
):
    builds: list[Path] = []

    async def expiring(source: str, run_dir: Path, *, invalidate: frozenset[str]) -> TaskDraft:
        builds.append(run_dir)
        raise BuildExpired("ran for more than 1800 seconds without returning", ExpiredClock.STEP, "grader")

    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "n"})),
    )
    build = replace(session_build(tmp_path, model, services, proposal, 0), builder=expiring)

    with pytest.raises(BuildExpired) as raised:
        await author_round(CONFIG, model, LLMPolicy(), ledger, build, workpiece, WORKPIECE, 0, proposal, None)

    assert (raised.value.clock, raised.value.step) == (ExpiredClock.STEP, "grader") and len(builds) == 1
    run = read(agentic(workpiece, 0, 0) / "runs" / "0" / "run.json")
    assert (run["status"], run["clock"], run["step"]) == ("expired", "step", "grader")
    assert run["text"] == "ran for more than 1800 seconds without returning"
    assert len(model.replies) == 1, "the session ended at the build"
    assert not any("ran for more than" in text for text in tool_texts(model))
    assert not (authored(workpiece, 0, 0) / "author.json").exists(), "the round wrote no record and spent no revision"


async def test_tools_refuse_paths_outside_the_workspace_and_writes_to_the_reference(tmp_path, author):
    model = ScriptedModel(
        turn(("read_file", {"path": "../workpiece.json"}), ("write_file", {"path": "reference/sdk.md", "content": "x"})),
        turn(("edit_file", {"path": "program.py", "old_string": "@rl_env_build_step", "new_string": "@x"})),
        turn(("read_file", {"path": "program.py", "start_line": 4, "max_lines": 2})),
        turn(("submit", {"path": "missing.py", "notes": ""})),
        answer("done"),
        answer("done"),
        answer("done"),
    )
    outcome = await author(tmp_path / "workpiece", model)

    assert outcome.status is AuthorStatus.HANDED_OFF
    escape, reference, edit, window, missing = tool_outputs(model)[:5]
    assert escape.startswith("error: '../workpiece.json' is not a path inside the workspace")
    assert reference == "error: reference/ is read-only"
    assert edit.startswith("error: old text has ") and edit.endswith(" matches in program.py")
    assert window.startswith("lines 4-5 of ") and "    4| " in window
    assert missing.startswith("error: no file 'missing.py'")


async def test_unsubmitted_session_hands_off_once_then_the_round_spends_a_revision(tmp_path, author):
    workpiece = tmp_path / "workpiece"
    first = ScriptedModel(
        turn(("edit_file", {"path": "notes.md", "old_string": "", "new_string": "x"})),
        answer("I think it is done."),
        answer("Done."),
        answer("Still done."),
    )
    assert (await author(workpiece, first)).status is AuthorStatus.HANDED_OFF
    assert [m["content"] for m in first.requests[-1].messages if m["role"] == "user"][-2:] == [aa.NUDGE, aa.NUDGE]
    session = read(authored(workpiece, 0, 0) / "session.json")
    assert session["stop_reason"] == "unsubmitted" and session["nudges"] == 2
    assert read(authored(workpiece, 0, 0) / "author.json")["status"] == "handed_off"

    (agentic(workpiece, 0, 0) / "workspace" / "notes.md").write_text("fixtures step is next")
    second = ScriptedModel(answer("no"), answer("no"), answer("no"))
    assert (await author(workpiece, second)).status is AuthorStatus.AUTHOR_FAILED
    opening = second.requests[0].messages
    assert opening[-1]["role"] == "user" and "ended without a submission (unsubmitted)" in str(opening[-1]["content"])
    assert (agentic(workpiece, 0, 1) / "workspace" / "notes.md").read_text() == "fixtures step is next"
    assert read(authored(workpiece, 0, 1) / "session.json")["mode"] == "handoff"
    assert read(authored(workpiece, 0, 1) / "author.json")["status"] == "author_failed"


async def test_context_guard_hands_off_to_fresh_sessions_until_the_round_cap(tmp_path, author):
    """A session over the context cap after any turn hands off, a handoff session included; the round stops at
    MAX_SESSIONS_PER_ROUND."""
    workpiece = tmp_path / "workpiece"
    statuses = []
    for _ in range(aa.MAX_SESSIONS_PER_ROUND):
        model = ScriptedModel(turn(("list_files", {}), prompt=220_000, out=1_000))
        statuses.append((await author(workpiece, model)).status)
    stops = [read(authored(workpiece, 0, k) / "session.json")["stop_reason"] for k in range(aa.MAX_SESSIONS_PER_ROUND)]
    assert stops == ["context"] * aa.MAX_SESSIONS_PER_ROUND
    assert statuses == [AuthorStatus.HANDED_OFF] * (aa.MAX_SESSIONS_PER_ROUND - 1) + [AuthorStatus.AUTHOR_FAILED]
    assert (agentic(workpiece, 0, 1) / "workspace" / "program.py").exists()


async def test_a_cut_turn_is_nudged_to_a_smaller_step_and_never_continued(tmp_path, author):
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        cut(),
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        cut(),
        turn(("submit", {"path": "program.py", "notes": ""})),
        turn(("submit", {"path": "program.py", "notes": ""})),
    )
    outcome = await author(workpiece, model)

    assert outcome.status is AuthorStatus.ACCEPTED
    assert {r.policy.max_continuations for r in model.requests} == {0}
    assert {r.policy.max_tokens for r in model.requests} == {49_152}
    second = model.requests[1].messages
    assert second[-1]["role"] == "user" and str(second[-1]["content"]).startswith(
        "Your reply was cut off at the 49152-token output limit"
    )
    session = read(authored(workpiece, 0, 0) / "session.json")
    assert session["length_nudges"] == 2 and session["nudges"] == 0 and session["length_turns"] == 2
    assert session["turns"] == 5 and session["continuations"] == 0


async def test_cut_turns_past_the_context_cap_hand_off(tmp_path, author):
    model = ScriptedModel(
        cut(), Completion("", "", (), FinishReason.LENGTH, Usage(215_000, 1_000, 1_000, 0), 1.0, 0.1, 0.9, 0, ())
    )
    outcome = await author(tmp_path / "workpiece", model)
    assert outcome.status is AuthorStatus.HANDED_OFF and not model.replies


def prior_session(workpiece: Path, stop: str, last_prompt: int, turns_total: int) -> Path:
    """Round 0's accepted agentic session as ``author_round`` leaves it, for a build failure on its program."""
    session_dir = agentic(workpiece, 0, 0)
    (session_dir / "workspace").mkdir(parents=True)
    (session_dir / "workspace" / "program.py").write_text("# edited\n")
    conversation = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    (session_dir / "conversation.json").write_text(json.dumps(conversation))
    target = authored(workpiece, 0, 0)
    target.mkdir(parents=True)
    session = {"stop_reason": stop, "last_prompt_tokens": last_prompt, "turns_total": turns_total}
    (target / "session.json").write_text(json.dumps(session))
    (target / "author.json").write_text(
        json.dumps({"author_kind": "agentic", "status": "accepted", "stop_reason": stop})
    )
    (target / "submit.json").write_text(json.dumps({"notes": "split the fixtures"}))
    return session_dir


@pytest.mark.parametrize(
    ("stop", "last_prompt", "turns_total", "mode"),
    [
        ("submitted", 150_000, 36, SessionMode.SAME_SESSION),
        ("submitted", 150_001, 20, SessionMode.REVISION),
        ("submitted", 90_000, 37, SessionMode.REVISION),
        ("context", 90_000, 20, SessionMode.REVISION),
    ],
    ids=["within-bounds", "prompt-too-long", "too-many-turns", "not-ended-by-submit"],
)
def test_build_failure_reenters_the_session_only_within_its_bounds(
    tmp_path, proposal, stop, last_prompt, turns_total, mode
):
    workpiece = tmp_path / "workpiece"
    previous = prior_session(workpiece, stop, last_prompt, turns_total)
    plan = plan_session(workpiece, 1, proposal, FAILURE, CONFIG)
    assert plan.mode is mode and plan.seed == previous / "workspace"
    revision_turn = str(plan.messages[-1]["content"])
    assert plan.messages[-1]["role"] == "user" and "fixtures: KeyError 'rows'" in revision_turn
    if mode is SessionMode.SAME_SESSION:
        assert [m["content"] for m in plan.messages[:3]] == ["s", "u", "a"]
        assert plan.turns == 12 and plan.turns_before == 36 and plan.continued_from == "rounds/0/agentic/0"
    else:
        assert plan.messages[0]["role"] == "system" and "# This session" in str(plan.messages[0]["content"])
        assert "split the fixtures" in revision_turn and plan.turns == 48 and plan.turns_before == 0


async def test_same_session_revision_continues_the_conversation(tmp_path, author):
    workpiece = tmp_path / "workpiece"
    prior_session(workpiece, "submitted", 100_000, 30)
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("submit", {"path": "program.py", "notes": ""})),
        turn(("submit", {"path": "program.py", "notes": ""})),
    )
    outcome = await author(workpiece, model, round_no=1, revision=FAILURE)

    assert outcome.status is AuthorStatus.ACCEPTED
    sent = model.requests[0].messages
    assert [m["content"] for m in sent[:3]] == ["s", "u", "a"] and "KeyError 'rows'" in str(sent[3]["content"])
    session = read(authored(workpiece, 1, 0) / "session.json")
    assert session["mode"] == "same_session" and session["turns_total"] == 33
    assert (agentic(workpiece, 1, 0) / "workspace" / "program.py").read_text() == PROGRAM


def test_tool_results_keep_head_and_tail_within_the_cap():
    text = "h" * 4000 + "m" * 3000 + "t" * 3000
    clipped = aa.clip(text)
    assert len(clipped) < aa.TOOL_OUTPUT_CHARS
    assert clipped.startswith("h" * aa.TOOL_OUTPUT_HEAD) and clipped.endswith("t" * aa.TOOL_OUTPUT_TAIL)
    assert "[4500 characters dropped]" in clipped
    assert aa.clip("short") == "short"


def test_relaunch_starts_a_handoff_from_an_interrupted_session_workspace(tmp_path, proposal):
    workpiece = tmp_path / "workpiece"
    interrupted = agentic(workpiece, 0, 0) / "workspace"
    interrupted.mkdir(parents=True)
    (interrupted / "program.py").write_text("# half done\n")
    plan = plan_session(workpiece, 0, proposal, None, CONFIG)
    assert plan.mode is SessionMode.HANDOFF and plan.seed == interrupted and plan.turns == 48
    assert "ended without a submission (interrupted)" in str(plan.messages[-1]["content"])


async def test_an_overlong_write_is_refused_briefly_and_counts_as_executed(tmp_path, author):
    """A write past the bound is refused by the handler in one line: a schema ``maxLength`` would have sent back
    jsonschema's message, which quotes the whole file, unclipped."""
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": "x" * (aa.WRITE_CHARS + 2961)})),
        turn(("submit", {"path": "program.py", "notes": "n" * (aa.NOTES_CHARS + 1)})),
        answer("done"),
        answer("done"),
        answer("done"),
    )
    await author(workpiece, model)

    write, notes = tool_outputs(model)
    assert write == f"error: content holds {aa.WRITE_CHARS + 2961} characters, more than {aa.WRITE_CHARS}; use edit_file"
    assert notes.startswith("error: notes hold 4001 characters, more than 4000; nothing handed in (the loader accepts")
    target = authored(workpiece, 0, 0)
    session = read(target / "session.json")
    assert session["rejected_tool_calls"] == 0
    assert [s["outcome"] for s in session["submissions"]] == ["refused_notes"]
    assert session["submit_refusals"] == {"refused_notes": 1} and session["first_submit_accepted"] is False
    submit = read(target / "submit.json")
    assert submit["accepted"] is False and [c["outcome"] for c in submit["calls"]] == ["refused_notes"]
