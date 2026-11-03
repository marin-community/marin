# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The agentic author offline: a scripted model drives whole author calls against the real loader and ``run_build``.

``ScriptedModel`` (``scripted_author``) answers each request with the next queued ``Completion``, so a test
sets each turn's tool calls and token usage exactly; one author call runs as many sessions as its replies
reach. The sessions' ``ProgramBuilder`` runs ``run_build`` in process on a step store under ``tmp_path``,
recording its step spans to a ``JsonlLedger``; the toy ``PROGRAM`` grades in process and calls no model.

Invariants: empty answer at most the ceiling (``_check_empty_answer``); build program cannot reach the host
(``LOAD_TIMEOUT`` child load); task runs on its declared machine (``machine_available`` before the author).
"""

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.defaults import DEFAULT_POLICY
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind
from taskforge.llm.client import Completion, FinishReason, Usage
from taskforge.llm.policy import LLMPolicy
from taskforge.stage10_proposal.model import SolverMachine, TaskProposal
from taskforge.stage30_author import load_process
from taskforge.stage30_author.author import AuthorContext, AuthoredProgram, Revision, RevisionKind
from taskforge.stage30_author.load_process import ProgramManifest, load_program_in_process
from taskforge.stage40_build_with_program import agentic_author as aa
from taskforge.stage40_build_with_program.agentic_author import (
    AgenticAuthor,
    ProgramBuilder,
    ProgramLoader,
    SessionBuild,
    SessionMode,
    plan_session,
)
from taskforge.stage40_build_with_program.context import BuildExpired, BuildInfrastructureFailure, ExpiredClock
from taskforge.stage40_build_with_program.glm_author import AUTHOR_DIR, GlmAuthor
from taskforge.stage40_build_with_program.infrastructure import InfrastructureCause
from taskforge.stage40_build_with_program.program_sdk import load_program, sdk_exports
from taskforge.stage40_build_with_program.run import ResearchRequirement, TaskDraft, run_build
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage40_build_with_program.step_store import StoreCorruption
from taskforge.stage40_build_with_program.template import standard
from tests.stage40_build_with_program.conftest import SAMPLING, WORKPIECE, RecordingImageBuilder, declaring
from tests.stage40_build_with_program.scripted_author import (
    CONFIG,
    PROGRAM,
    ScriptedModel,
    answer,
    in_process_loader,
    turn,
)

FAILURE = Revision(RevisionKind.BUILD_FAILURE, "fixtures: KeyError 'rows'")
assert CONFIG.turns is not None
GIVE_UP = (answer("no"),) * CONFIG.turns
"""Replies that exercise an explicitly configured author invocation budget."""


def cut(reasoning_tokens: int = 49_152) -> Completion:
    """A turn cut at the output limit while reasoning, before any tool call."""
    usage = Usage(30_000, reasoning_tokens, reasoning_tokens, 29_000)
    return Completion("", "r" * 100, (), FinishReason.LENGTH, usage, 900.0, 0.1, 899.0, 0, ())


def write_build_submit(source: str = PROGRAM, path: str = "program.py", submits: int = 1) -> tuple[Completion, ...]:
    """Turns writing ``source`` to ``path``, building it once, and submitting it ``submits`` times."""
    submit = [turn(("submit", {"path": path, "notes": "n"})) for _ in range(submits)]
    return (turn(("write_file", {"path": path, "content": source})), turn(("run_build", {"path": path})), *submit)


def session_ledger(root: Path) -> JsonlLedger:
    return JsonlLedger(root / "ledger" / "author_session")


def session_build(
    root: Path, model: ScriptedModel, services: RLEnvBuildServices, proposal: TaskProposal, round_no: int
) -> SessionBuild:
    """In-process builds on ``root/store``, recording their spans to the session ledger under ``root``."""
    store = str(root / "store")
    ledger = session_ledger(root)
    build_services = replace(services, ledger=ledger)

    async def builder(source: str, run_dir: Path, *, invalidate: frozenset[str]) -> TaskDraft:
        program = load_program(source)
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

    return SessionBuild(builder, store, services, ledger.path_for(WORKPIECE), in_process_loader)


@pytest.fixture
def author(services, proposal, ledger):
    """``author(workpiece_dir, model, round_no, revision, *, proposal, services, builder, loader)``: one
    ``AgenticAuthor`` call whose sessions build in process under ``workpiece_dir``'s parent.

    ``proposal`` and ``services`` replace the fixtures, ``builder`` maps the in-process builder to the one the
    sessions use, and ``loader`` replaces the in-process loader.
    """
    default_proposal, default_services = proposal, services

    async def run(
        workpiece_dir: Path,
        model: ScriptedModel,
        round_no: int = 0,
        revision: Revision | None = None,
        *,
        proposal: TaskProposal | None = None,
        services: RLEnvBuildServices | None = None,
        builder: Callable[[ProgramBuilder], ProgramBuilder] | None = None,
        loader: ProgramLoader | None = None,
    ) -> AuthoredProgram:
        proposal = proposal or default_proposal
        build = session_build(workpiece_dir.parent, model, services or default_services, proposal, round_no)
        build = replace(
            build, builder=builder(build.builder) if builder else build.builder, loader=loader or build.loader
        )
        author = AgenticAuthor(CONFIG, model, LLMPolicy(), ledger, lambda *_: build, DEFAULT_POLICY.author)
        return await author(AuthorContext(WORKPIECE, round_no, workpiece_dir), proposal, revision, {})

    return run


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


def session_outputs(workpiece: Path, k: int) -> list[str]:
    """The tool results of round 0's session ``k``, in order, as its transcript keeps them."""
    transcript = json.loads((agentic(workpiece, 0, k) / "transcript.json").read_text())
    return [result["output"] for entry in transcript for result in entry.get("tool_results", [])]


@dataclass
class FailingBuilder:
    """Raises ``error()`` on its first ``failures`` builds, then builds with the builder ``wrap`` was given."""

    error: Callable[[], Exception]
    failures: int = 1_000
    calls: int = 0
    builder: ProgramBuilder | None = None

    def wrap(self, builder: ProgramBuilder) -> ProgramBuilder:
        self.builder = builder
        return self

    async def __call__(self, source: str, run_dir: Path, *, invalidate: frozenset[str]) -> TaskDraft:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error()
        assert self.builder is not None
        return await self.builder(source, run_dir, invalidate=invalidate)


def step_statuses(path: Path) -> list[tuple[str, str]]:
    return [(e.step, e.attrs["cache"]) for e in read_entries(path) if e.kind is EntryKind.STEP and "tool" not in e.attrs]


async def test_a_candidate_file_at_an_absolute_path_is_refused_and_the_session_goes_on(tmp_path, author):
    absolute = {"path": "/workspace/report.md", "content": "the report"}
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("run_grade_candidate", {"reply": "41", "files": [absolute]})),
        turn(("run_grade_candidate", {"reply": "41", "files": [{**absolute, "path": "workspace/report.md"}]})),
        turn(("submit", {"path": "program.py", "notes": "grades 6 * 7"})),
    )
    authored = await author(tmp_path / "workpiece", model)

    assert authored.source == PROGRAM
    refused, graded = tool_outputs(model)[2:4]
    assert refused.startswith("error: grading refused: '/workspace/report.md'") and "workspace/report.md" in refused
    assert graded.startswith("status graded, reward 0.0")


@pytest.mark.parametrize(
    ("machine", "environment"),
    [(SolverMachine.SHELLSIM, "draft.requirements(image=None)"), (SolverMachine.NONE, "draft.NO_MACHINE")],
    ids=["shellsim", "no-machine"],
)
async def test_a_task_with_or_without_a_machine_grades_in_its_build_and_through_the_tool(
    tmp_path, author, proposal, machine, environment
):
    """The program's grader step and ``run_grade_candidate`` both pass the task's requirements as they are."""
    program = PROGRAM.replace("context, draft.NO_MACHINE,", f"context, {environment},").replace(
        "environment=draft.requirements(image=None).model_copy(\n"
        '            update={"package_spec": replace(draft.SHELLSIM_CPYTHON, pypi_packages=None)}\n        )',
        f"environment={environment}",
    )
    assert program.count(environment) == 2
    model = ScriptedModel(
        *write_build_submit(program, submits=0),
        turn(("run_grade_candidate", {"reply": "42"})),
        turn(("run_grade_candidate", {"reply": "41"})),
        turn(("submit", {"path": "program.py", "notes": "grades 6 * 7"})),
    )
    await author(tmp_path / "workpiece", model, proposal=declaring(proposal, machine))

    built, right, wrong = tool_outputs(model)[1:4]
    assert built.startswith("built: program ")
    assert right.startswith("status graded, reward 1.0") and wrong.startswith("status graded, reward 0.0")


async def test_session_builds_grades_and_submits_and_a_later_build_hits_every_step(tmp_path, author, proposal, services):
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_loader", {"path": "program.py"}), ("submit", {"path": "program.py", "notes": "first try"})),
        turn(("run_build", {"path": "program.py"})),
        turn(("run_grade_candidate", {"reply": "41"})),
        turn(("submit", {"path": "program.py", "notes": "grades 6 * 7"})),
    )
    authored = await author(workpiece, model)

    program = load_program(PROGRAM)
    assert authored.source == PROGRAM and authored.program is program
    wrote = tool_outputs(model)[0]
    graded = tool_outputs(model)[-1]
    assert graded.startswith("status graded, reward 0.0")
    assert not model.replies, "the session stopped at the accepted submit"

    target = agentic(workpiece, 0, 0)
    assert (target / "workspace" / "program.py").read_text() == authored.source
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
    run_dir = target / "runs" / "0"
    assert {p.name for p in run_dir.iterdir()} == {"draft", "run.json"}
    assert read(run_dir / "run.json")["status"] == "built"
    misses = [("sources", "miss"), ("grader", "miss"), ("assemble", "miss")]
    assert step_statuses(session_ledger(tmp_path).path_for(WORKPIECE)) == misses
    stage_ledger = JsonlLedger(tmp_path / "ledger")
    await run_build(
        authored.program,
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
    assert step_statuses(stage_ledger.path_for(WORKPIECE)) == [
        ("sources", "hit"),
        ("grader", "hit"),
        ("assemble", "hit"),
    ]


async def test_a_grader_crediting_the_empty_answer_fails_the_build_in_the_grader_step(tmp_path, author, monkeypatch):
    """The program grades nothing but its reference; the library's empty-answer grade still fails the build."""

    async def credit_everything(
        services, workpiece_id, requirements, package, answer_type, answer_format, instruction, reply, **kw
    ):
        return GradeResult(status=Outcome.GRADED, reward=1.0 if reply in ("", "42") else 0.0)

    monkeypatch.setattr(aa, "grade_candidate_on_host", credit_everything)
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(*write_build_submit(submits=2))
    await author(workpiece, model)

    _, built, refused = tool_outputs(model)
    assert built.startswith("build failed (build_failure in step grader)")
    assert "the empty answer is graded with reward 1.0, above the grader's declared rejection_ceiling 0.0" in built
    assert "never built in this session" in refused
    assert read(agentic(workpiece, 0, 0) / "submit.json")["tested"] is False
    run = read(agentic(workpiece, 0, 0) / "runs" / "0" / "run.json")
    assert (run["status"], run["step"]) == ("build_failure", "grader")


async def test_agentic_record_has_the_one_shot_record_shape(tmp_path, author, proposal, ledger):
    one_shot_workpiece, agentic_workpiece = tmp_path / "one_shot" / "workpiece", tmp_path / "agentic" / "workpiece"
    one_shot_model = ScriptedModel(answer(json.dumps({"notes": "n", "source": PROGRAM})))
    await GlmAuthor(one_shot_model, SAMPLING, ledger, DEFAULT_POLICY.author)(
        AuthorContext(WORKPIECE, 0, one_shot_workpiece), proposal, None, {}
    )
    await author(
        agentic_workpiece,
        ScriptedModel(
            turn(("write_file", {"path": "program.py", "content": PROGRAM})),
            turn(("submit", {"path": "program.py", "notes": "n"})),
            turn(("submit", {"path": "program.py", "notes": "n"})),
        ),
    )

    target = agentic(agentic_workpiece, 0, 0)
    assert (one_shot_workpiece / "program.py").read_bytes() == (target / "workspace" / "program.py").read_bytes()
    one_shot, agentic_record = read(one_shot_workpiece / AUTHOR_DIR / "00" / "record.json"), read(target / "record.json")
    shared = {"program_digest", "sdk_version", "model", "revision", "notes"}
    assert {k: one_shot[k] for k in shared} == {k: agentic_record[k] for k in shared}
    assert agentic_record["author_kind"] == "agentic"
    assert read(target / "author.json")["tested"] is False, "the second submit hands in an untested program"


FIRST_STEP = "\n\n\n@rl_env_build_step\n"
SPINNING_CLASS = "class Spin:\n    while True:\n        pass\n\n\n"
"""An output model whose class body never returns; the template with it added is a program that spins on load."""


async def test_a_module_body_that_never_returns_is_refused_by_the_loader_tool(tmp_path, author):
    """The default loader runs in a child process, so a spinning module body costs the session one refusal."""

    async def bounded(source: str) -> ProgramManifest:
        return await load_program_in_process(source, sdk_exports(), wall_timeout=2)

    spinning = inspect.getsource(standard).replace(FIRST_STEP, f"\n\n\n{SPINNING_CLASS}@rl_env_build_step\n", 1)
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": spinning})),
        turn(("run_loader", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": ""})),
        answer("giving up"),
        answer("still nothing"),
        answer("done"),
        *GIVE_UP,
    )

    with pytest.raises(BuildInfrastructureFailure):
        await author(workpiece, model, loader=bounded)

    _, loader, submitted = session_outputs(workpiece, 0)[:3]
    cut = "the program's module body ran for more than 2 seconds without returning"
    assert loader.startswith(f"refused by the loader: {cut}")
    assert submitted.startswith(f"refused by the loader, nothing handed in: {cut}")


async def test_a_loader_child_the_host_killed_ends_the_round_as_infrastructure(tmp_path, author, monkeypatch):
    """A loader child that dies before it writes a result, here because its module is missing, is the host's."""
    monkeypatch.setattr(load_process, "MODULE", "taskforge.no_such_loader")
    model = ScriptedModel(turn(("run_loader", {"path": "program.py"})), answer("done"))

    with pytest.raises(BuildInfrastructureFailure) as raised:
        await author(tmp_path / "workpiece", model, loader=aa.load_in_child)

    assert raised.value.cause == InfrastructureCause.BUILD_PROCESS
    assert "the program loader process exited with status 1" in raised.value.message
    assert len(model.replies) == 1 and not tool_texts(model)


def scheduling_timeout() -> BuildInfrastructureFailure:
    return BuildInfrastructureFailure(InfrastructureCause.SCHEDULING_TIMEOUT, "the machine factory gave up waiting")


async def test_a_build_the_host_failed_once_is_built_again_inside_the_tool(tmp_path, author, monkeypatch):
    monkeypatch.setattr(aa, "HOST_RETRY_BACKOFF", 0.0)
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(*write_build_submit())
    flaky = FailingBuilder(scheduling_timeout, failures=1)

    authored = await author(workpiece, model, builder=flaky.wrap)

    assert authored.source == PROGRAM and flaky.calls == 2
    assert tool_outputs(model)[1].startswith("built: program ")
    assert read(agentic(workpiece, 0, 0) / "runs" / "0" / "run.json")["status"] == "built"


@pytest.mark.parametrize("cause", [InfrastructureCause.NO_IMAGE_BUILDER, InfrastructureCause.NO_FACTORY])
async def test_a_container_proposal_on_an_unavailable_host_ends_before_the_author_runs(
    tmp_path, author, services, proposal, cause
):
    if cause is InfrastructureCause.NO_FACTORY:
        services = replace(services, images=RecordingImageBuilder(), factories={})
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel()

    with pytest.raises(BuildInfrastructureFailure) as raised:
        await author(workpiece, model, proposal=declaring(proposal, None), services=services)

    assert raised.value.cause == cause
    assert not model.requests, "the author never ran"
    assert not workpiece.exists(), "the round wrote no record and spent no revision"


EXPIRED = "ran for more than 1800 seconds without returning"
INFRASTRUCTURE = {"status": "infrastructure"}
UNSEEN_FAILURES = {
    "retried": (
        scheduling_timeout,
        BuildInfrastructureFailure,
        {"cause": InfrastructureCause.SCHEDULING_TIMEOUT},
        1 + aa.HOST_RETRIES,
        INFRASTRUCTURE,
    ),
    "host-rejection": (
        lambda: BuildInfrastructureFailure(InfrastructureCause.NO_FACTORY, "this host has no docker machine factory"),
        BuildInfrastructureFailure,
        {"cause": InfrastructureCause.NO_FACTORY},
        1,
        INFRASTRUCTURE,
    ),
    "unclassified": (
        lambda: RuntimeError("Docker daemon disconnected"),
        BuildInfrastructureFailure,
        {"cause": "unclassified_build_failure"},
        1,
        INFRASTRUCTURE,
    ),
    "expired": (
        lambda: BuildExpired(EXPIRED, ExpiredClock.STEP, "grader"),
        BuildExpired,
        {"clock": ExpiredClock.STEP, "step": "grader"},
        1,
        {"status": "expired", "clock": "step", "step": "grader", "text": EXPIRED},
    ),
    "storage": (
        lambda: StoreCorruption("Blob sha256:00 of step grader does not match its digest"),
        StoreCorruption,
        {},
        1,
        {"status": "storage"},
    ),
}


@pytest.mark.parametrize(
    ("error", "raises", "raised_fields", "builds", "run"), UNSEEN_FAILURES.values(), ids=UNSEEN_FAILURES.keys()
)
async def test_a_build_the_host_or_store_fails_ends_the_round_unseen_by_the_author(
    tmp_path, author, monkeypatch, error, raises, raised_fields, builds, run
):
    """A host failure (retried while retryable), a build deadline or a storage fault ends the round with no
    revision: the author sees no build result and the call writes no record."""
    monkeypatch.setattr(aa, "HOST_RETRY_BACKOFF", 0.0)
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(*write_build_submit())
    failing = FailingBuilder(error)

    with pytest.raises(raises) as raised:
        await author(workpiece, model, builder=failing.wrap)

    assert {name: getattr(raised.value, name) for name in raised_fields} == raised_fields
    assert failing.calls == builds
    record = read(agentic(workpiece, 0, 0) / "runs" / "0" / "run.json")
    assert {key: record[key] for key in run} == run
    assert len(model.replies) == 1, "the session ended at the build"
    hidden = ("infrastructure", str(error()))
    assert not any(text in output for output in tool_texts(model) for text in hidden)
    assert not (agentic(workpiece, 0, 0) / "author.json").exists(), "the call wrote no record and spent no revision"


async def test_a_grade_the_host_keeps_failing_ends_the_round_unseen_by_the_author(tmp_path, author, monkeypatch):
    monkeypatch.setattr(aa, "HOST_RETRY_BACKOFF", 0.0)
    graded: list[str] = []
    grade_candidate = aa.grade_candidate_on_host

    async def host_down_for_candidates(
        services, workpiece_id, requirements, grader, answer_type, answer_format, instruction, reply, **kw
    ):
        if not reply:  # the library's own empty-answer grade after the build
            return await grade_candidate(
                services, workpiece_id, requirements, grader, answer_type, answer_format, instruction, reply, **kw
            )
        graded.append(reply)
        raise BuildInfrastructureFailure(InfrastructureCause.HOST_UNREACHABLE, "machine controller RPC failed")

    monkeypatch.setattr(aa, "grade_candidate_on_host", host_down_for_candidates)
    model = ScriptedModel(
        *write_build_submit(submits=0),
        turn(("run_grade_candidate", {"reply": "41"})),
        turn(("submit", {"path": "program.py", "notes": "n"})),
    )

    with pytest.raises(BuildInfrastructureFailure) as raised:
        await author(tmp_path / "workpiece", model)

    assert raised.value.cause == InfrastructureCause.HOST_UNREACHABLE
    assert graded == ["41"] * (1 + aa.HOST_RETRIES)
    assert tool_outputs(model)[1].startswith("built: program ")
    assert not any("grading failed" in text or "infrastructure" in text for text in tool_texts(model))


async def test_an_os_error_the_program_raises_reaches_the_author(tmp_path, author):
    missing = PROGRAM.replace(
        '    return Sources(notes="", findings=())', '    raise FileNotFoundError("fixtures/matrices.json")'
    )
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(*write_build_submit(missing, submits=2))
    await author(workpiece, model)

    assert tool_outputs(model)[1].startswith("build failed (")
    assert "fixtures/matrices.json" in tool_outputs(model)[1]
    assert read(agentic(workpiece, 0, 0) / "runs" / "0" / "run.json")["status"] != "storage"


async def test_a_crashing_step_leads_the_build_result_with_the_program_line_that_raised_it(tmp_path, author):
    """``sources`` raises while two later steps await it; the result names its line before any traceback frame."""
    raising = '    return Sources(notes="", findings=({}["rows"],))'
    crashing = PROGRAM.replace('    return Sources(notes="", findings=())', raising).replace(
        "assemble(context: RLEnvBuildContext, grader: Grader)",
        "assemble(context: RLEnvBuildContext, sources: Sources, grader: Grader)",
    )
    line = crashing.splitlines().index(raising) + 1
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(*write_build_submit(crashing, "draft.py", submits=2))
    await author(workpiece, model)

    built = tool_outputs(model)[1]
    preamble, _, failure = built.partition("\n\n")
    assert preamble.startswith("build failed (crash)\nsteps: ")
    head, _, trace = failure.partition("\n\n")
    assert head == f"draft.py:{line}, in step sources: {raising.strip()}\nKeyError: 'rows'"
    assert trace.startswith("Traceback (most recent call last)") and raising.strip() in trace
    assert read(agentic(workpiece, 0, 0) / "runs" / "0" / "run.json")["text"] == failure


async def test_tools_refuse_paths_outside_the_workspace_and_writes_to_the_reference(tmp_path, author):
    model = ScriptedModel(
        turn(("read_file", {"path": "../workpiece.json"}), ("write_file", {"path": "reference/sdk.md", "content": "x"})),
        turn(("edit_file", {"path": "program.py", "old_string": "@rl_env_build_step", "new_string": "@x"})),
        turn(("read_file", {"path": "program.py", "start_line": 4, "max_lines": 2})),
        turn(("submit", {"path": "missing.py", "notes": ""})),
        answer("done"),
        answer("done"),
        answer("done"),
        *GIVE_UP,
    )
    workpiece = tmp_path / "workpiece"
    with pytest.raises(BuildInfrastructureFailure):
        await author(workpiece, model)

    assert read(agentic(workpiece, 0, 0) / "author.json")["status"] == "handed_off"
    escape, reference, edit, window, missing = session_outputs(workpiece, 0)[:5]
    assert escape.startswith("error: '../workpiece.json' is not a path inside the workspace")
    assert reference == "error: reference/ is read-only"
    assert edit.startswith("error: old text has ") and edit.endswith(" matches in program.py")
    assert window.startswith("lines 4-5 of ") and "    4| " in window
    assert missing.startswith("error: no file 'missing.py'")


async def test_exhausted_author_session_holds_and_resumes_from_its_workspace(tmp_path, author):
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "notes.md", "content": "fixtures step is next"})),
        *GIVE_UP,
    )
    with pytest.raises(BuildInfrastructureFailure, match="saved for resumption"):
        await author(workpiece, model)
    session = read(agentic(workpiece, 0, 0) / "session.json")
    assert session["stop_reason"] == "max_turns"
    assert session["turns"] == CONFIG.turns
    assert read(agentic(workpiece, 0, 0) / "author.json")["status"] == "handed_off"
    resumed = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("submit", {"path": "program.py", "notes": ""})),
        turn(("submit", {"path": "program.py", "notes": ""})),
    )
    authored = await author(workpiece, resumed)
    assert authored.source == PROGRAM
    assert (agentic(workpiece, 0, 1) / "workspace" / "notes.md").read_text() == "fixtures step is next"
    assert read(agentic(workpiece, 0, 1) / "session.json")["mode"] == "handoff"


async def test_a_cut_turn_is_nudged_to_a_smaller_step_and_never_continued(tmp_path, author):
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        cut(),
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        cut(),
        turn(("submit", {"path": "program.py", "notes": ""})),
        turn(("submit", {"path": "program.py", "notes": ""})),
    )
    await author(workpiece, model)

    assert {r.policy.max_continuations for r in model.requests} == {0}
    assert {r.policy.max_tokens for r in model.requests} == {49_152}
    second = model.requests[1].messages
    assert second[-1]["role"] == "user" and str(second[-1]["content"]).startswith(
        "Your reply was cut off at the 49152-token output limit"
    )
    session = read(agentic(workpiece, 0, 0) / "session.json")
    assert session["length_nudges"] == 2 and session["nudges"] == 0 and session["length_turns"] == 2
    assert session["turns"] == 5 and session["continuations"] == 0


@pytest.mark.parametrize(
    ("first_session", "requests"),
    [
        ((turn(("list_files", {}), prompt=220_000, out=1_000),), 1),
        ((cut(), Completion("", "", (), FinishReason.LENGTH, Usage(215_000, 1_000, 1_000, 0), 1.0, 0.1, 0.9, 0, ())), 2),
    ],
    ids=["tool-turn", "cut-turns"],
)
async def test_the_context_cap_saves_the_conversation_and_hands_off_to_a_session_that_submits(
    tmp_path, author, first_session, requests
):
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(*first_session)
    with pytest.raises(BuildInfrastructureFailure, match="context"):
        await author(workpiece, model)

    first = read(agentic(workpiece, 0, 0) / "author.json")
    assert (first["status"], first["stop_reason"]) == ("handed_off", "context")
    assert len(read(agentic(workpiece, 0, 0) / "conversation.json")) > 2
    assert len(model.requests) == requests
    model.replies.extend(
        (
            turn(("write_file", {"path": "program.py", "content": PROGRAM})),
            turn(("submit", {"path": "program.py", "notes": ""})),
            turn(("submit", {"path": "program.py", "notes": ""})),
        )
    )
    authored = await author(workpiece, model)
    assert read(agentic(workpiece, 0, 1) / "session.json")["mode"] == "handoff"
    assert authored.source == PROGRAM and not model.replies


def prior_session(workpiece: Path, stop: str, last_prompt: int, turns_total: int) -> Path:
    """Round 0's accepted agentic session as an author call leaves it, for a build failure on its program."""
    session_dir = agentic(workpiece, 0, 0)
    (session_dir / "workspace").mkdir(parents=True)
    (session_dir / "workspace" / "program.py").write_text("# edited\n")
    conversation = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    (session_dir / "conversation.json").write_text(json.dumps(conversation))
    session = {"stop_reason": stop, "last_prompt_tokens": last_prompt, "turns_total": turns_total}
    (session_dir / "session.json").write_text(json.dumps(session))
    (session_dir / "author.json").write_text(
        json.dumps({"author_kind": "agentic", "status": "accepted", "stop_reason": stop})
    )
    (session_dir / "submit.json").write_text(json.dumps({"notes": "split the fixtures"}))
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
@pytest.mark.parametrize("round_no", [0, 1], ids=["same-round", "next-round"])
def test_build_failure_reenters_the_session_only_within_its_bounds(
    tmp_path, proposal, stop, last_prompt, turns_total, mode, round_no
):
    """The coordinator revises a build failure in the same round; a revision in the next round reads round 0."""
    workpiece = tmp_path / "workpiece"
    previous = prior_session(workpiece, stop, last_prompt, turns_total)
    plan = plan_session(workpiece, round_no, proposal, FAILURE, CONFIG, DEFAULT_POLICY.author)
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
    await author(workpiece, model, round_no=1, revision=FAILURE)

    sent = model.requests[0].messages
    assert [m["content"] for m in sent[:3]] == ["s", "u", "a"] and "KeyError 'rows'" in str(sent[3]["content"])
    session = read(agentic(workpiece, 1, 0) / "session.json")
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
    plan = plan_session(workpiece, 0, proposal, None, CONFIG, DEFAULT_POLICY.author)
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
        *GIVE_UP,
    )
    with pytest.raises(BuildInfrastructureFailure):
        await author(workpiece, model)

    write, notes = session_outputs(workpiece, 0)
    assert write == f"error: content holds {aa.WRITE_CHARS + 2961} characters, more than {aa.WRITE_CHARS}; use edit_file"
    assert notes.startswith("error: notes hold 4001 characters, more than 4000; nothing handed in (the loader accepts")
    target = agentic(workpiece, 0, 0)
    session = read(target / "session.json")
    assert session["rejected_tool_calls"] == 0
    assert [s["outcome"] for s in session["submissions"]] == ["refused_notes"]
    assert session["submit_refusals"] == {"refused_notes": 1} and session["first_submit_accepted"] is False
    submit = read(target / "submit.json")
    assert submit["accepted"] is False and [c["outcome"] for c in submit["calls"]] == ["refused_notes"]
