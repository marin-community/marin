# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The agentic author offline: a scripted model drives whole author calls against the real loader and ``run_build``.

``ScriptedModel`` (``scripted_author``) answers each request with the next queued ``Completion``, so a test
sets each turn's tool calls and token usage exactly; one author call runs as many sessions as its replies
reach. The sessions' ``ProgramBuilder`` runs ``run_build`` in process on a step store under ``tmp_path``,
recording its step spans to a ``JsonlLedger``; the toy ``PROGRAM`` grades in process and calls no model.
"""

import inspect
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind, Ledger
from taskforge.llm.client import Completion, FinishReason, Usage
from taskforge.llm.policy import LLMPolicy
from taskforge.stage10_proposal.model import SolverMachine, TaskProposal, Verification
from taskforge.stage30_author import load_process
from taskforge.stage30_author.author import (
    AuthorContext,
    AuthoredProgram,
    AuthorFailure,
    Revision,
    RevisionKind,
    VerificationChange,
)
from taskforge.stage30_author.load_process import ProgramManifest, load_program_in_process
from taskforge.stage40_build_with_program import agentic_author as aa
from taskforge.stage40_build_with_program.agentic_author import (
    AgenticAuthor,
    ProgramBuilder,
    SessionBuild,
    SessionMode,
    plan_session,
)
from taskforge.stage40_build_with_program.context import BuildExpired, BuildInfrastructureFailure, ExpiredClock
from taskforge.stage40_build_with_program.glm_author import AUTHOR_DIR, GlmAuthor
from taskforge.stage40_build_with_program.infrastructure import InfrastructureCause
from taskforge.stage40_build_with_program.program_sdk import load_program, sdk_exports
from taskforge.stage40_build_with_program.run import DRAFT_DIR, ResearchRequirement, TaskDraft, load_draft, run_build
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage40_build_with_program.step_store import StoreCorruption
from taskforge.stage40_build_with_program.template import standard
from tests.stage40_build_with_program.conftest import SAMPLING, WORKPIECE, declaring
from tests.stage40_build_with_program.scripted_author import (
    CONFIG,
    PROGRAM,
    ScriptedModel,
    answer,
    in_process_loader,
    turn,
)

FAILURE = Revision(RevisionKind.BUILD_FAILURE, "fixtures: KeyError 'rows'")
FINDINGS = Revision(RevisionKind.MODIFY, "the grader leaves the reply's format unenforced")
TO_ANSWER_IN_BOX = VerificationChange(Verification.CODE, Verification.ANSWER_IN_BOX, "the answer is one number")
SCHEDULING = BuildInfrastructureFailure(InfrastructureCause.SCHEDULING_TIMEOUT, "the machine factory gave up waiting")
NO_FACTORY = BuildInfrastructureFailure(InfrastructureCause.NO_FACTORY, "this host has no docker machine factory")
ALWAYS = 1_000
"""A failure count no session reaches: the host fails every build."""
GIVE_UP = (answer("no"), answer("no"), answer("no"))
"""A session's replies without a tool call: it ends unsubmitted after its nudges."""


def cut(reasoning_tokens: int = 49_152) -> Completion:
    """A turn cut at the output limit while reasoning, before any tool call."""
    usage = Usage(30_000, reasoning_tokens, reasoning_tokens, 29_000)
    return Completion("", "r" * 100, (), FinishReason.LENGTH, usage, 900.0, 0.1, 899.0, 0, ())


def session_ledger(root: Path) -> JsonlLedger:
    return JsonlLedger(root / "ledger" / "author_session")


def session_build(root: Path, model: ScriptedModel, services: RLEnvBuildServices, round_no: int) -> SessionBuild:
    """In-process builds on ``root/store``, recording their spans to the session ledger under ``root``."""
    store = str(root / "store")
    ledger = session_ledger(root)
    build_services = replace(services, ledger=ledger)

    async def builder(
        source: str,
        run_dir: Path,
        *,
        invalidate: frozenset[str],
        proposal: TaskProposal,
        verification_change: VerificationChange | None,
    ) -> TaskDraft:
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
            verification_change=verification_change,
        )

    return SessionBuild(builder, store, services, ledger.path_for(WORKPIECE), in_process_loader)


async def author_call(
    build: SessionBuild,
    model: ScriptedModel,
    ledger: Ledger,
    workpiece: Path,
    proposal: TaskProposal,
    round_no: int = 0,
    revision: Revision | None = None,
) -> AuthoredProgram:
    """One ``AgenticAuthor`` call for ``WORKPIECE`` in ``workpiece``, whose sessions build with ``build``."""

    def builds(context: AuthorContext, proposal: TaskProposal) -> SessionBuild:
        return build

    author = AgenticAuthor(CONFIG, model, LLMPolicy(), ledger, builds)
    return await author(AuthorContext(WORKPIECE, round_no, workpiece), proposal, revision, {})


@pytest.fixture
def author(services, proposal, ledger):
    """``author(workpiece_dir, model, round_no=0, revision=None)``: one author call on the in-process builds.

    The call's sessions build under ``workpiece_dir``'s parent.
    """

    async def run(
        workpiece_dir: Path, model: ScriptedModel, round_no: int = 0, revision: Revision | None = None
    ) -> AuthoredProgram:
        build = session_build(workpiece_dir.parent, model, services, round_no)
        return await author_call(build, model, ledger, workpiece_dir, proposal, round_no, revision)

    return run


def agentic(workpiece: Path, round_no: int, k: int) -> Path:
    return workpiece / "rounds" / str(round_no) / "agentic" / str(k)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def tool_outputs(model: ScriptedModel) -> list[str]:
    """The tool results the model saw, in order, from the last request's conversation."""
    return [str(m["content"]) for m in model.requests[-1].messages if m["role"] == "tool"]


def offered_tools(model: ScriptedModel) -> set[str]:
    """The names of the tools the session's first request offered."""
    tools = model.requests[0].fields["tools"]
    assert isinstance(tools, list)
    return {tool["function"]["name"] for tool in tools}


def declaring_code(proposal: TaskProposal) -> TaskProposal:
    """``proposal`` declaring code verification, which ``PROGRAM``'s exact-match grader does not implement."""
    return replace(proposal, header=replace(proposal.header, verification=Verification.CODE))


def tool_texts(model: ScriptedModel) -> list[str]:
    """Every tool result any request of the session carried."""
    return [str(m["content"]) for request in model.requests for m in request.messages if m["role"] == "tool"]


def session_outputs(workpiece: Path, k: int) -> list[str]:
    """The tool results of round 0's session ``k``, in order, as its transcript keeps them."""
    transcript = json.loads((agentic(workpiece, 0, k) / "transcript.json").read_text())
    return [result["output"] for entry in transcript for result in entry.get("tool_results", [])]


@dataclass
class HostFailingBuilder:
    """A ``ProgramBuilder`` whose first ``failures`` builds the host fails with ``failure``; ``calls`` counts builds."""

    builder: ProgramBuilder
    failure: BuildInfrastructureFailure
    failures: int
    calls: int = 0

    async def __call__(
        self,
        source: str,
        run_dir: Path,
        *,
        invalidate: frozenset[str],
        proposal: TaskProposal,
        verification_change: VerificationChange | None,
    ) -> TaskDraft:
        self.calls += 1
        if self.calls <= self.failures:
            raise BuildInfrastructureFailure(self.failure.cause, self.failure.message)
        return await self.builder(
            source, run_dir, invalidate=invalidate, proposal=proposal, verification_change=verification_change
        )


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
    tmp_path, services, proposal, ledger, machine, environment
):
    """The program's grader step and ``run_grade_candidate`` both pass the task's requirements as they are."""
    declared = declaring(proposal, machine)
    program = PROGRAM.replace("context, draft.NO_MACHINE,", f"context, {environment},").replace(
        "environment=draft.requirements(image=None).model_copy(\n"
        '            update={"package_spec": replace(draft.SHELLSIM_CPYTHON, pypi_packages=None)}\n        )',
        f"environment={environment}",
    )
    assert program.count(environment) == 2
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": program})),
        turn(("run_build", {"path": "program.py"})),
        turn(("run_grade_candidate", {"reply": "42"})),
        turn(("run_grade_candidate", {"reply": "41"})),
        turn(("submit", {"path": "program.py", "notes": "grades 6 * 7"})),
    )
    build = session_build(tmp_path, model, services, 0)
    await author_call(build, model, ledger, tmp_path / "workpiece", declared)

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
    wrote, loader, refused, built, graded = tool_outputs(model)
    assert wrote == f"wrote {len(PROGRAM.encode())} bytes to program.py"
    assert loader.startswith("loads: program ")
    assert refused.startswith("not handed in: the loader accepts it") and "never built in this session" in refused
    assert built.startswith("built: program ") and "sources MISS" in built and "grader MISS" in built
    assert "reference: graded reward 1.0" in built
    assert "empty answer: submission_failure reward 0.0 (graded by the library; ceiling 0.0)" in built
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
    graded = (Outcome.GRADED, 1.0)

    async def credit_everything(
        services, workpiece_id, requirements, package, answer_type, answer_format, instruction, reply, **kw
    ):
        status, reward = graded if reply == "" else (Outcome.GRADED, 1.0 if reply == "42" else 0.0)
        return GradeResult(status=status, reward=reward)

    monkeypatch.setattr(aa, "grade_candidate_on_host", credit_everything)
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "ships anyway"})),
        turn(("submit", {"path": "program.py", "notes": "ships anyway"})),
    )
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
    await GlmAuthor(one_shot_model, SAMPLING, ledger)(
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


SPINNING_CLASS = "class Spin:\n    while True:\n        pass\n\n\n"
"""An output model whose class body never returns; the template with it added is a program that spins on load."""


async def test_a_module_body_that_never_returns_is_refused_by_the_loader_tool(tmp_path, services, proposal, ledger):
    """The default loader runs in a child process, so a spinning module body costs the session one refusal."""

    async def bounded(source: str) -> ProgramManifest:
        return await load_program_in_process(source, sdk_exports(), wall_timeout=2)

    spinning = inspect.getsource(standard).replace("PROGRAM = ", f"{SPINNING_CLASS}PROGRAM = ")
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
    build = replace(session_build(workpiece.parent, model, services, 0), loader=bounded)

    with pytest.raises(AuthorFailure):
        await author_call(build, model, ledger, workpiece, proposal)

    _, loader, submitted = session_outputs(workpiece, 0)[:3]
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
    build = replace(session_build(tmp_path, model, services, 0), loader=aa.load_in_child)

    with pytest.raises(BuildInfrastructureFailure) as raised:
        await author_call(build, model, ledger, workpiece, proposal)

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
    build = session_build(tmp_path, model, services, 0)
    flaky = HostFailingBuilder(build.builder, SCHEDULING, failures=1)

    authored = await author_call(replace(build, builder=flaky), model, ledger, workpiece, proposal)

    assert authored.source == PROGRAM and flaky.calls == 2
    assert tool_outputs(model)[1].startswith("built: program ")
    assert read(agentic(workpiece, 0, 0) / "runs" / "0" / "run.json")["status"] == "built"


async def test_a_container_proposal_on_a_host_without_an_image_builder_ends_before_the_author_runs(
    tmp_path, services, proposal, ledger
):
    container = declaring(proposal, None)
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel()
    build = session_build(tmp_path, model, services, 0)

    with pytest.raises(BuildInfrastructureFailure) as raised:
        await author_call(build, model, ledger, workpiece, container)

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
    build = session_build(tmp_path, model, services, 0)
    failing = HostFailingBuilder(build.builder, failure, failures=ALWAYS)

    with pytest.raises(BuildInfrastructureFailure) as raised:
        await author_call(replace(build, builder=failing), model, ledger, workpiece, proposal)

    assert raised.value.cause == failure.cause and failing.calls == builds
    assert read(agentic(workpiece, 0, 0) / "runs" / "0" / "run.json")["status"] == "infrastructure"
    assert len(model.replies) == 1, "the session ended at the build"
    assert not any("infrastructure" in text for text in tool_texts(model))
    assert not (agentic(workpiece, 0, 0) / "author.json").exists(), "the call wrote no record and spent no revision"


async def test_a_grade_the_host_keeps_failing_ends_the_round_unseen_by_the_author(
    tmp_path, services, proposal, ledger, monkeypatch
):
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
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("run_grade_candidate", {"reply": "41"})),
        turn(("submit", {"path": "program.py", "notes": "n"})),
    )
    build = session_build(tmp_path, model, services, 0)

    with pytest.raises(BuildInfrastructureFailure) as raised:
        await author_call(build, model, ledger, tmp_path / "workpiece", proposal)

    assert raised.value.cause == InfrastructureCause.HOST_UNREACHABLE
    assert graded == ["41"] * (1 + aa.HOST_RETRIES)
    assert tool_outputs(model)[1].startswith("built: program ")
    assert not any("grading failed" in text or "infrastructure" in text for text in tool_texts(model))


async def test_a_build_past_its_clock_ends_the_round_unseen_by_the_author_and_unretried(
    tmp_path, services, proposal, ledger
):
    builds: list[Path] = []

    async def expiring(source: str, run_dir: Path, **build: object) -> TaskDraft:
        builds.append(run_dir)
        raise BuildExpired("ran for more than 1800 seconds without returning", ExpiredClock.STEP, "grader")

    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "n"})),
    )
    build = replace(session_build(tmp_path, model, services, 0), builder=expiring)

    with pytest.raises(BuildExpired) as raised:
        await author_call(build, model, ledger, workpiece, proposal)

    assert (raised.value.clock, raised.value.step) == (ExpiredClock.STEP, "grader") and len(builds) == 1
    run = read(agentic(workpiece, 0, 0) / "runs" / "0" / "run.json")
    assert (run["status"], run["clock"], run["step"]) == ("expired", "step", "grader")
    assert run["text"] == "ran for more than 1800 seconds without returning"
    assert len(model.replies) == 1, "the session ended at the build"
    assert not any("ran for more than" in text for text in tool_texts(model))
    assert not (agentic(workpiece, 0, 0) / "author.json").exists(), "the call wrote no record and spent no revision"


async def test_a_storage_fault_in_a_build_ends_the_round_unseen_by_the_author(tmp_path, services, proposal, ledger):
    async def corrupt_store(source: str, run_dir: Path, **build: object) -> TaskDraft:
        raise StoreCorruption("Blob sha256:00 of step grader does not match its digest")

    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "n"})),
    )
    build = replace(session_build(tmp_path, model, services, 0), builder=corrupt_store)

    with pytest.raises(StoreCorruption):
        await author_call(build, model, ledger, workpiece, proposal)

    assert read(agentic(workpiece, 0, 0) / "runs" / "0" / "run.json")["status"] == "storage"
    assert len(model.replies) == 1, "the session ended at the build"
    assert not any("does not match its digest" in text for text in tool_texts(model))
    assert not (agentic(workpiece, 0, 0) / "author.json").exists(), "the call wrote no record and spent no revision"


async def test_an_os_error_the_program_raises_reaches_the_author(tmp_path, author):
    missing = PROGRAM.replace(
        '    return Sources(notes="", findings=())', '    raise FileNotFoundError("fixtures/matrices.json")'
    )
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": missing})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "n"})),
        turn(("submit", {"path": "program.py", "notes": "n"})),
    )
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
    model = ScriptedModel(
        turn(("write_file", {"path": "draft.py", "content": crashing})),
        turn(("run_build", {"path": "draft.py"})),
        turn(("submit", {"path": "draft.py", "notes": "n"})),
        turn(("submit", {"path": "draft.py", "notes": "n"})),
    )
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
    with pytest.raises(AuthorFailure):
        await author(workpiece, model)

    assert read(agentic(workpiece, 0, 0) / "author.json")["status"] == "handed_off"
    escape, reference, edit, window, missing = session_outputs(workpiece, 0)[:5]
    assert escape.startswith("error: '../workpiece.json' is not a path inside the workspace")
    assert reference == "error: reference/ is read-only"
    assert edit.startswith("error: old text has ") and edit.endswith(" matches in program.py")
    assert window.startswith("lines 4-5 of ") and "    4| " in window
    assert missing.startswith("error: no file 'missing.py'")


async def test_unsubmitted_session_hands_off_once_then_the_call_fails_and_spends_a_revision(tmp_path, author):
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        turn(("write_file", {"path": "notes.md", "content": "fixtures step is next"})),
        answer("I think it is done."),
        answer("Done."),
        answer("Still done."),
        *GIVE_UP,
    )
    with pytest.raises(AuthorFailure, match="2 agentic sessions of round 0 ended without a submission"):
        await author(workpiece, model)

    first_session_last = model.requests[3].messages
    assert [m["content"] for m in first_session_last if m["role"] == "user"][-2:] == [aa.NUDGE, aa.NUDGE]
    session = read(agentic(workpiece, 0, 0) / "session.json")
    assert session["stop_reason"] == "unsubmitted" and session["nudges"] == 2
    assert read(agentic(workpiece, 0, 0) / "author.json")["status"] == "handed_off"
    opening = model.requests[4].messages
    assert opening[-1]["role"] == "user" and "ended without a submission (unsubmitted)" in str(opening[-1]["content"])
    assert (agentic(workpiece, 0, 1) / "workspace" / "notes.md").read_text() == "fixtures step is next"
    assert read(agentic(workpiece, 0, 1) / "session.json")["mode"] == "handoff"
    assert read(agentic(workpiece, 0, 1) / "author.json")["status"] == "author_failed"


async def test_context_guard_hands_off_to_fresh_sessions_until_the_call_cap(tmp_path, author):
    """A session over the context cap after any turn hands off, a handoff session included; the call stops at
    MAX_SESSIONS_PER_CALL."""
    workpiece = tmp_path / "workpiece"
    sessions = aa.MAX_SESSIONS_PER_CALL
    model = ScriptedModel(*[turn(("list_files", {}), prompt=220_000, out=1_000)] * sessions)
    with pytest.raises(AuthorFailure):
        await author(workpiece, model)

    records = [read(agentic(workpiece, 0, k) / "author.json") for k in range(sessions)]
    assert [record["stop_reason"] for record in records] == ["context"] * sessions
    assert [record["status"] for record in records] == ["handed_off"] * (sessions - 1) + ["author_failed"]
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


async def test_cut_turns_past_the_context_cap_hand_off_to_a_session_that_submits(tmp_path, author):
    workpiece = tmp_path / "workpiece"
    model = ScriptedModel(
        cut(),
        Completion("", "", (), FinishReason.LENGTH, Usage(215_000, 1_000, 1_000, 0), 1.0, 0.1, 0.9, 0, ()),
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("submit", {"path": "program.py", "notes": ""})),
        turn(("submit", {"path": "program.py", "notes": ""})),
    )
    authored = await author(workpiece, model)

    first = read(agentic(workpiece, 0, 0) / "author.json")
    assert (first["status"], first["stop_reason"]) == ("handed_off", "context")
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
    plan = plan_session(workpiece, round_no, proposal, FAILURE, CONFIG)
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


async def test_a_session_answering_findings_swaps_the_verifier_kind_and_builds_and_submits_the_new_kind(
    tmp_path, services, proposal, ledger
):
    workpiece = tmp_path / "workpiece"
    prior_session(workpiece, "submitted", 100_000, 30)
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("change_verification", {"kind": "answer_in_box", "reason": TO_ANSWER_IN_BOX.reason})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "grades 6 * 7 exactly"})),
    )
    build = session_build(tmp_path, model, services, 1)

    authored = await author_call(build, model, ledger, workpiece, declaring_code(proposal), 1, FINDINGS)

    assert "change_verification" in offered_tools(model)
    refused, changed, built = tool_outputs(model)[1:4]
    assert refused.startswith("build failed (build_failure in step verification)")
    assert "declares code verification but the built grader is answer_in_box" in refused
    assert changed.startswith("verification changed from code to answer_in_box;")
    assert built.startswith("built: program ")
    assert (authored.source, authored.verification_change) == (PROGRAM, TO_ANSWER_IN_BOX)
    record = read(agentic(workpiece, 1, 0) / "author.json")
    assert (record["status"], record["tested"]) == ("accepted", True)
    assert record["verification_change"] == {
        "declared": "code",
        "revised": "answer_in_box",
        "reason": TO_ANSWER_IN_BOX.reason,
    }
    session = agentic(workpiece, 1, 0)
    assert "verification: answer_in_box" in (session / "workspace" / "proposal.md").read_text()
    provenance = load_draft(session / "runs" / "1" / DRAFT_DIR).provenance
    assert (provenance.verification, provenance.verification_change) == (Verification.ANSWER_IN_BOX, TO_ANSWER_IN_BOX)


async def test_a_session_starts_from_the_verification_change_its_revision_carries(tmp_path, services, proposal, ledger):
    workpiece = tmp_path / "workpiece"
    prior_session(workpiece, "submitted", 100_000, 30)
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "n"})),
    )
    revision = replace(FINDINGS, verification_change=TO_ANSWER_IN_BOX)

    build = session_build(tmp_path, model, services, 1)

    authored = await author_call(build, model, ledger, workpiece, declaring_code(proposal), 1, revision)

    assert tool_outputs(model)[1].startswith("built: program ")
    assert (authored.source, authored.verification_change) == (PROGRAM, TO_ANSWER_IN_BOX)


async def test_change_verification_refuses_what_it_cannot_swap_to_and_the_declared_kind_undoes_a_swap(tmp_path, author):
    workpiece = tmp_path / "workpiece"
    prior_session(workpiece, "submitted", 100_000, 30)
    model = ScriptedModel(
        turn(("change_verification", {"kind": "composite", "reason": "two graders"})),
        turn(("change_verification", {"kind": "answer_in_box", "reason": "it already is"})),
        turn(("change_verification", {"kind": "code", "reason": " "})),
        turn(("change_verification", {"kind": "code", "reason": "the answer needs execution"})),
        turn(("change_verification", {"kind": "answer_in_box", "reason": "one number after all"})),
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "n"})),
    )

    authored = await author(workpiece, model, round_no=1, revision=FINDINGS)

    composite, same, blank, swapped, undone, _, built = tool_outputs(model)
    assert composite.startswith("error: tool arguments do not match the schema at $.kind: enum")
    assert same == "error: the verification is already answer_in_box; nothing changed"
    assert blank == "error: verification unchanged: a verification change states its reason"
    assert swapped.startswith("verification changed from answer_in_box to code;")
    assert undone.startswith("verification changed from code to answer_in_box (the round's own kind);")
    assert built.startswith("built: program ")
    assert (authored.source, authored.verification_change) == (PROGRAM, None)
    assert read(agentic(workpiece, 1, 0) / "author.json")["verification_change"] is None


async def test_round_zero_and_a_build_failure_offer_no_change_verification(tmp_path, author):
    """Each call's first session tries the tool and gives up; its handoff gives up too, failing the call."""

    def calling_it() -> ScriptedModel:
        return ScriptedModel(
            turn(("change_verification", {"kind": "code", "reason": "the answer needs execution"})),
            *GIVE_UP,
            *GIVE_UP,
        )

    round_zero, after_failure = calling_it(), calling_it()
    with pytest.raises(AuthorFailure):
        await author(tmp_path / "zero" / "workpiece", round_zero)
    failed = tmp_path / "failed" / "workpiece"
    prior_session(failed, "submitted", 100_000, 30)
    with pytest.raises(AuthorFailure):
        await author(failed, after_failure, round_no=1, revision=FAILURE)

    for model in (round_zero, after_failure):
        assert "change_verification" not in offered_tools(model)
        assert tool_texts(model)[0].startswith("error: unknown tool 'change_verification'")


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
        *GIVE_UP,
    )
    with pytest.raises(AuthorFailure):
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
