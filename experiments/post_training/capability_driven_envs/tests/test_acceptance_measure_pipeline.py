# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An arm's author, review and records run end to end through the stack's coordinator, offline.

The author is the arm's ``TreatedAgenticAuthor`` on a scripted model; builds run ``run_build`` in process on a toy
program that grades ``42``; validation returns synthetic rounds (``dryrun.synthetic_round``). The records and the
reducer then read back what the run wrote.
"""

import asyncio
import hashlib
import json
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

pytest.importorskip("taskforge")

from pydantic import BaseModel, TypeAdapter
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend
from taskcompendium.grading_result import GradeResult, Outcome
from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.agent import ReasoningReplay
from taskforge.llm.client import Completion, FinishReason, StructuredResult, ToolCall, Usage
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.sandbox.factories import MachineHost
from taskforge.stage20_triage.preflight import preflight
from taskforge.stage30_author.author import AuthorContext, program_manifest
from taskforge.stage30_author.load_process import ProgramManifest
from taskforge.stage40_build_with_program.agentic_author import SessionBuild
from taskforge.stage40_build_with_program.context import BuildInfrastructureFailure
from taskforge.stage40_build_with_program.program_sdk import load_program
from taskforge.stage40_build_with_program.run import (
    ResearchRequirement,
    TaskDraft,
    build_model_digest,
    load_draft,
    run_build,
)
from taskforge.stage40_build_with_program.services import AgentLimits, RLEnvBuildServices
from taskforge.stage50_validate.adversary import AdversaryMode, AdversaryRecord, Defect, SearchRole, SubmissionRecord
from taskforge.stage50_validate.round import ValidationContext, ValidationResult
from taskforge.stage_coordinator.coordinator import STORE_DIR, StageCoordinator, run_workpiece
from taskforge.stage_coordinator.event_log import InMemoryEventLog, JsonEventStore
from taskforge.stage_coordinator.stage_functions import StageFunctions
from taskforge.stage_coordinator.workpiece_machine import Terminal

from experiments.post_training.capability_driven_envs.acceptance_measure.author import (
    EVENTS_FILE,
    NON_CONVERGING,
    TreatedAgenticAuthor,
    TreatedAuthorSession,
)
from experiments.post_training.capability_driven_envs.acceptance_measure.dryrun import FA_EXPLANATION, synthetic_round
from experiments.post_training.capability_driven_envs.acceptance_measure.pipeline import LEDGER_DIR, no_repair
from experiments.post_training.capability_driven_envs.acceptance_measure.proposals import read_proposal
from experiments.post_training.capability_driven_envs.acceptance_measure.records import Allocation, allocation_rows
from experiments.post_training.capability_driven_envs.acceptance_measure.reducer import reduce_rows
from experiments.post_training.capability_driven_envs.acceptance_measure.review import shadow_decisions, treated_review
from experiments.post_training.capability_driven_envs.acceptance_measure.runtime import SAMPLING, arm_runtime
from experiments.post_training.capability_driven_envs.acceptance_measure.shadows import (
    ClaimVerdict,
    Judge,
    Verdict,
    shadow_round,
)
from experiments.post_training.capability_driven_envs.acceptance_measure.treatment import load_arms

FIXTURE = (
    Path(__file__).parent / "data" / "acceptance_measure" / "d01.uncertainty.statistics.point-estimation.proposal.json"
)
CAPABILITY = "d01.uncertainty.statistics.point-estimation"
PROGRAM = '''"""Grades 6 * 7 in process."""

from dataclasses import replace
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
    reference = await grade_candidate(
        context, draft.NO_MACHINE, package, AnswerType.TEXT, ANSWER_FORMAT, "(instruction)", "42"
    )
    if reference.status != Outcome.GRADED:
        raise BuildFailure(f"the reference was graded {reference.status}")
    return Grader(
        package, "Reply with the number.", "42", (), ("42",), 0.0, (graded_candidate("42", (), reference),)
    )


@rl_env_build_sink
async def assemble(context: RLEnvBuildContext, grader: Grader) -> TaskSpec:
    return draft.assemble(
        context.workpiece_id,
        "What is 6 * 7? Reply with the number alone.",
        AnswerType.TEXT,
        ANSWER_FORMAT,
        grader.package,
        Source(dataset="test", revision="r", row="0", importer_revision="test"),
        environment=draft.requirements(image=None).model_copy(
            update={"package_spec": replace(draft.SHELLSIM_CPYTHON, pypi_packages=None)}
        ),
    )


PROGRAM = rl_env_build_program(sources, grader, sink=assemble)
'''
EDITED = ("Reply with the number alone.", "Reply with the integer alone, digits only.")
"""A change to a step's source: a docstring edit leaves the program digest, so the resubmit would stay unchanged."""


def turn(*calls: tuple[str, Mapping[str, object]]) -> Completion:
    return Completion(
        content="",
        reasoning="thinking",
        tool_calls=tuple(ToolCall(f"call-{uuid.uuid4().hex}", name, json.dumps(args)) for name, args in calls),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(prompt_tokens=30_000, completion_tokens=2_000, reasoning_tokens=1_000, cached_tokens=29_000),
        wall_time=2.0,
        ttft=0.1,
        decode_time=1.9,
        continuations=0,
        attempts=(),
    )


class ScriptedModel:
    model = "scripted"

    def __init__(self, *replies: Completion) -> None:
        self.replies: deque[Completion] = deque(replies)
        self.requests: list[tuple[list[Message], dict]] = []

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        self.requests.append((list(messages), dict(request_fields or {})))
        return self.replies.popleft()

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        raise AssertionError("no structured call")


async def in_process_loader(source: str) -> ProgramManifest:
    return ProgramManifest.model_validate(program_manifest(load_program(source)))


def services(root: Path) -> RLEnvBuildServices:
    return RLEnvBuildServices(
        host=MachineHost.LAPTOP,
        factories={Backend.SHELLSIM.value: ShellSimMachineFactory()},
        images=None,
        web_tools=(),
        ledger=JsonlLedger(root / LEDGER_DIR),
        reasoning_replay=ReasoningReplay.KEEP,
        agent_limits=AgentLimits(max_turns=1, max_tokens=1024, max_calls=1),
    )


def proposal():
    raw = FIXTURE.read_bytes()
    return read_proposal(raw, hashlib.sha256(raw).hexdigest())


class Rounds:
    """Validation that returns a scripted synthetic round per round number, keyed to the context's workpiece."""

    def __init__(self, *rounds: ValidationResult) -> None:
        self.rounds = rounds

    async def __call__(self, draft: TaskDraft, context: ValidationContext) -> ValidationResult:
        template = self.rounds[context.round]
        return ValidationResult(
            context.workpiece_id,
            context.round,
            *[getattr(template, f) for f in ("lowering", "solver", "controls", "adversary", "early")],
        )


def coordinator(root: Path, arm: str, model: ScriptedModel, validate: Rounds) -> StageCoordinator:
    runtime = arm_runtime(load_arms()[arm])
    svc = services(root)
    store = str(root / STORE_DIR)
    digest = build_model_digest(model, SAMPLING, svc)

    def builds(context: AuthorContext, proposal_) -> SessionBuild:
        async def builder(source: str, run_dir: Path, *, invalidate: frozenset[str]) -> TaskDraft:
            return await run_build(
                load_program(source), proposal_, context.workpiece_id, run_dir, store, model, SAMPLING, svc,
                context.round, research=ResearchRequirement.OFF, invalidate=invalidate, model_digest=digest,
            )  # fmt: skip

        return SessionBuild(
            builder, store, svc, JsonlLedger(root / LEDGER_DIR).path_for(context.workpiece_id), in_process_loader
        )

    author = TreatedAgenticAuthor(runtime.author_config, model, SAMPLING, svc.ledger, builds, treatment=runtime.author)
    stages = StageFunctions(
        preflight, no_repair, author, load_program, run_build, validate,
        partial(treated_review, treatment=runtime.review, root=root),
    )  # fmt: skip
    return StageCoordinator(
        stages, runtime.budget, root, JsonEventStore(root), model, SAMPLING, svc, ResearchRequirement.OFF, digest
    )


def round_session(root: Path, workpiece: str, round_no: int, k: int = 0) -> Path:
    return root / "workpieces" / workpiece / "rounds" / str(round_no) / "agentic" / str(k)


def test_arm_a_refuses_an_unchanged_resubmission_after_review_and_the_records_recompute_the_funnel(tmp_path):
    workpiece = f"A.{CAPABILITY}"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "grades 42"})),
        turn(("submit", {"path": "program.py", "notes": "unchanged"})),
        turn(("edit_file", {"path": "program.py", "old_string": EDITED[0], "new_string": EDITED[1]})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "revised"})),
    )
    validate = Rounds(synthetic_round("x", 5, (Defect((1,), FA_EXPLANATION, True),)), synthetic_round("x", 4))

    terminal = asyncio.run(
        run_workpiece(coordinator(tmp_path, "A", model, validate), workpiece, proposal(), InMemoryEventLog())
    )

    assert terminal is Terminal.ACCEPTED and not model.replies
    events = [
        json.loads(line) for line in (round_session(tmp_path, workpiece, 1) / EVENTS_FILE).read_text().splitlines()
    ]
    assert [e["kind"] for e in events] == ["session_start", "unchanged_resubmit_refused"]
    resubmitted = json.loads((round_session(tmp_path, workpiece, 1) / "submit.json").read_text())
    assert resubmitted["accepted"] and resubmitted["unchanged_from_previous"] is False
    review = json.loads((tmp_path / "workpieces" / workpiece / "rounds" / "0" / "acceptance-review.json").read_text())
    assert review["decision"]["findings"] == review["stack_decision"]["findings"]
    assert review["finding_classes"] == ["label_normalisation"] and review["class_feedback"] is False

    allocation = Allocation("A", CAPABILITY, "d01", 0, 23, str(FIXTURE), "sha")
    rows = allocation_rows(tmp_path, allocation, proposal())
    kinds = [r["row"] for r in rows]
    assert kinds[0] == "proposal" and kinds[-1] == "terminal"
    submissions = [r for r in rows if r["row"] == "submission"]
    assert [(s["round"], s["charged"]) for s in submissions] == [(0, True), (1, False), (1, True)]
    assert [r["round"] for r in rows if r["row"] == "feedback"] == [0]
    assert rows[-1]["decision"] == "accepted" and rows[-1]["rounds"] == 2
    assert sum(r["tokens_in"] for r in rows if r["row"] == "cost" and r["stage"] == "author") == 7 * 30_000

    reduced = reduce_rows(rows, [CAPABILITY], ("A",))["arms"]["A"]
    assert reduced["funnel"] == {"accepted": 1} and reduced["acceptance_per_declared"] == 1.0
    assert reduced["tokens_by_stage"]["author"]["uncached_plus_output"] == 7 * (1_000 + 2_000)
    assert reduced["attribution"]["refused_unchanged_resubmit_refused"] == 1


def test_arm_c_accepts_a_saturated_round_that_arm_b_sends_back(tmp_path):
    for arm, expected in (("B", Terminal.ABANDONED), ("C", Terminal.ACCEPTED)):
        root = tmp_path / arm
        workpiece = f"{arm}.{CAPABILITY}"
        script = [
            turn(("write_file", {"path": "program.py", "content": PROGRAM})),
            turn(("run_build", {"path": "program.py"})),
            turn(("run_probe_battery", {"framing": "text", "transport_noise": "strip"})),
            turn(("submit", {"path": "program.py", "notes": "grades 42"})),
            turn(("submit", {"path": "program.py", "notes": "again"})),
            turn(("submit", {"path": "program.py", "notes": "again"})),
        ]
        rounds = [synthetic_round("x", 8)] * 4
        model = ScriptedModel(*script, *[turn(("submit", {"path": "program.py", "notes": "same"}))] * 40)
        terminal = asyncio.run(
            run_workpiece(coordinator(root, arm, model, Rounds(*rounds)), workpiece, proposal(), InMemoryEventLog())
        )

        assert terminal is expected, arm
        review = json.loads((root / "workpieces" / workpiece / "rounds" / "0" / "acceptance-review.json").read_text())
        assert review["l3"]["accept"] and review["shadow_decisions"]["l3"] == "accept"
        assert review["band_label"] == ("saturated" if arm == "C" else None)
        events = [
            json.loads(line) for line in (round_session(root, workpiece, 0) / EVENTS_FILE).read_text().splitlines()
        ]
        # The toy grader does not strip a byte-order mark, which the default contract promises: the battery misses it,
        # submit refuses twice, then hands the program in and records that the battery was never met.
        assert [e["kind"] for e in events] == [
            "session_start", "probe_battery", "battery_refused", "battery_refused", "battery_unmet",
        ]  # fmt: skip
        missed = [p["name"] for p in events[1]["probes"] if not p["met"]]
        assert missed == ["bom"] and events[1]["probes"][0]["name"] == "reference"


def test_the_identical_error_detector_hints_then_holds_as_infrastructure(tmp_path):
    runtime = arm_runtime(load_arms()["A"])
    build = SessionBuild(None, str(tmp_path / "store"), services(tmp_path), tmp_path / "ledger.jsonl", in_process_loader)
    session = TreatedAuthorSession(
        tmp_path / "session", "w", 0, proposal(), build, "0.0",
        treatment=runtime.author, revision=None, previous_digest=None, witnesses=(),
    )  # fmt: skip
    session.prepare(None)
    [read_file] = [t for t in session.tools(()) if t.name == "read_file"]

    results = [asyncio.run(read_file.handler({"path": "missing.py"})) for _ in range(runtime.author.hint_after)]
    assert results[0].startswith("error:") and "consecutive identical result" not in results[0]
    assert "This is the 3th consecutive identical result from `read_file`" in results[-1]
    asyncio.run(read_file.handler({"path": "program.py"}))
    with pytest.raises(BuildInfrastructureFailure) as held:
        for _ in range(runtime.author.hold_after):
            asyncio.run(read_file.handler({"path": "missing.py"}))
    assert held.value.cause == NON_CONVERGING


class JudgeModel:
    """Answers every structured call with ``verdict`` and records the materials it was shown."""

    model = "judge"

    def __init__(self, verdict: ClaimVerdict) -> None:
        self.verdict = verdict
        self.materials: list[dict] = []

    async def complete_structured(self, messages, output_type, policy) -> StructuredResult:
        self.materials.append(json.loads(messages[1]["content"].split("\n", 1)[1]))
        usage = Usage(prompt_tokens=20_000, completion_tokens=500, reasoning_tokens=0, cached_tokens=0)
        done = Completion("{}", "", (), FinishReason.STOP, usage, 1.0, 0.1, 0.9, 0, ())
        return StructuredResult(self.verdict, (done,))


def test_the_shadow_checks_read_the_round_claims_and_judge_the_decisive_ones_only(tmp_path):
    workpiece = f"A.{CAPABILITY}"
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": PROGRAM})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "grades 42"})),
    )
    asyncio.run(
        run_workpiece(
            coordinator(tmp_path, "A", model, Rounds(synthetic_round("x", 4))), workpiece, proposal(), InMemoryEventLog()
        )
    )
    round_dir = tmp_path / "workpieces" / workpiece / "rounds" / "0"
    result = synthetic_round(workpiece, 5, (Defect((1,), FA_EXPLANATION, True), Defect((2,), "noted, harmless", False)))
    fa = result.adversary[0]
    second = replace(fa.submissions[0], ordinal=2)
    result = replace(result, adversary=(replace(fa, submissions=(*fa.submissions, second)), *result.adversary[1:]))
    grade = GradeResult(Outcome.GRADED, 1.0)
    submissions = tuple(
        SubmissionRecord(i, 1, f"s{i}", reply, (), grade, False, 1.0) for i, reply in ((1, "CONVERGES"), (2, "42 "))
    )
    record = AdversaryRecord(
        SearchRole.FALSE_ACCEPT, "m", "p", AdversaryMode.WHITEBOX, "b", "s", result.adversary[0].report, "g", submissions
    )
    attempt = round_dir / result.adversary[0].attempt_file
    attempt.parent.mkdir(parents=True)
    attempt.write_text(json.dumps({"adversary": json.loads(TypeAdapter(AdversaryRecord).dump_json(record))}))
    context = ValidationContext(workpiece, 0, round_dir, proposal(), str(tmp_path / STORE_DIR), 3, 1)
    judge = JudgeModel(
        ClaimVerdict(
            verdict=Verdict.FINDER_FALSE,
            false_class="fa_requirement_not_stated",
            defect_class=None,
            plausible_solver_output=False,
            contract_quote="",
            rationale="invented strictness",
        )
    )
    draft = load_draft(round_dir / "draft")

    shadow = asyncio.run(
        shadow_round(
            draft, context, result, arm_runtime(load_arms()["A"]).review.taxonomy, Judge(judge, SAMPLING, "prompt", 2)
        )
    )

    first, second = shadow["claims"]
    assert (first["ordinal"], first["needs_addressing"], first["class"]) == (1, True, "label_normalisation")
    assert first["l4"]["precondition"] == "fail" and first["l4"]["quotes"] == 1
    assert first["judge"]["verdict"] == "finder_false" and first["judge"]["tokens_in"] == 20_000
    assert second["judge"] is None, "only decisive claims are judged"
    [materials] = judge.materials
    assert materials["task_contract"] == "What is 6 * 7? Reply with the number alone."
    assert materials["claim"]["candidate_reply"] == "CONVERGES" and materials["reference_reply"] == "42"
    on_disk = json.loads((round_dir / "acceptance-shadow.json").read_text())
    decisions = asyncio.run(shadow_decisions(result, on_disk, arm_runtime(load_arms()["A"]).review.band))
    assert decisions == {"l4": "accept", "judge": "accept", "realistic_only": "accept"}
