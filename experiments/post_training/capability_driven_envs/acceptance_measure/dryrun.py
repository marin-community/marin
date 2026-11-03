# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render what each arm sends and decides, with no model and no machine, and diff the arms.

For each proposal and arm, ``render`` drives the arm's real author (``TreatedAgenticAuthor._session``, built by
``runtime.arm_runtime`` exactly as a launch builds it) against a recording model that captures the first request and
then fails it, so the session records its opening and stops. The surfaces it returns:

* the author's system and user turns, the tool definitions as sent to the API, the turn's sampling, and the files
  the session workspace holds;
* the finder briefs (``adversary_brief``) of each configured search on a fixed toy task, and the validation plan;
* the arm's review on three synthetic rounds: a saturated 8/8 round whose FA search passed, a round with an FA
  defect, and an in-band round;
* every arm key's content (``policy:<key>``).

``diff_arms`` reports, for each pair of arms, the surfaces that differ, and ``check_arms`` refuses a pair whose
differences fall outside the surfaces the changed keys declare (``treatment.intended_surfaces``) or whose intended
treatment did not show up.
"""

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from rolloutengine.models import TaskRolloutSpec
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, InlineFile, PlainText, Source, TaskResource
from taskforge.ledger.records import DiscardingLedger
from taskforge.llm.agent import AgentInterrupted, ReasoningReplay
from taskforge.llm.client import Completion, ModelUnavailable, StructuredResult
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.sandbox.factories import MachineHost
from taskforge.spec.controls import ControlKind
from taskforge.spec.draft import TOOL_HARNESS, assemble, requirements
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage30_author.author import AuthorContext
from taskforge.stage40_build_with_program.agentic_author import AGENTIC_DIR, WORKSPACE_DIR, SessionBuild
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage50_validate.adversary import (
    FALSE_ACCEPT,
    FALSE_REJECT,
    AdversaryMode,
    AdversarySpec,
    Defect,
    SearchReport,
    adversary_brief,
)
from taskforge.stage50_validate.controls import ControlVerdict
from taskforge.stage50_validate.evidence import RewardStats
from taskforge.stage50_validate.round import (
    AdversaryResult,
    ControlResult,
    Lowering,
    LoweringStatus,
    SolverEvidence,
    SubmissionResult,
    TrialResult,
    TrialStatus,
    ValidationResult,
)
from taskforge.stage60_review.decision import Decision
from verifyit.spec import NumericSpec

from experiments.post_training.capability_driven_envs.acceptance_measure.author import TreatedAgenticAuthor
from experiments.post_training.capability_driven_envs.acceptance_measure.review import treated_review
from experiments.post_training.capability_driven_envs.acceptance_measure.runtime import (
    BUILD_AGENT_LIMITS,
    SAMPLING,
    TRIAL_LIMITS,
    ArmRuntime,
    arm_runtime,
)
from experiments.post_training.capability_driven_envs.acceptance_measure.treatment import (
    Treatment,
    changed_keys,
    intended_surfaces,
)

RENDERED = "dry run: the request is recorded and the session stops here"
FA_EXPLANATION = (
    "The grader lowercases the label before comparing, so 'CONVERGES' earns full credit although the instruction "
    'says "the label must be one of converges or diverges, exactly as written".'
)
REQUIRED = {
    ("A", "B"): {"author.system", "author.tools", "author.workspace", "author.submit_rule", "review.modify_feedback"},
    ("B", "C"): {"review.saturated_decision"},
}
"""Surfaces that must differ between two arms, so a treatment that silently failed to apply is caught."""


def sha(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


class RecordingModel:
    """A ``ModelClient`` that records each request and fails it, so a session renders its opening and stops."""

    model = "dry-run"

    def __init__(self) -> None:
        self.requests: list[tuple[list[Message], LLMPolicy, dict]] = []

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping | None = None
    ) -> Completion:
        self.requests.append((list(messages), policy, dict(request_fields or {})))
        raise ModelUnavailable(RENDERED, ())

    async def complete_structured(self, messages, output_type, policy) -> StructuredResult:
        raise AssertionError("the dry run makes no structured call")


async def _never(*args: object, **kwargs: object):
    raise AssertionError("the dry run builds and loads nothing")


def dry_services() -> RLEnvBuildServices:
    return RLEnvBuildServices(
        host=MachineHost.LAPTOP,
        factories={},
        images=None,
        web_tools=(),
        ledger=DiscardingLedger(),
        reasoning_replay=ReasoningReplay.KEEP,
        agent_limits=BUILD_AGENT_LIMITS,
    )


async def render_author(runtime: ArmRuntime, proposal: TaskProposal, root: Path) -> dict:
    model = RecordingModel()
    workpiece_dir = root / "workpieces" / "dry"
    build = SessionBuild(_never, str(root / "store"), dry_services(), root / "ledger.jsonl", _never)
    author = TreatedAgenticAuthor(
        runtime.author_config,
        model,
        SAMPLING,
        DiscardingLedger(),
        lambda c, p: build,
        runtime.policy,
        treatment=runtime.author,
    )
    try:
        await author._session(build, AuthorContext("dry", 0, workpiece_dir), proposal, None)
    except AgentInterrupted as interrupted:
        if RENDERED not in str(interrupted):
            raise
    [(messages, policy, fields)] = model.requests
    workspace = workpiece_dir / "rounds" / "0" / AGENTIC_DIR / "0" / WORKSPACE_DIR
    return {
        "author.system": messages[0]["content"],
        "author.user": [m["content"] for m in messages[1:]],
        "author.tools": fields.get("tools"),
        "author.sampling": asdict(policy),
        "author.workspace": {
            str(p.relative_to(workspace)): sha(p.read_bytes()) for p in sorted(workspace.rglob("*")) if p.is_file()
        },
    }


def toy_spec() -> TaskRolloutSpec:
    numbers = TaskResource(
        path="workspace/numbers.txt", source=InlineFile(content_base64=base64.b64encode(b"6 7\n").decode())
    )
    task = assemble(
        "dry",
        'Multiply the two numbers in /workspace/numbers.txt. Reply with JSON exactly: {"product": <integer>}.',
        AnswerType.TEXT,
        PlainText(),
        verifyit_package(NumericSpec(expected="42", tolerance_abs=0.0, tolerance_rel=0.0)),
        Source(dataset="taskforge", revision="dry", row="dry", importer_revision="dry"),
        environment=requirements(image=None),
        files=(numbers,),
    )
    return TaskRolloutSpec(
        task=task, harness=TOOL_HARNESS, task_machine=None, verifier_machine=None, limits=TRIAL_LIMITS
    )


def render_validation(runtime: ArmRuntime) -> dict:
    spec = toy_spec()
    rendered: dict = {
        "validation.plan": {
            "k": runtime.solver_k,
            "max_retries": runtime.max_retries,
            "retry_backoff": asdict(runtime.retry_backoff),
            "command_timeout": runtime.command_timeout,
        }
    }
    for policy in runtime.adversaries:
        settings = asdict(policy)
        del settings["prompts"]  # the policy's prompts, which ``brief`` renders
        rendered[f"adversary.brief.{policy.spec.name}"] = {
            "brief": adversary_brief(spec, (), policy),
            "policy": json.loads(json.dumps(settings, default=str)),
        }
    return rendered


def _adversary(spec: AdversarySpec, defects: tuple[Defect, ...]) -> AdversaryResult:
    name = f"{spec.name}-0"
    submissions = (
        SubmissionResult(1, Outcome.GRADED, 1.0 if spec is FALSE_ACCEPT else 0.0, False, f"adversary/{name}/s1.json"),
    )
    return AdversaryResult(
        name,
        spec.name,
        AdversaryMode.WHITEBOX,
        TrialStatus.GRADED,
        None,
        SearchReport(defects, "searched"),
        submissions,
        f"adversary/{name}/attempt-0.json",
    )


def synthetic_round(name: str, solved: int, fa_defects: tuple[Defect, ...] = ()) -> ValidationResult:
    k = 8
    trials = tuple(
        TrialResult(str(i), TrialStatus.GRADED, 1.0 if i < solved else 0.0, None, f"solver/{i}/attempt-0.json")
        for i in range(k)
    )
    stats = RewardStats(graded=k, mean_reward=solved / k, solved=solved, timed_out=0, candidate_code_errors=0)
    control = ControlResult(
        "negative", ControlKind.NEGATIVE, ControlVerdict.MET, 0.0, None, "controls/negative/attempt-0.json"
    )
    return ValidationResult(
        name,
        0,
        Lowering(LoweringStatus.LOWERED, "", (), None, "task"),
        SolverEvidence(trials, stats, {}),
        (control,),
        (_adversary(FALSE_ACCEPT, fa_defects), _adversary(FALSE_REJECT, ())),
        None,
    )


def _decision(decision: Decision) -> dict:
    return {"type": type(decision).__name__, **json.loads(json.dumps(asdict(decision), default=str))}


async def render_review(runtime: ArmRuntime, root: Path) -> dict:
    band, treatment = runtime.band, runtime.review
    saturated = await treated_review(synthetic_round("saturated", 8), band=band, treatment=treatment, root=root)
    defect = Defect((1,), FA_EXPLANATION, True)
    fa_defect = synthetic_round("fa-defect", 5, (defect,))
    modify = await treated_review(fa_defect, band=band, treatment=treatment, root=root)
    in_band = await treated_review(synthetic_round("in-band", 4), band=band, treatment=treatment, root=root)
    saturated_record = json.loads(
        (root / "workpieces" / "saturated" / "rounds" / "0" / "acceptance-review.json").read_text()
    )
    return {
        "review.saturated_decision": {"decision": _decision(saturated), "band_label": saturated_record["band_label"]},
        "review.modify_feedback": _decision(modify),
        "review.band": {"low": band.low, "high": band.high, "in_band": _decision(in_band)},
        "shadow.l3": {"enabled": runtime.shadow_l3, "shadow": saturated_record["shadow_decisions"]["l3"]},
    }


def render_config(runtime: ArmRuntime, treatment: Treatment) -> dict:
    a = runtime.author
    rendered = {
        "author.model": runtime.models["author"],
        "build.model": runtime.models["build"],
        "controls.model": runtime.models["controls"],
        "adversary.model": runtime.models["adversary"],
        "solver.model": runtime.models["solver"],
        "author.submit_rule": {"enforce_battery": a.enforce_battery, "max_refusals": a.battery_max_refusals},
        "author.battery": {"witness_replay": a.witness_replay},
        "author.unchanged_resubmit": {"text": sha(a.unchanged_text), "max_refusals": a.unchanged_max_refusals},
        "author.repeated_error": {
            "text": sha(a.repeated_error_text),
            "hint_after": a.hint_after,
            "hold_after": a.hold_after,
        },
        "shadow.l4": runtime.shadow_l4,
        "shadow.judge": {
            "model": runtime.models["shadow_judge"],
            "prompt": None if runtime.judge_prompt is None else sha(runtime.judge_prompt),
            "max_claims_per_component": runtime.judge_max_claims,
        },
    }
    return rendered | {f"policy:{key}": value for key, value in treatment.content_values().items()}


async def render(
    treatment: Treatment, proposal: TaskProposal, root: Path, overlays: Sequence[Mapping[str, object]] = ()
) -> dict:
    """Every surface of ``treatment`` with ``overlays`` on ``proposal``; ``root`` is a scratch directory of its own."""
    runtime = arm_runtime(treatment, tuple(overlays))
    surfaces = await render_author(runtime, proposal, root / "author")
    surfaces |= render_validation(runtime)
    surfaces |= await render_review(runtime, root / "review")
    return surfaces | render_config(runtime, treatment)


def diff_arms(rendered: Mapping[str, Sequence[Mapping[str, object]]], a: str, b: str) -> set[str]:
    """The surfaces whose rendering differs between arms ``a`` and ``b`` on any proposal."""
    changed = set()
    for left, right in zip(rendered[a], rendered[b], strict=True):
        assert left.keys() == right.keys(), f"arms {a} and {b} render different surface sets"
        changed |= {k for k in left if json.dumps(left[k], sort_keys=True) != json.dumps(right[k], sort_keys=True)}
    return changed


def check_arms(treatments: Mapping[str, Treatment], rendered: Mapping[str, Sequence[Mapping]]) -> dict:
    """The per-pair diff summary.

    Raises:
        AssertionError: a pair differs where it should not, or fails to differ where it must.
    """
    summary = {}
    names = sorted(treatments)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            changed = diff_arms(rendered, a, b)
            intended = intended_surfaces(treatments[a], treatments[b])
            unintended = sorted(changed - intended)
            missing = sorted(REQUIRED.get((a, b), set()) - changed)
            summary[f"{a}-{b}"] = {
                "changed_keys": list(changed_keys(treatments[a], treatments[b])),
                "changed_surfaces": sorted(changed),
                "unintended": unintended,
                "required_but_unchanged": missing,
            }
            assert not unintended, f"arms {a} and {b} differ outside their keys' surfaces: {unintended}"
            assert not missing, f"arms {a} and {b} should differ in {missing}"
    return summary


def system_addition(rendered: Mapping[str, Sequence[Mapping]], a: str, b: str) -> str:
    """What arm ``b``'s author system turn adds to arm ``a``'s on the first proposal (the L1 text)."""
    left, right = rendered[a][0]["author.system"], rendered[b][0]["author.system"]
    assert right.startswith(left), "arm guidance is appended, so the control's system turn is a prefix"
    return right[len(left) :]


async def dry_run(
    treatments: Mapping[str, Treatment],
    proposals: Sequence[tuple[str, TaskProposal]],
    out: Path,
    overlays: Sequence[Mapping[str, object]] = (),
) -> dict:
    """Render every arm with the driver's ``--policy`` ``overlays`` on every proposal under ``out``, write the renders
    and ``diff.json``, return the summary."""
    rendered: dict[str, list[dict]] = {}
    for name, treatment in sorted(treatments.items()):
        rendered[name] = []
        for capability, proposal in proposals:
            surfaces = await render(treatment, proposal, out / "scratch" / name / capability, overlays)
            rendered[name].append(surfaces)
            target = out / name / capability
            target.mkdir(parents=True, exist_ok=True)
            (target / "system.md").write_text(str(surfaces["author.system"]))
            (target / "surfaces.json").write_text(json.dumps(surfaces, indent=1, sort_keys=True, default=str))
    summary = {
        "proposals": [capability for capability, _ in proposals],
        "pairs": check_arms(treatments, rendered),
        "b_system_addition_chars": len(system_addition(rendered, "A", "B")) if {"A", "B"} <= rendered.keys() else None,
        "tools": {name: [t["function"]["name"] for t in r[0]["author.tools"]] for name, r in rendered.items()},
        "workspace": {name: sorted(r[0]["author.workspace"]) for name, r in rendered.items()},
        "saturated_decision": {name: r[0]["review.saturated_decision"] for name, r in rendered.items()},
        "fa_defect_findings": {name: len(r[0]["review.modify_feedback"]["findings"]) for name, r in rendered.items()},
    }
    (out / "diff.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary
