# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A validation round end to end on ShellSim: controls first, trials, evidence read back, and an adversary
pass turned into a control that proves the grader fix."""

import asyncio
import json
import shutil
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from rolloutengine.contracts import ModelRequest, ModelTurn
from taskcompendium.environment import EnvironmentKind, StdoutReward
from taskcompendium.submission import PlainText

from taskforge.llm.client import GlmUnavailable
from taskforge.sandbox.factories import SHELLSIM
from taskforge.spec.draft import file, shell_verifier
from taskforge.validate.adversary import ROLE_PREAMBLES, SENTINEL_REPLIES, AdversaryRole, run_adversaries
from taskforge.validate.calibration import FindingKind, RoleStats, summarize
from taskforge.validate.controls import ControlVerdict
from taskforge.validate.outcome import Cause, Ungraded
from taskforge.validate.run import ValidationEvidence, controls_passed, load_validation, replay_controls
from taskforge.validate.solver import run_solver
from taskforge.validate.trials import EngineSettings, task_digest

PLAIN = PlainText(id="plain")
ROLE_IDS = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
LENIENT_CHECK = "if grep -q 60 /workspace/sum.txt; then echo 1; else echo 0; fi\n"
SHORTCUT = "seq 0 100 > /workspace/sum.txt"


def render(messages) -> tuple[int, ...]:
    ids: list[int] = []
    for message in messages:
        ids.append(ROLE_IDS[message["role"]])
        ids.extend(json.dumps({key: message[key] for key in ("content", "tool_calls") if key in message}).encode())
    return tuple(ids)


@dataclass
class TemplateTokenizer:
    """The server's chat template, deterministically; fails the first ``failures`` calls as a drained router."""

    failures: int = 0
    calls: list[int] = field(default_factory=list)

    async def prompt_ids(self, messages, options):
        self._count()
        return (*render(messages), ROLE_IDS["assistant"])

    async def rendered_ids(self, messages, options):
        self._count()
        return render(messages)

    def _count(self) -> None:
        self.calls.append(len(self.calls))
        if len(self.calls) <= self.failures:
            raise GlmUnavailable("router drained", ())


def settings(fakes) -> EngineSettings:
    return EngineSettings(
        factories={EnvironmentKind.SHELLSIM: fakes.flaky_factory(0, RuntimeError)},
        capabilities={EnvironmentKind.SHELLSIM: SHELLSIM},
        max_turns=6,
        command_timeout=10,
        cleanup_timeout=10,
        conventions=(PLAIN,),
    )


def lenient(task):
    """``task`` graded by a check that accepts any file mentioning 60."""
    verifier = shell_verifier(
        ("sh", "/grader/check.sh"), StdoutReward(), timeout=30, files=(file("/grader/check.sh", LENIENT_CHECK),)
    )
    return task.model_copy(update={"verifier": verifier})


async def test_a_round_reads_back_from_its_attempt_files_as_it_ran(tmp_path, file_task, file_controls, rounds, fakes):
    draft = rounds.draft(file_task, file_controls, PLAIN)
    policy = rounds.policy(k=3, adversary_k=2)
    site = rounds.site(tmp_path)

    controls = await replay_controls(draft, policy, site, settings(fakes), TemplateTokenizer())
    assert controls_passed(controls)
    solver_model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])
    adversary_model = fakes.script_model([fakes.text("NO_SHORTCUT_FOUND")])
    solver, adversaries = await asyncio.gather(
        run_solver(draft, policy, site, settings(fakes), solver_model),
        run_adversaries(draft, policy, site, settings(fakes), adversary_model),
    )
    ran = ValidationEvidence(task_digest(draft.task, draft.execution, draft.convention), controls, solver, adversaries)

    loaded = load_validation(draft, site.evidence_dir)

    assert summarize(loaded, policy) == summarize(ran, policy)
    summary = summarize(loaded, policy)
    assert [f.kind for f in summary.findings] == [FindingKind.TOO_EASY]
    assert summary.controls_met == tuple(c.id for c in file_controls)
    assert summary.roles[AdversaryRole.SHORTCUT].sentinel_replies == 2


@pytest.mark.parametrize("removed", ["solver/1", "adversary/shortcut/0"])
async def test_a_round_missing_a_trial_below_the_highest_index_does_not_load(
    tmp_path, file_task, rounds, fakes, removed
):
    draft = rounds.draft(file_task, (), PLAIN)
    policy = rounds.policy(k=3, adversary_k=2)
    site = rounds.site(tmp_path)
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])
    await asyncio.gather(
        run_solver(draft, policy, site, settings(fakes), model),
        run_adversaries(draft, policy, site, settings(fakes), model),
    )
    shutil.rmtree(site.evidence_dir / removed)

    with pytest.raises(ValueError, match=removed):
        load_validation(draft, site.evidence_dir)


async def test_re_entered_controls_replay_only_the_unsettled_ones(tmp_path, math_task, math_controls, rounds, fakes):
    draft = rounds.draft(math_task, math_controls, PLAIN)
    policy = rounds.policy(max_retries=0)
    site = rounds.site(tmp_path)

    first = await replay_controls(draft, policy, site, settings(fakes), TemplateTokenizer(failures=1))
    (unsettled,) = [c for c in first if c.verdict is ControlVerdict.UNGRADED]
    assert isinstance(unsettled.outcome, Ungraded) and unsettled.outcome.cause is Cause.MODEL_UNAVAILABLE

    tokenizer = TemplateTokenizer()
    second = await replay_controls(draft, policy, site, settings(fakes), tokenizer)

    assert [c.verdict for c in second] == [ControlVerdict.MET] * len(math_controls)
    assert len(tokenizer.calls) == 2
    attempts = sorted(p.name for p in (site.evidence_dir / "control" / unsettled.control.id).iterdir())
    assert attempts == ["attempt-0.json", "attempt-1.json"]


async def test_a_shortcut_pass_becomes_a_control_the_lenient_grader_violates_and_a_strict_one_meets(
    tmp_path, file_task, file_controls, rounds, fakes
):
    leaky = rounds.draft(lenient(file_task), file_controls, PLAIN)
    policy = rounds.policy(adversary_k=1)
    site = rounds.site(tmp_path / "round-0")
    shortcut = fakes.script_model([fakes.shell(SHORTCUT), fakes.text("Done.")])
    shortcut_only = replace(policy, roles=(AdversaryRole.SHORTCUT,))
    adversaries = await run_adversaries(leaky, shortcut_only, site, settings(fakes), shortcut)

    summary = summarize(ValidationEvidence("ab" * 32, (), (), adversaries), policy)

    (finding,) = summary.findings
    assert finding.kind is FindingKind.SHORTCUT_PASSED and SHORTCUT in finding.detail
    (new_control,) = finding.new_controls
    revised_controls = (*file_controls, new_control)
    still_leaky = await replay_controls(
        replace(leaky, controls=revised_controls),
        policy,
        rounds.site(tmp_path / "leaky"),
        settings(fakes),
        TemplateTokenizer(),
    )
    fixed = await replay_controls(
        rounds.draft(file_task, revised_controls, PLAIN),
        policy,
        rounds.site(tmp_path / "fixed"),
        settings(fakes),
        TemplateTokenizer(),
    )
    assert [c.verdict for c in still_leaky if c.control == new_control] == [ControlVerdict.VIOLATED]
    assert controls_passed(fixed)


@dataclass
class HonestAdversary:
    """Does the task's work, then gives up with the sentinel reply of the role whose preamble it sees."""

    work: dict[str, Any]

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        role = next(r for r in AdversaryRole if request.messages[0]["content"].startswith(ROLE_PREAMBLES[r]))
        done = any(message["role"] == "assistant" for message in request.messages)
        message = {"role": "assistant", "content": SENTINEL_REPLIES[role]} if done else self.work
        prompt = (*request.prefix_token_ids, 90) if request.prefix_token_ids else (10, 11)
        return ModelTurn(message, prompt, (21,), (-0.5,), "stop" if done else "tool_calls")


async def test_only_an_ambiguity_pass_that_ends_on_its_sentinel_reply_is_no_finding(tmp_path, file_task, rounds, fakes):
    draft = rounds.draft(file_task, (), PLAIN)
    policy = rounds.policy(adversary_k=2)
    honest = HonestAdversary(fakes.shell("echo 60 > /workspace/sum.txt"))

    adversaries = await run_adversaries(draft, policy, rounds.site(tmp_path), settings(fakes), honest)
    summary = summarize(ValidationEvidence("ab" * 32, (), (), adversaries), policy)

    assert [f.kind for f in summary.findings] == [FindingKind.SHORTCUT_PASSED] * 2 + [FindingKind.LEAK_PASSED] * 2
    assert all(len(f.new_controls) == 1 for f in summary.findings)
    assert summary.roles == {
        role: RoleStats(required=2, graded=2, passes=2, sentinel_replies=2) for role in AdversaryRole
    }
