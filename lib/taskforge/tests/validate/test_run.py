# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A validation round end to end on ShellSim: controls first, trials, evidence read back, and an accepted adversary
candidate turned into a control that proves the grader fix. Adversaries run against ``fake_glm``."""

import asyncio
import shutil
from dataclasses import replace

import pytest
from shellbox.machine import Backend

from taskforge.sandbox.factories import LOCAL_DOCKER, SHELLSIM
from taskforge.validate.adversary import AdversaryRole, ClaimKind, run_adversaries
from taskforge.validate.calibration import DefectTier, FindingKind, summarize
from taskforge.validate.controls import ControlVerdict
from taskforge.validate.outcome import Cause, Ungraded
from taskforge.validate.run import controls_passed, load_validation, replay_controls
from taskforge.validate.solver import run_solver
from taskforge.validate.trials import EngineSettings, task_digest
from tests.sandbox.fixture_images import FixtureImageFactory
from tests.validate.conftest import TemplateTokenizer

SUM = "/workspace/sum.txt"
SHELLSIM_BACKEND = Backend.SHELLSIM.value
DOCKER_BACKEND = Backend.DOCKER.value
FLOOD = f"cat /workspace/numbers.txt; seq 0 100 > {SUM}"
SOLVE = f"awk '{{s+=$1}} END{{print s}}' /workspace/numbers.txt > {SUM}"
SHORTCUT = AdversaryRole.SHORTCUT


def settings(fakes) -> EngineSettings:
    return EngineSettings(
        factories={SHELLSIM_BACKEND: fakes.flaky_factory(0, RuntimeError), DOCKER_BACKEND: FixtureImageFactory()},
        capabilities={SHELLSIM_BACKEND: SHELLSIM, DOCKER_BACKEND: LOCAL_DOCKER},
        max_turns=6,
        command_timeout=10,
        tool_turn_timeout=20,
        model_turn_timeout=30,
        cleanup_timeout=10,
    )


@pytest.fixture
def lenient_file_task(file_task_with, verdict_grader):
    """``file_task`` graded by a check that accepts any sum file mentioning 60."""
    return file_task_with(grader_script=verdict_grader(f"'60' in captured({SUM!r})"))


async def test_a_round_reads_back_from_its_attempt_files_as_it_ran(
    tmp_path, file_task, file_controls, file_facts, rounds, fakes, fake_glm, glm_client, turns
):
    draft = rounds.draft(file_task, file_controls)
    policy = rounds.policy(k=3, adversary_k=2)
    site = rounds.site(tmp_path)

    controls = await replay_controls(draft, policy, site, settings(fakes), TemplateTokenizer())
    assert controls_passed(controls)
    solver_model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])
    # Both trials draw from one scripted queue in whatever order they run; each turn pair is self-contained.
    turns(fake_glm, *[("submit", "Done.", ()), "NO_SHORTCUT"] * 2)
    solver, adversaries = await asyncio.gather(
        run_solver(draft, policy, site, settings(fakes), lambda _: solver_model),
        run_adversaries(draft, policy, site, settings(fakes), glm_client, ""),
    )
    digest = task_digest(draft.lowered)
    ran = rounds.evidence(digest, controls, solver, adversaries, file_facts)

    loaded = load_validation(draft, site.evidence_dir)

    assert loaded.facts == file_facts and loaded.adversaries == adversaries
    assert sum(len(t.submissions) for t in loaded.adversaries[SHORTCUT]) == 2

    assert summarize(loaded, policy) == summarize(ran, policy)
    summary = summarize(loaded, policy)
    assert [f.kind for f in summary.findings] == [FindingKind.TOO_EASY]
    assert summary.controls_met == tuple(c.id for c in file_controls)
    assert summary.roles[SHORTCUT].claims[ClaimKind.NO_SHORTCUT] == 2 and summary.notes == ()


@pytest.mark.parametrize("removed", ["solver/1", "adversary/shortcut/0"])
async def test_a_round_missing_a_trial_below_the_highest_index_does_not_load(
    tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns, removed
):
    draft = rounds.draft(file_task, ())
    policy = rounds.policy(k=3, adversary_k=2)
    site = rounds.site(tmp_path)
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])
    turns(fake_glm, "NO_SHORTCUT", "NO_SHORTCUT")
    await asyncio.gather(
        run_solver(draft, policy, site, settings(fakes), lambda _: model),
        run_adversaries(draft, policy, site, settings(fakes), glm_client, ""),
    )
    shutil.rmtree(site.evidence_dir / removed)

    with pytest.raises(ValueError, match=removed):
        load_validation(draft, site.evidence_dir)


async def test_re_entered_controls_replay_only_the_unsettled_ones(tmp_path, math_task, math_controls, rounds, fakes):
    draft = rounds.draft(math_task, math_controls)
    policy = rounds.policy(max_retries=0)
    site = rounds.site(tmp_path)

    first = await replay_controls(draft, policy, site, settings(fakes), TemplateTokenizer(failures=1))
    (unsettled,) = [c for c in first if c.verdict is ControlVerdict.UNGRADED]
    assert isinstance(unsettled.outcome, Ungraded) and unsettled.outcome.cause is Cause.MODEL_UNAVAILABLE

    tokenizer = TemplateTokenizer()
    second = await replay_controls(draft, policy, site, settings(fakes), tokenizer)

    assert [c.verdict for c in second] == [ControlVerdict.MET] * len(math_controls)
    assert tokenizer.calls == 2
    attempts = sorted(p.name for p in (site.evidence_dir / "control" / unsettled.control.id).iterdir())
    assert attempts == ["attempt-0.json", "attempt-1.json"]


async def test_a_shortcut_pass_becomes_a_control_the_lenient_grader_violates_and_a_strict_one_meets(
    tmp_path, file_task, lenient_file_task, file_controls, file_facts, rounds, fakes, fake_glm, glm_client, turns
):
    leaky = rounds.draft(lenient_file_task, file_controls)
    policy = rounds.policy(adversary_k=1)
    site = rounds.site(tmp_path / "round-0")
    turns(fake_glm, ("shell", FLOOD), ("submit", "Done.", (SUM,)), "SHORTCUT: any file mentioning 60 passes")
    adversaries = await run_adversaries(leaky, policy, site, settings(fakes), glm_client, "")

    summary = summarize(rounds.evidence("ab" * 32, (), (), adversaries, file_facts), policy)

    (finding,) = summary.findings
    assert finding.kind is FindingKind.SHORTCUT_PASSED and "any file mentioning 60 passes" in finding.detail
    assert [(a.tier, a.rule) for a in summary.assessments] == [(DefectTier.REPAIR, "6")]
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
        rounds.draft(file_task, revised_controls),
        policy,
        rounds.site(tmp_path / "fixed"),
        settings(fakes),
        TemplateTokenizer(),
    )
    assert [c.verdict for c in still_leaky if c.control == new_control] == [ControlVerdict.VIOLATED]
    assert controls_passed(fixed)


async def test_an_honest_probe_is_no_finding_and_a_late_shortcut_is_a_note(
    tmp_path, file_task, lenient_file_task, file_facts, rounds, fakes, fake_glm, glm_client, turns
):
    policy = rounds.policy(adversary_k=1)
    turns(fake_glm, ("shell", SOLVE), ("submit", "Done.", (SUM,)), "Solved honestly.\nNO_SHORTCUT")
    honest = await run_adversaries(
        rounds.draft(file_task, ()), policy, rounds.site(tmp_path / "honest"), settings(fakes), glm_client, ""
    )
    probes = [("submit", "Done.", ())] * 2
    turns(fake_glm, ("shell", FLOOD), *probes, ("submit", "Done.", (SUM,)), "SHORTCUT: any file mentioning 60 passes")
    late = await run_adversaries(
        rounds.draft(lenient_file_task, ()),
        policy,
        rounds.site(tmp_path / "late"),
        settings(fakes),
        glm_client,
        "",
    )
    adversaries = {SHORTCUT: (*honest[SHORTCUT], *late[SHORTCUT])}

    summary = summarize(rounds.evidence("ab" * 32, (), (), adversaries, file_facts), rounds.policy(adversary_k=2))

    assert summary.findings == ()
    (note,) = summary.notes
    assert note.kind is FindingKind.SHORTCUT_PASSED and note.new_controls == ()
    assert [(a.index, a.tier, a.rule, a.signals.exploit) for a in summary.assessments] == [
        (0, DefectTier.NONE, "10", 1),
        (1, DefectTier.NOTED, "7", 3),
    ]
    stats = summary.roles[SHORTCUT]
    assert (stats.passes, stats.submissions, stats.claims[ClaimKind.SHORTCUT]) == (2, 4, 1)
