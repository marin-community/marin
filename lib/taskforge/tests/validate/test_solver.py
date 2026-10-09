# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend

from taskforge.ledger.records import EntryKind
from taskforge.llm.policy import LLMPolicy
from taskforge.review.decision import Accept, BandOutcome, Reject, RejectKind
from taskforge.review.rules import BandChoice, BandRule, BandRules, ItemHistory, decide
from taskforge.sandbox.factories import SHELLSIM
from taskforge.validate.calibration import CalibrationBand, FindingKind, load_summary, summarize, write_summary
from taskforge.validate.outcome import Graded
from taskforge.validate.run import ValidationPolicy, load_validation
from taskforge.validate.solver import ValidationSite, run_solver
from taskforge.validate.trials import Deadlines, EngineSettings, RetryBackoff

SETTINGS = EngineSettings(
    factories={Backend.SHELLSIM.value: ShellSimMachineFactory()},
    capabilities={Backend.SHELLSIM.value: SHELLSIM},
    max_turns=4,
    command_timeout=10.0,
    tool_turn_timeout=20.0,
    model_turn_timeout=20.0,
    cleanup_timeout=10.0,
)
POLICY = ValidationPolicy(
    k=2,
    band=CalibrationBand(min_solve_rate=0.125, max_solve_rate=0.875),
    sampling=LLMPolicy(max_continuations=0),
    deadlines=Deadlines(total_turn_timeout=30.0, attempt_timeout=60.0),
    max_retries=0,
    token_contract_retries=0,
    retry_backoff=RetryBackoff(initial=0.1, maximum=0.1, factor=1.0, jitter=0.0),
)


def rules(too_easy: BandChoice) -> BandRules:
    return BandRules(too_easy=BandRule(0, too_easy), too_hard=BandRule(0, BandChoice.REJECT))


async def test_settled_trials_load_from_disk_and_a_solved_round_is_decided_by_the_band_rule(
    tmp_path, draft, ledger, solver_models
):
    site = ValidationSite("IDEA--0", 0, tmp_path / "evidence", ledger)

    outcomes = await run_solver(draft, POLICY, site, SETTINGS, solver_models)

    assert [outcome.reward for outcome in outcomes if isinstance(outcome, Graded)] == [1.0, 1.0]
    trials = [entry for entry in ledger.entries if entry.kind is EntryKind.TRIAL]
    assert sorted(entry.step for entry in trials) == ["solver/0/0", "solver/1/0"]

    await run_solver(draft, POLICY, site, SETTINGS, solver_models)

    assert [entry for entry in ledger.entries if entry.kind is EntryKind.TRIAL] == trials
    summary = summarize(load_validation(draft, site.evidence_dir), POLICY)
    assert (summary.solver.solved, summary.solve_rate) == (2, 1.0)
    assert [finding.kind for finding in summary.findings] == [FindingKind.TOO_EASY]
    assert (summary.roles, summary.assessments, summary.controls_met) == ({}, (), ())
    write_summary(tmp_path / "calibration.json", summary)
    assert load_summary(tmp_path / "calibration.json") == summary

    history = ItemHistory(repairs_used=0, max_repairs=1, band_repairs={})
    assert decide(draft, summary, history, rules(BandChoice.ACCEPT)) == Accept(summary, BandOutcome.TOO_EASY)
    rejected = decide(draft, summary, history, rules(BandChoice.REJECT))
    assert isinstance(rejected, Reject) and rejected.kind is RejectKind.TASK
