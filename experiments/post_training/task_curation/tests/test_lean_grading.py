# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Lean proof grader accepts only kernel-replayed proofs of the frozen statement.

See ``local_grader`` for building the ``LEAN_KERNEL_GRADER_IMAGE`` these tests run in.
"""

import json

import pytest
from taskcompendium.grading_result import Outcome
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.models import CheckStatus, Reply
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.lean.evals import SHIPPED_PROOFS
from experiments.post_training.task_curation.datasets.lean.numina import NUMINA_CONTROLS
from experiments.post_training.task_curation.datasets.lean.proofs import LEAN_GRADER_MEMORY_MB, LEAN_PROOF_CONTROLS
from experiments.post_training.task_curation.tests.conversion import converted_task
from experiments.post_training.task_curation.tests.lean_rows import (
    MINIF2F,
    NUMINA,
    NUMINA_ROWS,
    PROOFNET,
    PUTNAMBENCH,
    REAL_PROOFS,
    eval_row,
    prime_inputs,
    proofnet_row,
)
from experiments.post_training.task_curation.tests.local_grader import grade, with_verifier_file

pytestmark = [pytest.mark.docker, pytest.mark.timeout(900)]

ADD_ZERO = "theorem t (n : Nat) : n + 0 = n := by\n  sorry"
TWO_PLUS_TWO = "theorem t : (2 : Nat) + 2 = 4 := by\n  sorry"
ONE_IS_TWO = "theorem t : (1 : Nat) = 2 := by\n  sorry"
SKIPPED_KERNEL_CHECK = """by
  run_tac do
    let decl := Lean.Declaration.thmDecl
      { name := `t.cheat, levelParams := [], type := Lean.mkConst ``False, value := Lean.mkConst ``True.intro }
    Lean.withOptions (fun o => o.setBool `debug.skipKernelTC true) (Lean.addDecl decl)
  exact t.cheat.elim"""
"""Elaboration accepts a declaration the kernel never checked; only kernel replay rejects it."""


def task(statement: str) -> TaskSpec:
    return converted_task(PROOFNET, proofnet_row(name="t", header="import Mathlib", formal_stmt=statement))


def with_contract(task: TaskSpec, **contract: str) -> TaskSpec:
    """``task`` grading a contract the converter would not have produced."""
    return with_verifier_file(task, inline_resource("config.json", json.dumps(contract).encode()))


def reply(block: str) -> Reply:
    return Reply(TextMessage(role="assistant", content=f"Here is the proof.\n```lean4\n{block}\n```"))


@pytest.mark.parametrize(
    ("statement", "block", "reward"),
    [
        (ADD_ZERO, "by\n  simp", 1.0),
        (TWO_PLUS_TWO, "by\n  decide", 1.0),
        (ADD_ZERO, "theorem t (n : Nat) : n + 0 = n := by\n  simp", 1.0),
        (ADD_ZERO, 'by\n  let _msg := "theorem t"\n  simp', 1.0),
        (ADD_ZERO, 'by\n  run_tac do\n    let _ ← IO.Process.spawn { cmd := "sleep", args := #["600"] }\n  simp', 1.0),
        (ADD_ZERO, 'by\n  run_tac do\n    let _ ← IO.FS.readFile "/tests/config.json"\n  simp', 0.0),
        (ADD_ZERO, "by\n  run_tac do\n    (IO.Process.exit 3 : IO Unit)\n  simp", 0.0),
        (ADD_ZERO, "by\n  definitely_not_a_lean_tactic", 0.0),
        (ADD_ZERO, "by\n  sorry", 0.0),
        (TWO_PLUS_TWO, "by\n  native_decide", 0.0),
        (ADD_ZERO, "by\n  exact sorryAx _ false", 0.0),
        (ADD_ZERO, 'by\n  simp\nrun_cmd Lean.Elab.Command.liftIO <| IO.println "1.0"', 0.0),
        (ADD_ZERO, 'by\n  run_tac do IO.println "1.0"\n  definitely_not_a_lean_tactic', 0.0),
    ],
    ids=[
        "simp",
        "decide",
        "restated_theorem",
        "string_names_the_target",
        "background_process",
        "reads_grader_files",
        "exits_while_compiling",
        "invalid",
        "sorry",
        "native_decide",
        "explicit_sorry_axiom",
        "trailing_command",
        "printed_reward",
    ],
)
def test_the_grader_accepts_only_kernel_checked_proofs_with_standard_axioms(
    statement, block, reward, lean_kernel_machines
):
    result = grade(task(statement), reply(block), lean_kernel_machines, LEAN_GRADER_MEMORY_MB)
    assert (result.status, result.reward) == (Outcome.GRADED, reward), result


def test_a_declaration_the_kernel_never_checked_is_rejected_by_replay(lean_kernel_machines):
    result = grade(task(ONE_IS_TWO), reply(SKIPPED_KERNEL_CHECK), lean_kernel_machines, LEAN_GRADER_MEMORY_MB)
    assert (result.status, result.reward) == (Outcome.GRADED, 0.0), result
    details = json.loads(str((result.diagnostics or {}).get("stderr", "")).strip().splitlines()[-1])
    assert details["status"] == "rejected", details


@pytest.mark.parametrize("name", sorted(REAL_PROOFS))
def test_proofnet_rows_accept_an_independently_checked_proof(name, lean_kernel_machines):
    row, block = REAL_PROOFS[name]
    result = grade(converted_task(PROOFNET, row), reply(block), lean_kernel_machines, LEAN_GRADER_MEMORY_MB)
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0), result


@pytest.mark.parametrize(
    "contract",
    [
        {"prefix": "import Mathlib", "statement": "theorem t : definitely_not_a_lean_constant :=", "target": "t"},
        {"prefix": "import Mathlib\n\naxiom cheat : False", "statement": "theorem t : (1 : Nat) = 2 :=", "target": "t"},
        {
            "prefix": "import Mathlib\n\ntheorem t : True := trivial",
            "statement": "theorem u : (1 : Nat) = 2 :=",
            "target": "t",
        },
        {"prefix": "import Mathlib", "statement": "private theorem t : True :=", "target": "t"},
        {"prefix": "import Mathlib", "statement": "theorem t : (sorry : Nat) = 1 :=", "target": "t"},
    ],
    ids=["does_not_compile", "declares_an_axiom", "sorry_on_another_theorem", "private_target", "sorry_in_statement"],
)
def test_a_contract_that_does_not_validate_is_a_grader_failure_not_a_score(contract, lean_kernel_machines):
    broken = with_contract(task(ADD_ZERO), **contract)
    result = grade(broken, reply("by\n  exact cheat.elim"), lean_kernel_machines, LEAN_GRADER_MEMORY_MB)
    assert result.status == Outcome.INFRA_ERROR, result


def test_the_negative_control_scores_zero_after_every_step(lean_kernel_machines):
    report = run_controls(task(ADD_ZERO), controls=LEAN_PROOF_CONTROLS, machines=lean_kernel_machines)
    statuses = {check.check: check.status for check in report.checks}
    assert statuses == {"negative": CheckStatus.PASS}, report


@pytest.mark.parametrize("name", sorted(SHIPPED_PROOFS))
def test_the_proof_a_minif2f_theorem_shipped_with_scores_one_on_its_frozen_statement(name, lean_kernel_machines):
    task = converted_task(MINIF2F, eval_row("minif2f", name))
    proof = SHIPPED_PROOFS[name].removeprefix(":=").strip()
    result = grade(task, reply(proof), lean_kernel_machines, LEAN_GRADER_MEMORY_MB)
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0), result


def test_a_putnam_task_with_helper_definitions_passes_its_negative_control(lean_kernel_machines):
    task = converted_task(PUTNAMBENCH, eval_row("putnam", "putnam_1997_b5"))
    report = run_controls(task, controls=LEAN_PROOF_CONTROLS, machines=lean_kernel_machines)
    statuses = {check.check: check.status for check in report.checks}
    assert statuses == {"negative": CheckStatus.PASS}, report


@pytest.mark.parametrize("row", NUMINA_ROWS, ids=[row["uuid"][:8] for row in NUMINA_ROWS])
def test_a_numina_human_proof_is_a_golden_control_that_scores_one(row, lean_kernel_machines, tmp_path):
    task = converted_task(NUMINA, row, inputs=prime_inputs(tmp_path))
    report = run_controls(task, controls=NUMINA_CONTROLS, machines=lean_kernel_machines)
    statuses = {check.check: check.status for check in report.checks}
    assert statuses == {"golden": CheckStatus.PASS}, report
