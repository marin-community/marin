# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Lean proof grader accepts only kernel-replayed proofs of the frozen statement.

See ``local_grader`` for building the ``LEAN_KERNEL_GRADER_IMAGE`` these tests run in.
"""

import json
import subprocess

import pytest
from taskcompendium.grading_result import Outcome
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.models import CheckStatus, Reply
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.datasets.lean.numina import NUMINA_CONTROLS
from experiments.post_training.task_curation.datasets.lean.proofs import LEAN_GRADER_MEMORY_MB, LEAN_PROOF_CONTROLS
from experiments.post_training.task_curation.tests.conversion import converted_task
from experiments.post_training.task_curation.tests.lean_rows import (
    NUMINA,
    NUMINA_ROWS,
    PROOFNET,
    REAL_PROOFS,
    prime_inputs,
    proofnet_row,
)
from experiments.post_training.task_curation.tests.local_grader import (
    LEAN_KERNEL_SAME_USER_IMAGE,
    grade,
    local_grader_machines,
    with_verifier_file,
)

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


REMOVE_REFERENCE = 'by\n  run_tac do\n    IO.FS.removeFile "/tmp/lean-grade/reference/Task.olean"\n  simp'
"""Deletes the compiled reference, which a candidate running as the grader's user can write."""
TOUCH_TOOLCHAIN = 'by\n  run_tac do\n    IO.FS.writeFile "/opt/lean/written-by-the-candidate" ""\n  simp'
"""Writes into the toolchain, which the local backend's unpacked image leaves writable by the grader's user."""
WRITE_PRELOAD = 'by\n  run_tac do\n    IO.FS.writeFile "/etc/ld.so.preload" "/tmp/verdict.so\\n"\n  simp'
"""Asks the loader to preload a library into every later process, the verifier included."""
FORGE_VERDICT_SCRIPT = (
    "import ctypes, json, threading, time\\n"
    "v = json.dumps({'ok': True, 'targets': [{'name': 't', 'ok': True, 'mode': 'prove', 'axioms': []}]}) + chr(10)\\n"
    "def forge():\\n"
    " for _ in range(600):\\n"
    "  for d in ('logs', 'logs-trusted'):\\n"
    "   try: open('/tmp/lean-grade/' + d + '/verify.out', 'a').write(v)\\n"
    "   except OSError: pass\\n"
    "  time.sleep(0.1)\\n"
    "threading.Thread(target=forge).start()\\n"
    "ctypes.CDLL(None).pthread_exit(None)\\n"
)
FORGE_VERDICT = (
    'by\n  run_tac do\n    let _ ← IO.Process.spawn { cmd := "python3", args := #["-c", "'
    + FORGE_VERDICT_SCRIPT
    + '"] }\n  sorry'
)
"""Leaves a process whose main thread exited, so it shows as a zombie, while another thread keeps appending an
accepting verdict to the verifier's log."""
READ_EVIDENCE = 'by\n  run_tac do\n    let _ ← IO.FS.readFile "/tests/reference/source-evidence.json"\n  simp'
"""Reads the task's source evidence, which the grader removes before the candidate runs."""
STAGE_AS_LOCAL_BACKEND = """
import json, os, sys
files = json.load(sys.stdin)
for path, text in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)
for top in ("/tests", "/app"):
    for directory, names, entries in os.walk(top):
        for name in (directory, *(os.path.join(directory, entry) for entry in (*names, *entries))):
            os.chown(name, 65534, 65534)
os.execv("/usr/bin/setpriv", ["setpriv", "--reuid=65534", "--regid=65534", "--clear-groups", "--inh-caps=-all",
                              "--bounding-set=-all", "--no-new-privs", "python3", "/tests/grade.py"])
"""
"""Writes the task's files as their owner, the unprivileged user, and runs the grade script as that user with no
capabilities, as Shellbox's local backend does."""


def grade_without_root(statement: str, block: str) -> tuple[int, str, dict]:
    """Grade ``block`` the way the local backend does: the grade script's exit code, its stdout and its details."""
    local_grader_machines(LEAN_KERNEL_SAME_USER_IMAGE)
    files = {
        f"/tests/{resource.path}": resource_bytes(resource).decode() for resource in task(statement).resources.verifier
    }
    files["/app/answer.txt"] = reply(block).event.content
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network=none",
            "--memory=8g",
            "--user=0",
            LEAN_KERNEL_SAME_USER_IMAGE,
            "python3",
            "-c",
            STAGE_AS_LOCAL_BACKEND,
        ],
        input=json.dumps(files),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    lines = [line for line in result.stderr.splitlines() if line.startswith("{")]
    return (
        result.returncode,
        result.stdout.strip(),
        json.loads(lines[-1]) if lines else {"stderr": result.stderr[-2000:]},
    )


@pytest.mark.parametrize(
    ("block", "reward"),
    [
        ("by\n  simp", "1.0"),
        ("by\n  sorry", "0.0"),
        ("by\n  decide", "0.0"),
        (REMOVE_REFERENCE, "1.0"),
        ('by\n  run_tac do\n    let _ ← IO.Process.spawn { cmd := "sleep", args := #["600"] }\n  simp', "1.0"),
        (READ_EVIDENCE, "0.0"),
        (FORGE_VERDICT, "0.0"),
        (TOUCH_TOOLCHAIN, None),
        (WRITE_PRELOAD, None),
    ],
    ids=[
        "proof",
        "sorry",
        "failing_tactic",
        "reference_removed",
        "process_left_running",
        "evidence_read",
        "forged_verdict",
        "toolchain_written",
        "preload_written",
    ],
)
def test_without_root_the_candidate_runs_as_the_graders_user_and_cannot_change_the_verdict(block, reward):
    code, stdout, details = grade_without_root(ADD_ZERO, block)
    if reward is None:
        assert code != 0 and "files changed while the candidate's steps ran" in details.get("stderr", ""), details
        return
    assert (code, stdout, details.get("isolation")) == (0, reward, "same_user"), details


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


@pytest.mark.parametrize("row", NUMINA_ROWS, ids=[row["uuid"][:8] for row in NUMINA_ROWS])
def test_a_numina_human_proof_is_a_golden_control_that_scores_one(row, lean_kernel_machines, tmp_path):
    task = converted_task(NUMINA, row, inputs=prime_inputs(tmp_path))
    report = run_controls(task, controls=NUMINA_CONTROLS, machines=lean_kernel_machines)
    statuses = {check.check: check.status for check in report.checks}
    assert statuses == {"golden": CheckStatus.PASS}, report
