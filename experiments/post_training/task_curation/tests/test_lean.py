# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lean proof sources freeze each statement into the contract the kernel-replay grader checks."""

import io
import json
import os
import tarfile
import time

import pytest
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TextMessage
from taskcompendium.pipeline.models import ImportRejection

from experiments.post_training.task_curation.datasets.lean.check_statements import audit_failures, failed_unexpectedly
from experiments.post_training.task_curation.datasets.lean.eval_overlap import statement_key
from experiments.post_training.task_curation.datasets.lean.evals import minif2f_rows, putnam_contract, putnam_rows
from experiments.post_training.task_curation.datasets.lean.lean_grade import (
    Contract,
    GraderFailure,
    accepted,
    candidate_source,
    copy_untrusted,
    declares_target,
    proof_block,
    verdict_line,
    without_comments,
)
from experiments.post_training.task_curation.datasets.lean.numina import (
    EVAL_OVERLAP,
    UNCOMPILABLE,
    human_proof,
    is_human_proof_row,
)
from experiments.post_training.task_curation.datasets.lean.proofs import proofnet_contract
from experiments.post_training.task_curation.tests.conversion import convert_row, converted_task
from experiments.post_training.task_curation.tests.lean_rows import (
    MINIF2F,
    MINIF2F_EVAL,
    NUMINA,
    PROOFNET,
    PUTNAMBENCH,
    contract,
    eval_row,
    numina_row,
    prime_inputs,
    proofnet_row,
)

ADD_ZERO = Contract(prefix="import Mathlib", statement="theorem t (n : Nat) : n + 0 = n :=", target="t")


def test_a_proofnet_row_freezes_its_statement_and_shows_it_with_a_sorry_proof():
    task = converted_task(PROOFNET, proofnet_row())
    frozen = contract(task)
    assert frozen["target"] == "Artin_exercise_2_2_9"
    assert frozen["statement"].startswith("theorem Artin_exercise_2_2_9")
    assert frozen["statement"].endswith(":=")
    prompt = task.context.events[0]
    assert isinstance(prompt, TextMessage)
    assert f"{frozen['statement']} by\n  sorry" in prompt.content
    assert "Let $H$ be the subgroup" in prompt.content


def test_helper_declarations_join_the_frozen_prefix():
    helper = "noncomputable def Subgroup.relindex {G : Type*} [Group G] (H K : Subgroup G) : Nat :=\n  H.relIndex K"
    frozen = contract(converted_task(PROOFNET, proofnet_row(helper=helper)))
    assert frozen["prefix"].endswith(helper)


@pytest.mark.parametrize(
    "statement",
    [
        "set_option maxRecDepth 2048 in\ntheorem Artin_exercise_2_2_9 : True := by\n  sorry",
        "set_option maxRecDepth 2048 in theorem Artin_exercise_2_2_9 : True := by\n  sorry",
        "open Real in\ntheorem Artin_exercise_2_2_9 : True := by\n  sorry",
        "@[simp] theorem Artin_exercise_2_2_9 : True := by\n  sorry",
        "/-- A docstring that mentions sorry. -/\ntheorem Artin_exercise_2_2_9 : True := by\n  sorry -- the proof",
        "theorem Artin_exercise_2_2_9 : let n := 1; n = 1 := by\n  sorry",
    ],
    ids=["set_option_newline", "set_option_same_line", "open_in", "attribute", "comments", "inner_assignment"],
)
def test_leading_options_attributes_and_comments_keep_the_target(statement):
    frozen = contract(converted_task(PROOFNET, proofnet_row(formal_stmt=statement)))
    assert frozen["target"] == "Artin_exercise_2_2_9"
    assert frozen["statement"].rstrip().endswith(":=")
    assert "sorry" not in frozen["statement"]


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"formal_stmt": "theorem Artin_exercise_2_2_9 : True := by\n  trivial"}, "no_sorry_proof"),
        ({"formal_stmt": "theorem Artin_exercise_2_2_9 : True := by\n  sorry\n  sorry"}, "no_sorry_proof"),
        ({"formal_stmt": "theorem other_name : True := by\n  sorry"}, "ambiguous_target"),
        ({"formal_stmt": "private theorem Artin_exercise_2_2_9 : True := by\n  sorry"}, "ambiguous_target"),
        (
            {"formal_stmt": "axiom cheat : False\ntheorem Artin_exercise_2_2_9 : True := by\n  sorry"},
            "ambiguous_target",
        ),
        (
            {"formal_stmt": "theorem t : True := by trivial\ntheorem Artin_exercise_2_2_9 : True := by\n  sorry"},
            "ambiguous_target",
        ),
        ({"helper": "lemma helper : True := by\n  sorry"}, "unproved_helper"),
        ({"helper": "lemma helper : True := by\n  admit"}, "unproved_helper"),
        ({"helper": "/-- given -/ axiom helper : False"}, "unproved_helper"),
        ({"helper": "namespace Foo"}, "opens_namespace"),
        ({"header": "import Mathlib\nimport SomethingElse"}, "non_mathlib_import"),
        ({"header": "import Mathlib Lean"}, "non_mathlib_import"),
        ({"formal_stmt": ""}, "missing_input"),
    ],
)
def test_a_statement_the_grader_cannot_freeze_is_a_source_defect(fields, reason):
    result = convert_row(PROOFNET, proofnet_row(**fields))
    assert isinstance(result, ImportRejection)
    assert result.reason == reason


def test_the_graded_block_is_the_last_lean_block_then_the_last_block_then_the_reply():
    assert proof_block("Thinking.\n```lean4\nby\n  simp\n```\nDone.") == "by\n  simp"
    assert proof_block("```lean4\nby\n  simp\n```\n```\nno goals\n```") == "by\n  simp"
    assert proof_block("```\nby\n  decide\n```") == "by\n  decide"
    assert proof_block("by\n  simp\n") == "by\n  simp"


def test_a_proof_term_goes_inside_the_declaration_in_parentheses():
    candidate = candidate_source(ADD_ZERO, "by\n  simp")
    assert candidate == "import Mathlib\n\ntheorem t (n : Nat) : n + 0 = n :=\n  (\n    by\n      simp\n  )\n"


def test_a_restated_theorem_replaces_everything_after_the_imports():
    block = "import Mathlib\n\ntheorem t (n : Nat) : n + 0 = n := by\n  simp"
    assert candidate_source(ADD_ZERO, block) == "import Mathlib\n\ntheorem t (n : Nat) : n + 0 = n := by\n  simp\n"


@pytest.mark.parametrize(
    ("block", "restated"),
    [
        ("theorem t (n : Nat) : n + 0 = n := by\n  simp", True),
        ("set_option maxRecDepth 2048 in theorem t (n : Nat) : n + 0 = n := by\n  simp", True),
        ("open Nat in\n@[simp] theorem t (n : Nat) : n + 0 = n := by\n  simp", True),
        ("by\n  /- theorem t : True := trivial -/\n  simp", False),
        ("by\n  simp -- theorem t", False),
        ('by\n  let _msg := "theorem t"\n  simp', False),
        ("by\n  exact other_t", False),
    ],
)
def test_a_reply_restates_the_target_only_with_a_declaration_outside_comments_and_strings(block, restated):
    assert declares_target(block, "t") is restated


@pytest.mark.parametrize(("index", "reason"), [(35, "no_sorry_proof"), (351, "repeated_exercise")])
def test_the_proofnet_eval_leaves_out_the_row_that_ships_its_proof_and_the_repeated_exercise(index, reason):
    assert PROOFNET.source.select is not None
    assert not PROOFNET.source.select(proofnet_row(index=index), None)
    statement = "theorem Artin_exercise_2_2_9 : True :=\n  trivial" if index == 35 else proofnet_row()["formal_stmt"]
    result = proofnet_contract(proofnet_row(index=index, formal_stmt=statement))
    assert isinstance(result, ImportRejection)
    assert result.reason == reason


PROOF_BYTES, REPLY_BYTES = 1 << 16, 1 << 20
SLOW_BLOCKS = {
    "blank_lines": "\n" * PROOF_BYTES,
    "open_in_per_line": "open X in\n" * (PROOF_BYTES // 10),
    "open_in_one_line": "open X in " * (PROOF_BYTES // 10),
    "open_without_in": "open " + "X " * (PROOF_BYTES // 2),
    "set_option_per_line": "set_option a b in\n" * (PROOF_BYTES // 18),
    "attributes_per_line": "@[simp]\n" * (PROOF_BYTES // 8),
    "open_attributes": "@[" * (PROOF_BYTES // 2),
    "spaces_then_import": " " * PROOF_BYTES + "import",
    "indented_lines": " \n" * (PROOF_BYTES // 2),
    "escaped_quotes": '"' + '\\"' * (PROOF_BYTES // 2),
    "quote_backslash": '"\\' * (PROOF_BYTES // 2),
    "open_block_comments": "/-" * (PROOF_BYTES // 2),
    "line_comments": "--\n" * (PROOF_BYTES // 3),
    "theorem_keywords": "theorem " * (PROOF_BYTES // 8),
}
"""Proof blocks of the largest graded size shaped to make a backtracking pattern slow."""


@pytest.mark.parametrize("block", list(SLOW_BLOCKS.values()), ids=list(SLOW_BLOCKS))
def test_a_proof_block_is_read_in_linear_time_before_any_step(block):
    start = time.monotonic()
    without_comments(block)
    declares_target(block, "t")
    candidate_source(ADD_ZERO, block)
    assert time.monotonic() - start < 2


@pytest.mark.parametrize(
    "reply",
    [
        "```lean4\n" * (REPLY_BYTES // 9),
        "```" * (REPLY_BYTES // 3),
        "`" * REPLY_BYTES,
        "\n" * REPLY_BYTES,
        "```" + " " * REPLY_BYTES,
    ],
    ids=["open_fences", "bare_fences", "backticks", "newlines", "fence_then_blanks"],
)
def test_a_reply_is_read_in_linear_time_before_any_step(reply):
    start = time.monotonic()
    proof_block(reply)
    assert time.monotonic() - start < 2


ACCEPTED = {"ok": True, "targets": [{"name": "t", "ok": True, "mode": "prove", "axioms": ["propext"]}]}


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        (ACCEPTED, True),
        ({"ok": False, "targets": [{"name": "t", "ok": False, "mode": "prove", "axioms": []}]}, False),
        ({"ok": True, "targets": [{"name": "t", "ok": True, "mode": "prove", "axioms": ["cheat"]}]}, False),
        ({"ok": True, "targets": [{"name": "t", "ok": True, "mode": "negation", "axioms": []}]}, False),
    ],
    ids=["accepted", "rejected", "other_axiom", "not_proving_mode"],
)
def test_a_well_formed_verdict_scores_only_a_proof_with_the_standard_axioms(verdict, expected):
    assert accepted(verdict, "t") is expected


@pytest.mark.parametrize(
    "verdict",
    [
        [],
        {"ok": True},
        {"ok": True, "targets": [{"name": "t", "ok": True, "axioms": []}]},
        {"ok": True, "targets": [{"name": "t", "ok": "yes", "mode": "prove", "axioms": []}]},
        {"ok": True, "targets": [{"name": "t", "ok": True, "mode": "prove", "axioms": [1]}]},
        {"ok": False, "targets": [{"name": "u", "ok": False, "mode": "prove", "axioms": []}]},
        {"ok": False, "reference_error": True, "error": "bad reference", "targets": []},
    ],
    ids=["not_an_object", "no_targets", "no_mode", "ok_not_boolean", "axiom_not_text", "other_target", "reference"],
)
def test_a_malformed_verdict_is_a_grader_failure(verdict):
    with pytest.raises(GraderFailure):
        accepted(verdict, "t")


def test_root_copies_only_a_regular_file_the_candidate_left_and_never_fails_as_the_grader(tmp_path):
    (tmp_path / "data").write_bytes(b"{}")
    os.mkfifo(tmp_path / "pipe")
    (tmp_path / "folder").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "data")
    assert copy_untrusted(tmp_path / "data", tmp_path / "copy", 16)
    assert (tmp_path / "copy").read_bytes() == b"{}"
    for source in ("missing", "pipe", "folder", "link"):
        assert not copy_untrusted(tmp_path / source, tmp_path / f"{source}.copy", 16)
    assert not copy_untrusted(tmp_path / "data", tmp_path / "no-such-directory" / "copy", 16)
    assert not copy_untrusted(tmp_path / "data", tmp_path / "small", 1)


def test_the_verdict_is_the_last_json_line_even_when_a_name_holds_a_line_separator():
    entry = {"name": "t", "ok": True, "mode": "prove", "axioms": ["propext\u2028"]}
    output = "replaying\n" + json.dumps({"ok": True, "targets": [entry]}, ensure_ascii=False) + "\n"
    assert json.loads(verdict_line(output)) == {"ok": True, "targets": [entry]}
    assert verdict_line("no verdict\n") is None


def test_a_numina_row_freezes_its_theorem_and_keeps_the_human_proof_for_the_golden_control(tmp_path):
    row = numina_row()
    task = converted_task(NUMINA, row, inputs=prime_inputs(tmp_path))
    assert contract(task) == {
        "prefix": "import Mathlib",
        "statement": "theorem algebra_1 (n : Nat) : n + 0 = n :=",
        "target": "algebra_1",
    }
    prompt = task.context.events[0]
    assert isinstance(prompt, TextMessage)
    assert "Show that adding zero" in prompt.content
    assert "Nat.add_zero" not in prompt.content
    golden = human_proof(task).event
    assert isinstance(golden, TextMessage)
    assert proof_block(golden.content) == row["formal_ground_truth"]


def test_the_golden_control_drops_the_human_proofs_comments_which_can_hold_code_fences(tmp_path):
    proof = "import Mathlib\n\n/-! Example:\n```lean\nexample : True := trivial\n```\n-/\n"
    proof += "theorem algebra_1 (n : Nat) : n + 0 = n := by\n  exact Nat.add_zero n"
    task = converted_task(NUMINA, numina_row(formal_ground_truth=proof), inputs=prime_inputs(tmp_path))
    golden = human_proof(task).event
    assert isinstance(golden, TextMessage)
    block = proof_block(golden.content)
    assert block.startswith("import Mathlib") and block.endswith("exact Nat.add_zero n")
    assert declares_target(block, "algebra_1")


@pytest.mark.parametrize(
    ("formal_statement", "prefix", "statement"),
    [
        (
            "import Mathlib\nset_option maxHeartbeats 400000 in\ntheorem algebra_1 (n : Nat) : n + 0 = n := by",
            "import Mathlib",
            "set_option maxHeartbeats 400000 in\ntheorem algebra_1 (n : Nat) : n + 0 = n :=",
        ),
        (
            "import Mathlib\nopen Nat\n\n/-- Adding zero. -/\n@[simp] theorem algebra_1 (n : Nat) : n + 0 = n := by",
            "import Mathlib\nopen Nat",
            "@[simp] theorem algebra_1 (n : Nat) : n + 0 = n :=",
        ),
        (
            "import Mathlib\ndef f (n : Nat) : Nat := n + 0\ntheorem algebra_1 (n : Nat) : f n = n := by",
            "import Mathlib\ndef f (n : Nat) : Nat := n + 0",
            "theorem algebra_1 (n : Nat) : f n = n :=",
        ),
    ],
    ids=["set_option_in", "open_docstring_attribute", "helper_definition"],
)
def test_a_numina_statement_freezes_its_last_theorem_after_its_prefix(formal_statement, prefix, statement, tmp_path):
    row = numina_row(formal_statement=formal_statement)
    frozen = contract(converted_task(NUMINA, row, inputs=prime_inputs(tmp_path)))
    assert (frozen["prefix"], frozen["statement"].strip(), frozen["target"]) == (prefix, statement, "algebra_1")


def test_eval_overlap_matches_statements_that_differ_only_in_binder_names_and_spacing():
    ours = statement_key("theorem algebra_1 (a b : Nat) (h0 : a = b) : a ^ 2 = b^2 := by")
    theirs = statement_key("theorem aime_1 (x y : Nat) (h1 : x = y) : x^2 = y ^ 2 := rfl")
    assert ours is not None and ours == theirs
    assert ours != statement_key("theorem aime_2 (x y : Nat) (h1 : x = y) : x ^ 3 = y ^ 3 := by")


def test_only_rows_with_a_human_statement_and_proof_are_selected():
    assert is_human_proof_row(numina_row(), None)
    assert not is_human_proof_row(numina_row(author="autoformalizer", ground_truth_type=None), None)
    assert not is_human_proof_row(numina_row(ground_truth_type="statement"), None)


OLD_SUM = "theorem algebra_1 (s : Finset Nat) : \u2211 x in s, 0 = 0 := by"


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"uuid": "listed-unported"}, "prime_unported"),
        ({"uuid": "listed-misformalized"}, "prime_misformalized"),
        ({"uuid": min(EVAL_OVERLAP)}, "eval_overlap"),
        ({"uuid": min(UNCOMPILABLE)}, "does_not_compile"),
        ({"formal_statement": f"import Mathlib\n{OLD_SUM}"}, "old_big_operator_syntax"),
        ({"formal_ground_truth": f"import Mathlib\n{OLD_SUM}\n  simp"}, "old_big_operator_syntax"),
        (
            {"formal_ground_truth": "import Mathlib\ntheorem algebra_1 (n : Nat) : n + 0 = n := by\n  native_decide"},
            "unusable_proof",
        ),
        (
            {"formal_ground_truth": "import Mathlib\ntheorem algebra_1 (n : Nat) : n + 0 = n := by\n  sorry"},
            "unusable_proof",
        ),
        (
            {"formal_ground_truth": "import Mathlib\n#eval 1\ntheorem algebra_1 (n : Nat) : n + 0 = n := by\n  simp"},
            "unusable_proof",
        ),
        (
            {
                "formal_statement": (
                    "import Mathlib\nlemma h : True := by sorry\ntheorem algebra_1 (n : Nat) : n + 0 = n := by"
                )
            },
            "unproved_helper",
        ),
        (
            {"formal_statement": "import Mathlib\nimport Aesop\ntheorem algebra_1 (n : Nat) : n + 0 = n := by"},
            "non_mathlib_import",
        ),
    ],
    ids=[
        "prime_unported",
        "prime_misformalized",
        "eval_overlap",
        "does_not_compile",
        "old_sum_in_statement",
        "old_sum_in_proof",
        "native_decide_proof",
        "sorry_proof",
        "proof_runs_code",
        "unproved_helper",
        "non_mathlib_import",
    ],
)
def test_a_numina_row_without_a_usable_statement_or_proof_is_rejected(fields, reason, tmp_path):
    inputs = prime_inputs(tmp_path, unported=["listed-unported"], misformalized=["listed-misformalized"])
    result = convert_row(NUMINA, numina_row(**fields), inputs=inputs)
    assert isinstance(result, ImportRejection)
    assert result.reason == reason


def test_a_minif2f_theorem_imports_mathlib_in_place_of_the_benchmarks_own_imports():
    task = converted_task(MINIF2F, eval_row("minif2f", "mathd_algebra_478"))
    frozen = contract(task)
    assert frozen["prefix"] == "import Mathlib\n\nopen scoped Nat\nopen scoped Real"
    assert frozen["target"] == "mathd_algebra_478"
    prompt = task.context.events[0]
    assert isinstance(prompt, TextMessage)
    assert "The volume of a cone" in prompt.content
    assert {"eval", "benchmark:miniF2F"} <= set(task.tags)


def test_a_minif2f_theorem_without_a_docstring_shows_only_its_lean_statement():
    task = converted_task(MINIF2F, eval_row("minif2f", "numbertheory_4x3m7y3neq2003"))
    prompt = task.context.events[0]
    assert isinstance(prompt, TextMessage)
    instructions, code = prompt.content.split("```lean4\n", 1)
    assert instructions.strip().count("\n") == 0
    assert f"{contract(task)['statement']} by\n  sorry" in code


@pytest.mark.parametrize(
    ("name", "statement"),
    [
        ("mathd_numbertheory_66", "theorem mathd_numbertheory_66 : 194 % 11 = 7 :="),
        ("mathd_algebra_302", "theorem mathd_algebra_302 : (Complex.I / 2) ^ 2 = -(1 / 4) :="),
    ],
)
def test_a_minif2f_theorem_that_ships_its_proof_is_frozen_without_it(name, statement):
    task = converted_task(MINIF2F, eval_row("minif2f", name))
    assert contract(task)["statement"] == statement
    prompt = task.context.events[0]
    assert isinstance(prompt, TextMessage)
    assert "rfl" not in prompt.content and "div_pow" not in prompt.content


def test_a_minif2f_theorem_whose_shipped_proof_changed_is_a_source_defect():
    row = eval_row("minif2f", "mathd_numbertheory_66")
    result = convert_row(MINIF2F, {**row, "formal_statement": row["formal_statement"].replace("rfl", "by decide")})
    assert isinstance(result, ImportRejection)
    assert result.reason == "unexpected_proof"


def test_test_lean_is_read_as_one_row_per_theorem_with_its_docstring_and_the_files_header(tmp_path):
    header = "import MiniF2F.ProblemImports\n\nopen scoped Real"
    path = tmp_path / "Test.lean"
    path.write_text(
        f"{header}\n\n/-- One and one. -/\ntheorem a : 1 + 1 = 2 := by\n  sorry\n\ntheorem b : 2 = 2 := by\n  sorry\n"
        "\n/-- Three. -/\n\n--/\ntheorem c : 3 = 3 := by\n  sorry\n"
    )
    assert list(minif2f_rows(StoragePath(str(path)))) == [
        {
            "name": "a",
            "informal": "One and one.",
            "header": header,
            "formal_statement": "theorem a : 1 + 1 = 2 := by\n  sorry",
        },
        {"name": "b", "informal": "", "header": header, "formal_statement": "theorem b : 2 = 2 := by\n  sorry"},
        {"name": "c", "informal": "Three.", "header": header, "formal_statement": "theorem c : 3 = 3 := by\n  sorry"},
    ]


def test_a_putnam_problem_freezes_the_theorem_named_after_its_file_below_its_helpers():
    task = converted_task(PUTNAMBENCH, eval_row("putnam", "putnam_1997_b5"))
    frozen = contract(task)
    assert frozen["target"] == "putnam_1997_b5"
    assert frozen["prefix"].startswith("import Mathlib")
    assert "def tetration" in frozen["prefix"]
    assert frozen["statement"].startswith("theorem putnam_1997_b5")
    prompt = task.context.events[0]
    assert isinstance(prompt, TextMessage)
    assert "Prove that for $n" in prompt.content


@pytest.mark.parametrize(
    ("name", "reason"), [("putnam_1962_a5", "answer_problem"), ("putnam_1966_a4", "does_not_compile")]
)
def test_a_putnam_problem_that_asks_for_an_answer_or_does_not_compile_is_left_out(name, reason):
    row = eval_row("putnam", name)
    assert PUTNAMBENCH.source.select is not None
    assert not PUTNAMBENCH.source.select(row, None)
    result, _ = putnam_contract(row)
    assert isinstance(result, ImportRejection)
    assert result.reason == reason


def putnam_archive(path, members):
    top = "PutnamBench-" + "0" * 40
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members:
            member = tarfile.TarInfo(f"{top}/{name}")
            member.size = len(data.encode())
            tar.addfile(member, io.BytesIO(data.encode()))
    return StoragePath(str(path))


def test_the_putnam_archive_yields_its_lean_problem_files_in_name_order(tmp_path):
    first, second = (eval_row("putnam", name)["text"] for name in ("putnam_1962_a1", "putnam_1962_a5"))
    archive = putnam_archive(
        tmp_path / "PutnamBench.tar.gz",
        [
            ("lean4/src/putnam_1962_a5.lean", second),
            ("coq/src/putnam_1962_a1.v", "Theorem putnam_1962_a1."),
            ("lean4/lakefile.lean", "import Lake"),
            ("lean4/src/putnam_1962_a1.lean", first),
        ],
    )
    assert list(putnam_rows(archive)) == [
        {"name": "putnam_1962_a1", "text": first},
        {"name": "putnam_1962_a5", "text": second},
    ]


def test_a_putnam_archive_that_holds_a_problem_twice_is_an_error(tmp_path):
    text = eval_row("putnam", "putnam_1962_a1")["text"]
    archive = putnam_archive(
        tmp_path / "PutnamBench.tar.gz",
        [("lean4/src/putnam_1962_a1.lean", text), ("lean4/src/putnam_1962_a1.lean", text.replace("5", "6"))],
    )
    with pytest.raises(ValueError, match="twice"):
        list(putnam_rows(archive))


def test_the_statement_check_fails_on_a_repeated_row_or_an_eval_population_it_does_not_match():
    population = MINIF2F_EVAL.protocol.expected_population
    complete = [{"row": f"t{index}", "status": "validates"} for index in range(population)]
    assert audit_failures("minif2f", complete) == 0
    assert audit_failures("minif2f", complete[1:]) == 1
    assert audit_failures("minif2f", [*complete[2:], complete[0], complete[0]]) == 1
    assert audit_failures("numina", complete[:3]) == 0


@pytest.mark.parametrize(
    ("source", "result", "unexpected"),
    [
        ("proofnet", {"row": 35, "status": "rejected", "reason": "no_sorry_proof"}, False),
        ("proofnet", {"row": 36, "status": "rejected", "reason": "no_sorry_proof"}, True),
        ("numina", {"row": "0:uuid", "status": "rejected", "reason": "eval_overlap"}, False),
        ("numina", {"row": "0:uuid", "status": "fails"}, True),
        ("numina", {"row": "0:uuid", "status": "timeout"}, True),
        ("proofnet", {"row": 1, "status": "validates"}, False),
        ("minif2f", {"row": "a", "status": "rejected", "reason": "unexpected_proof"}, True),
        ("putnam", {"row": "a", "status": "rejected", "reason": "answer_problem"}, False),
        ("putnam", {"row": "a", "status": "rejected", "reason": "ambiguous_target"}, True),
    ],
    ids=[
        "expected_rejection",
        "other_rejection",
        "numina_rejection",
        "fails",
        "timeout",
        "validates",
        "minif2f_rejection",
        "putnam_answer_problem",
        "putnam_other_rejection",
    ],
)
def test_the_statement_check_fails_on_a_reference_that_does_not_validate(source, result, unexpected):
    assert failed_unexpectedly(source, result) is unexpected
