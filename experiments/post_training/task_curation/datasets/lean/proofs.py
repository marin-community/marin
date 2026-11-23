# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lean theorem-proving sources, graded by kernel replay (``lean_grade.py``).

Each task freezes a theorem statement. The solver sees the statement file with the proof left as
``sorry`` and replies with a proof term, or with the theorem restated and proved. The grader compiles
the result and accepts it only if Prime Intellect's checker replays, in the kernel, a proof of the
frozen statement that uses no axiom beyond ``propext``, ``Classical.choice`` and ``Quot.sound``.

The converter rejects, as source defects, the statements the grader could not freeze: no single
``sorry`` proof to replace, any declaration in the statement besides the target, a ``private`` target,
a prefix that opens a ``namespace``, imports outside Mathlib, or helpers that use ``sorry``, ``admit``
or ``axiom``. These are text checks on the comment-free source; the grader's own reference check, run
by Lean, is what establishes the contract on every grade.

ProofNet-Verified is the audited ProofNet: undergraduate textbook exercises whose statements target Lean
and Mathlib v4.34.0. At the pinned revision every contract validates with the grader's toolchain
(``check_statements.py``) except two rows the converter leaves out: row 35 ships its proof in place of
``sorry``, and row 351 states Rudin's exercise 4.8a again, as row 168 does. ProofNet-Verified's
proofs are not public, so its tasks carry only the negative control, a ``sorry`` proof that passes
through every grader step and must score zero. It is evaluation data.
"""

import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.script_grader import grade_script, script_package
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, RawRow, Reply
from verifyit.spec import DEFAULT_OUTPUT

from experiments.post_training.task_curation.datasets.environments import LEAN_KERNEL_GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.lean.lean_grade import (
    DEADLINE_SECONDS,
    LEAN_VERSION,
    TARGET_NAME,
    Contract,
    without_comments,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

HERE = Path(__file__).resolve().parent
LEAN_GRADE = grade_script(HERE / "lean_grade.py")
LEAN_SCORER_TIMEOUT = DEADLINE_SECONDS + 30.0
"""Above the grade script's own deadline for its four steps."""
LEAN_GRADER_MEMORY_MB = 8192
"""``lean-verify`` loads Mathlib next to the replayed proof; it peaked near 6 GB on a small proof."""

LEADING_OPTIONS = r"(?:(?:set_option\s+\S+\s+\S+|open(?:[ \t]+(?!in\b)\S+)+)\s+in\s+)*(?:@\[[^\]]*\]\s*)*"
"""``set_option ... in`` and ``open ... in`` before a declaration, and its attributes; an ``open`` clause is
the words on its line before its ``in``."""
SORRY_PROOF = re.compile(r"(?s)^(?P<statement>.*:=)\s*(?:by\s+)?sorry\s*$")
"""A statement whose proof is a single trailing ``sorry``; ``statement`` runs through its final ``:=``."""
LEADING = re.compile(rf"^\s*{LEADING_OPTIONS}")
"""``set_option ... in`` and ``open ... in`` before the theorem, and its attributes."""
DECLARATION = re.compile(
    r"(?<![\w.'])(?:theorem|lemma|def|abbrev|instance|axiom|example|structure|inductive|class|opaque|"
    r"namespace|notation|macro|syntax|elab|attribute|run_cmd)(?![\w'])"
)
"""Keywords of the declarations and commands a frozen statement may hold only one of: the target."""
TARGET = re.compile(rf"(?P<keyword>theorem|lemma)\s+(?P<name>{TARGET_NAME.pattern})(?![\w'.!?])")
"""The target's declaration, with a name the grader accepts; any other name is not a target."""
PREFIX_DEFECT = re.compile(r"(?<![\w.'])(?:sorry|admit|axiom)(?![\w'])")
NAMESPACE = re.compile(r"(?<![\w.'])namespace(?![\w'])")
IMPORT = re.compile(r"^\s*import\s+(?P<modules>[^\n]+)$", re.MULTILINE)
SORRY = re.compile(r"(?<![\w.'])sorry(?![\w'])")

INSTRUCTIONS = """Prove the theorem below in Lean 4, with Lean v{version} and Mathlib v{version}.

{informal}

```lean4
{prefix}

{statement} by
  sorry
```

Reply with a ```lean4 code block holding the proof that replaces `by sorry`, for example

```lean4
by
  simp
```

or restate the theorem, with everything above it after the imports, and prove it there. The last ```lean4 block in
your reply is graded. The statement and the declarations above it must stay unchanged, and the proof may use Mathlib
but not `sorry`, `native_decide` or new axioms."""

PROOFNET_VERIFIED_REVISION = "08d4c516eb92d2c1442612a1e5afd5d25d86f4e4"
REPEATED_EXERCISES = {351: 168}
"""ProofNet-Verified rows that state an exercise an earlier row states, mapped to that row."""


def statement_contract(prefix: str, formal_statement: str, target: str) -> Contract | ImportRejection:
    """The grader's frozen contract for ``target``, or why the statement cannot be frozen."""
    prefix, formal_statement = without_comments(prefix).strip(), without_comments(formal_statement).strip()
    modules = [module for line in IMPORT.findall(prefix) for module in line.split()]
    if any(module != "Mathlib" and not module.startswith("Mathlib.") for module in modules):
        return source_defect("non_mathlib_import", "The grader image provides Mathlib only")
    if PREFIX_DEFECT.search(prefix):
        return source_defect("unproved_helper", "Helpers must be proved and add no axioms")
    if NAMESPACE.search(prefix):
        return source_defect("opens_namespace", "The prefix must not open a namespace, which renames the target")
    match = SORRY_PROOF.match(formal_statement)
    if match is None or len(SORRY.findall(formal_statement)) != 1:
        return source_defect("no_sorry_proof", "The statement must end in a single sorry proof to replace")
    statement = match["statement"].strip()
    declaration = statement[LEADING.match(statement).end() :]
    found = TARGET.match(declaration)
    if found is None or found["name"] != target or len(DECLARATION.findall(declaration)) != 1:
        names = [name for _, name in TARGET.findall(statement)]
        return source_defect("ambiguous_target", f"Expected only the public theorem {target}, found {names}")
    return Contract(prefix=prefix, statement=statement, target=target)


def proof_task(
    row: RawRow, context: ConversionContext, contract: Contract, informal: str, evidence: dict[str, Any]
) -> TaskSpec:
    """A conversation task asking for a proof of ``contract``, graded by ``lean_grade.py``."""
    prompt = INSTRUCTIONS.format(
        version=LEAN_VERSION, informal=informal.strip(), prefix=contract.prefix, statement=contract.statement
    )
    package = script_package(
        LEAN_GRADE,
        asdict(contract),
        environment=required_grader_environment(context),
        timeout=LEAN_SCORER_TIMEOUT,
        answer_path=DEFAULT_OUTPUT,
    )
    task = conversation_task(row, events=(TextMessage(role="user", content=prompt),), package=package, evidence=evidence)
    return task.model_copy(update={"tags": ("lean", "proof", "kernel-replay", f"lean-v{LEAN_VERSION}")})


def proofnet_contract(data: Mapping[str, Any]) -> Contract | ImportRejection:
    """The frozen contract of one ProofNet-Verified row; ``helper`` declarations join the header."""
    if (first := REPEATED_EXERCISES.get(data.get("index"))) is not None:
        return source_defect("repeated_exercise", f"Row {first} states the same exercise")
    header, helper = data.get("header"), data.get("helper") or ""
    name, statement = data.get("name"), data.get("formal_stmt")
    if not all(isinstance(value, str) and value.strip() for value in (header, name, statement)):
        return source_defect("missing_input", "header, name and formal_stmt are required")
    if not isinstance(helper, str):
        return source_defect("missing_input", "helper must be text")
    prefix = f"{header.strip()}\n\n{helper.strip()}" if helper.strip() else header
    return statement_contract(prefix, statement, name)


def convert_proofnet_verified(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    informal = row.data.get("informal_stmt")
    if not isinstance(informal, str) or not informal.strip():
        return source_defect("missing_input", "informal_stmt is required")
    contract = proofnet_contract(row.data)
    if isinstance(contract, ImportRejection):
        return contract
    evidence = {key: row.data[key] for key in ("index", "name", "textbook", "informal_proof") if key in row.data}
    return proof_task(row, context, contract, informal, evidence)


def sorry_proof(task: TaskSpec) -> Reply:
    """A proof that compiles and exports but uses ``sorry``, which kernel replay must score zero."""
    return answer_reply(task, "```lean4\nby\n  sorry\n```")


LEAN_PROOF_CONTROLS = Controls(negative=sorry_proof, memory_mb=LEAN_GRADER_MEMORY_MB)


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                id="Hugging Face:marcusm117/ProofNet-Verified",
                title="marcusm117/ProofNet-Verified",
                origin="Hugging Face",
                family="math-proof",
                tags=("eval", "lean", "theorem-proving", "single-turn", "benchmark", "license:apache-2.0"),
                count=367,
                notes="Audited ProofNet statements for Lean and Mathlib v4.34; proofs are not public.",
            ),
            config=CurationRecipe(
                name="lean-proofnet-verified",
                source=HfSource(
                    "marcusm117/ProofNet-Verified",
                    PROOFNET_VERIFIED_REVISION,
                    ("proofnet-verified.jsonl",),
                    SourceFormat.JSONL,
                ),
                convert=convert_proofnet_verified,
                version="1",
                intended_use=IntendedUse.EVAL,
                controls=LEAN_PROOF_CONTROLS,
                grader=LEAN_KERNEL_GRADER_PACKAGES,
            ),
        )
    ]
