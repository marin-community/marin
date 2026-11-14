# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NuminaMath-LEAN rows whose statement and proof people wrote, as Lean training tasks.

NuminaMath-LEAN (Project Numina, Apache-2.0) pairs competition problems with Lean 4 statements, most of
them autoformalized. This source keeps the 9,475 rows with a human statement and a complete human proof
(``author == "human"``, ``ground_truth_type == "complete"``). Each task asks for a proof of the frozen
statement and is graded like ProofNet-Verified (``proofs.py``). The row's own proof, kept in the task's
evidence and never shown to the solver, is the golden control. At the pinned revision 7,628 rows
convert, every one of their statements validates with the grader's toolchain (``check_statements.py``),
and 4,595 of their proofs compile; with that many failing, the pipeline's sampled check falls below
its 95% gate and runs every task's control, so a row whose proof no longer checks is not admitted.

A row is rejected before it reaches the grader when Prime Intellect's Lean environments (Apache-2.0),
which pin the same toolchain, record its statement as not compiling or as misformalized, when its
statement does not compile for another reason (``UNCOMPILABLE``), when its statement or proof uses the
``∑ x in s`` syntax that current Mathlib no longer parses, when its proof uses ``native_decide``,
``sorry`` or ``admit`` or runs code while it elaborates, and when it restates one of the eval problems
(``eval_overlap.py``).
"""

import gzip
import json
import re
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import EVIDENCE_PATH, source_defect, unsupported
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, RawRow, Reply
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.environments import LEAN_KERNEL_GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.lean.lean_grade import TARGET_NAME, Contract, without_comments
from experiments.post_training.task_curation.datasets.lean.proofs import (
    LEADING_OPTIONS,
    LEAN_GRADER_MEMORY_MB,
    proof_task,
    statement_contract,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, UrlSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

NUMINA_REVISION = "51fa67f1f647ae1ecd81eef9f19306aa8a7b3a94"
NUMINA_FILE = "data/train-00000-of-00001.parquet"
PRIME_ENVS_REVISION = "15e4e9d2f7d52d8d725a2668e483cf51a067e5ba"
PRIME_LISTS = {
    "unported": "480bbc9ac658e375a3f2c17a351bd847c3b157c7f2e3c65a60a23294365c6cae",
    "misformalized": "970772080d78a7e9383cce98235417c0a49124fa4b6e56b3039e44fab7903709",
}
"""Prime Intellect's two lists of NuminaMath-LEAN uuids, by SHA-256: statements that do not compile with
its toolchain, and statements that bind a misspelled name as a new variable."""
UNCOMPILABLE = frozenset(
    {
        "02ebf693-3300-56be-bb03-c2698fed047b",
        "16364ea0-2c3d-5ee0-bcf7-2b063ce7e435",
        "1ea042ec-2381-52b6-97b3-f914aa475ece",
        "4a0c0c05-35fa-5b46-a458-f14802c28962",
        "5448eed8-0238-5c1a-9bcc-143755ae842b",
        "6e6c4a43-2598-52a3-87ff-0dd122860c72",
        "7d3b55e0-7416-59e3-b94e-a23d67a519e6",
        "80cc5f79-626f-5bd4-95b6-2f8e468d49c5",
        "823823da-b876-5d55-bb7d-12f23e3b5105",
        "89528741-3b0b-5e53-b664-38ea45ae3e0c",
        "8f9719f2-54a2-5803-a6d4-e8d6cc090c11",
        "9cf2b5ae-0a00-57bb-a770-d6b927b1b8d6",
        "9d4bb29e-2eb1-51bd-9b6b-ea86ddea6abc",
        "c2c91166-9b1c-52d6-b76a-07ad11f6ecb9",
        "d020f221-f004-589a-989f-8de795c88d71",
        "dbece6f1-2292-525c-8294-2220eb303453",
    }
)
"""Rows whose statements Prime Intellect's lists pass but which do not compile with the grader's toolchain,
mostly because they import Mathlib modules that have since moved; ``check_statements.py numina`` finds
them once this set is emptied."""

EVAL_OVERLAP_FILE = Path(__file__).resolve().parent / "eval_overlap.json"
EVAL_OVERLAP = frozenset(json.loads(EVAL_OVERLAP_FILE.read_text())["overlapping"])
"""Rows that state one of the frozen eval problems (miniF2F, ProofNet-Verified, PutnamBench), as
``eval_overlap.py`` finds them at the pinned revisions; rerun it, and bump the recipe's ``version``, when
an eval set changes."""

DECLARATION_START = re.compile(
    rf"^[ \t]*{LEADING_OPTIONS}(?:theorem|lemma)\s+(?P<name>{TARGET_NAME.pattern})(?![\w'.!?])", re.MULTILINE
)
"""Where a theorem's declaration begins, with its ``set_option ... in``, ``open ... in`` and attributes."""
OPEN_PROOF = re.compile(r":=\s*(?:by\s*)?$")
"""A statement that ends where its proof would start, as NuminaMath-LEAN writes them."""
OLD_BIG_OPERATOR = re.compile(r"[∑∏][^,∑∏]*?\s+in\s+")
"""``∑ x in s, f x``, which Mathlib replaced by ``∑ x ∈ s, f x``; the pattern Prime Intellect rewrites."""
UNUSABLE_PROOF = re.compile(
    r"(?<![\w.'])(?:native_decide|sorry|admit|axiom|run_cmd|run_tac|run_elab|run_meta|by_elab|elab|elab_rules|"
    r"macro|macro_rules|simproc|dsimproc|initialize|unsafe|implemented_by|extern)(?![\w'])|#eval|(?<![\w.'])IO\."
)
"""What makes a human proof unusable as the golden control: what the grader rejects, and commands that
run code while the proof elaborates."""


def is_human_proof_row(row: dict[str, Any], _context: ConversionContext | None) -> bool:
    return row.get("author") == "human" and row.get("ground_truth_type") == "complete"


@cache
def prime_list(path: StoragePath) -> frozenset[str]:
    with path.open("rb") as stream:
        return frozenset(json.loads(gzip.decompress(stream.read())))


def excluded_by_prime(uuid: str, context: ConversionContext) -> str | None:
    """Which of Prime Intellect's lists holds ``uuid``, if any."""
    for name in PRIME_LISTS:
        if uuid in prime_list(context.inputs[name] / prime_list_source(name).filename):
            return name
    return None


def numina_contract(formal_statement: str) -> Contract | ImportRejection:
    """The frozen contract of a NuminaMath-LEAN statement: its last theorem, with what precedes it as the prefix."""
    source = without_comments(formal_statement).rstrip()
    declarations = list(DECLARATION_START.finditer(source))
    if not declarations:
        return source_defect("ambiguous_target", "The statement declares no theorem")
    target = declarations[-1]
    statement = source[target.start() :]
    if OPEN_PROOF.search(statement):
        statement = f"{statement} sorry"
    return statement_contract(source[: target.start()], statement, target["name"])


def row_contract(data: Mapping[str, Any], listed: Callable[[str], str | None]) -> Contract | ImportRejection:
    """The frozen contract of one row, or why the row is left out; ``listed`` names the Prime Intellect list
    that holds a uuid, if any."""
    uuid, problem, statement, proof = (
        data.get(key) for key in ("uuid", "problem", "formal_statement", "formal_ground_truth")
    )
    if not all(isinstance(value, str) and value.strip() for value in (uuid, problem, statement, proof)):
        return source_defect("missing_input", "uuid, problem, formal_statement and formal_ground_truth are required")
    if (name := listed(uuid)) is not None:
        return source_defect(f"prime_{name}", f"Prime Intellect's Lean environments list this row as {name}")
    if uuid in UNCOMPILABLE:
        return source_defect("does_not_compile", "The statement does not compile with the grader's toolchain")
    if uuid in EVAL_OVERLAP:
        return unsupported("eval_overlap", "The row states a problem of the frozen Lean eval set")
    if OLD_BIG_OPERATOR.search(without_comments(statement)) or OLD_BIG_OPERATOR.search(without_comments(proof)):
        return source_defect("old_big_operator_syntax", "Current Mathlib no longer parses `∑ x in s`")
    if UNUSABLE_PROOF.search(without_comments(proof)):
        return unsupported("unusable_proof", "The human proof cannot serve as the golden control")
    return numina_contract(statement)


def convert_numina(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    data = row.data
    contract = row_contract(data, lambda uuid: excluded_by_prime(uuid, context))
    if isinstance(contract, ImportRejection):
        return contract
    evidence = {key: data[key] for key in ("uuid", "source", "problem_type", "question_type", "exam") if key in data}
    return proof_task(row, context, contract, data["problem"], {**evidence, "proof": data["formal_ground_truth"]})


def human_proof(task: TaskSpec) -> Reply:
    """The row's own proof file, which restates the theorem and proves it, without its comments (some hold
    code fences)."""
    resource = next(resource for resource in task.resources.verifier if resource.path == EVIDENCE_PATH)
    proof = without_comments(json.loads(resource_bytes(resource))["proof"]).strip()
    return answer_reply(task, f"```lean4\n{proof}\n```")


NUMINA_CONTROLS = Controls(golden=human_proof, memory_mb=LEAN_GRADER_MEMORY_MB)


def prime_list_source(name: str) -> UrlSource:
    url = (
        f"https://raw.githubusercontent.com/PrimeIntellect-ai/prime-envs/{PRIME_ENVS_REVISION}"
        f"/environments/lean/numina/numina/data/{name}.json.gz"
    )
    return UrlSource(url, PRIME_LISTS[name], f"{name}.json.gz", SourceFormat.JSON)


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                id="Hugging Face:project-numina/NuminaMath-LEAN",
                title="project-numina/NuminaMath-LEAN",
                origin="Hugging Face",
                family="math-proof",
                tags=("lean", "theorem-proving", "single-turn", "license:apache-2.0"),
                count=9475,
                notes="Competition problems with human Lean statements and proofs; the proofs are golden controls.",
            ),
            config=CurationRecipe(
                name="lean-numina-human",
                source=HfSource(
                    "project-numina/NuminaMath-LEAN",
                    NUMINA_REVISION,
                    (NUMINA_FILE,),
                    SourceFormat.PARQUET,
                    select=is_human_proof_row,
                ),
                convert=convert_numina,
                version="1",
                intended_use=IntendedUse.TRAIN,
                controls=NUMINA_CONTROLS,
                inputs={name: prime_list_source(name) for name in PRIME_LISTS},
                grader=LEAN_KERNEL_GRADER_PACKAGES,
            ),
        )
    ]
