# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""miniF2F and PutnamBench as Lean evals, graded and published like ProofNet-Verified (``proofs.py``).

With ProofNet-Verified they make up the frozen Lean eval set: one greedy attempt per statement, and a
model's score on each benchmark is the fraction of its statements proved (``proofs.lean_eval``). Every task
carries only the ``sorry`` negative control: PutnamBench publishes no proofs, and the two miniF2F proofs
below are removed.

miniF2F is the test split as Google DeepMind maintains it (Apache-2.0): 244 competition problems in one
file, ``MiniF2F/Test.lean``. The file imports ``MiniF2F.ProblemImports``, which adds two FormalConjectures
modules to Mathlib that no test statement uses, so each prefix imports Mathlib alone. Two theorems come
with their proof (``SHIPPED_PROOFS``); the converter puts ``sorry`` back in its place, so all 244 problems
stay. 210 theorems carry their informal statement as a docstring; the others show only the Lean statement.

PutnamBench (Apache-2.0 for its Lean 4 part) has one file per Putnam problem, here at the commit Prime
Intellect's Lean environments pin; the later commits change no Lean file. The source leaves out the 6
statements that do not compile with the grader's toolchain (``UNCOMPILABLE``), then the 342 remaining
problems that also ask for an answer, a definition left as ``sorry`` with the answer in a comment below
it, since the grader checks proofs, not answers. The other 324 problems, a proof-only subset of the
benchmark, are tasks, each with its docstring as the informal statement.

Known statement problems are kept, so every statement is the one upstream states.
In miniF2F, ``mathd_numbertheory_247`` assumes ``3 * n % 2 = 11``, which no ``n`` satisfies, so it is
provable without solving the problem, and ``mathd_algebra_76`` and ``amc12_2000_p6`` ask for less than
their problems do. PutnamBench's own comments note that ``putnam_1994_a3`` and ``putnam_1967_b6`` change
their problems' hypotheses.

At the pinned revisions every converted contract validates with the grader's toolchain
(``check_statements.py``).
"""

import re
import tarfile
from collections.abc import Iterator, Mapping
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow

from experiments.post_training.task_curation.datasets.environments import LEAN_KERNEL_GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.lean.lean_grade import Contract, without_comments
from experiments.post_training.task_curation.datasets.lean.proofs import (
    EVAL_PROTOCOL_VERSION,
    LEAN_PROOF_CONTROLS,
    SORRY,
    lean_eval,
    proof_task,
    statement_contract,
)
from experiments.post_training.task_curation.evaluation import EvalDataset
from experiments.post_training.task_curation.pipeline import CurationRecipe, UrlSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

MINIF2F_REVISION = "f0a20e14c1eeccd859d51bb4c2b3ee487889c303"
MINIF2F_FILES_URL = f"https://raw.githubusercontent.com/google-deepmind/miniF2F/{MINIF2F_REVISION}/MiniF2F"
MINIF2F_TEST_SHA256 = "8e135204947654a6f5234ee3da830ea7b72337f33dc29229fd8be6c69ac1c0b7"
MINIF2F_IMPORT = "import MiniF2F.ProblemImports"
SHIPPED_PROOFS = {
    "mathd_numbertheory_66": ":=\n  rfl",
    "mathd_algebra_302": ":= by\n  norm_num [div_pow]",
}
"""The test theorems that ``Test.lean`` states together with their proof, mapped to that proof's exact text."""
THEOREM = re.compile(
    r"^(?:/--(?P<informal>(?:(?!-/).)*)-/(?:[ \t]*\n|[ \t]*--[^\n]*\n)+)?"
    r"(?P<statement>theorem[ \t]+(?P<name>\S+).*?)(?=^/--|^theorem[ \t]|\Z)",
    re.MULTILINE | re.DOTALL,
)
"""One theorem of a miniF2F file, with the docstring above it (blank lines and line comments may sit between
them, as in ``Valid.lean``), through the start of the next one."""

PUTNAMBENCH_REVISION = "f423c173cef31a61fc3e69cbe606c8bc31fcb229"
PUTNAMBENCH_ARCHIVE_SHA256 = "e1fd7eb44b51bc24430bafdbe5635f759350653e6d80c7e5bb416eea521fa9f3"
"""GitHub's tarball of the commit; a different digest means GitHub changed the archive, not the benchmark."""
PUTNAM_FILE = re.compile(r"PutnamBench-[0-9a-f]{40}/lean4/src/(?P<name>putnam_\w+)\.lean")
UNCOMPILABLE = frozenset(
    {"putnam_1966_a4", "putnam_1971_a4", "putnam_1987_a4", "putnam_2021_a1", "putnam_2022_a5", "putnam_2025_a3"}
)
"""PutnamBench statements Lean v4.34.1 rejects: Mathlib no longer has ``List.Chain'`` or
``MvPolynomial.coeff`` in the form they use, or no longer finds an instance they need."""
DOCSTRING_BEFORE = re.compile(r"/--(?P<informal>(?:(?!-/).)*)-/\s*\Z", re.DOTALL)


def minif2f_rows(path: StoragePath, _context: ConversionContext | None = None) -> Iterator[dict[str, Any]]:
    """Each theorem of ``Test.lean`` with its docstring and the file's header (its imports and ``open``s)."""
    with path.open("rb") as stream:
        text = stream.read().decode()
    theorems = list(THEOREM.finditer(text))
    header = text[: theorems[0].start()].strip() if theorems else ""
    for match in theorems:
        yield {
            "name": match["name"],
            "informal": (match["informal"] or "").strip(),
            "header": header,
            "formal_statement": match["statement"].strip(),
        }


def minif2f_contract(data: Mapping[str, Any]) -> Contract | ImportRejection:
    """The frozen contract of one miniF2F theorem, with Mathlib for the file's import and a shipped proof
    replaced by ``sorry``."""
    name, header, statement = (data.get(key) for key in ("name", "header", "formal_statement"))
    if not all(isinstance(value, str) and value.strip() for value in (name, header, statement)):
        return source_defect("missing_input", "name, header and formal_statement are required")
    if (proof := SHIPPED_PROOFS.get(name)) is not None:
        if not statement.endswith(proof):
            return source_defect("unexpected_proof", f"{name} no longer ends in its known proof")
        statement = f"{statement.removesuffix(proof)}:= by\n  sorry"
    return statement_contract(header.replace(MINIF2F_IMPORT, "import Mathlib"), statement, name)


def convert_minif2f(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    contract = minif2f_contract(row.data)
    if isinstance(contract, ImportRejection):
        return contract
    informal = row.data.get("informal") or ""
    return proof_task(row, context, contract, informal, {"name": contract.target}, ("eval", "benchmark:miniF2F"))


def putnam_rows(path: StoragePath, _context: ConversionContext | None = None) -> Iterator[dict[str, Any]]:
    """Each PutnamBench Lean file in the archive, by name: its problem name and text. A name the archive
    holds twice is an error, not a row to replace."""
    rows: dict[str, str] = {}
    with path.open("rb") as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if member.isfile() and (match := PUTNAM_FILE.fullmatch(member.name)) is not None:
                file = archive.extractfile(member)
                if file is not None:
                    if match["name"] in rows:
                        raise ValueError(f"The archive holds {match['name']} twice")
                    rows[match["name"]] = file.read().decode()
    for name in sorted(rows):
        yield {"name": name, "text": rows[name]}


def putnam_parts(name: str, text: str) -> tuple[str, str] | None:
    """The file's text above the theorem it is named after, and that theorem, if a line starts it."""
    target = re.search(rf"^theorem[ \t]+{re.escape(name)}(?![\w'.!?])", text, re.MULTILINE)
    return None if target is None else (text[: target.start()], text[target.start() :])


def asks_for_an_answer(prefix: str) -> bool:
    """Whether a definition above the theorem is left as ``sorry``: the answer the problem asks for."""
    return SORRY.search(without_comments(prefix)) is not None


def is_putnam_proof_problem(data: Mapping[str, Any], _context: ConversionContext | None) -> bool:
    """Select the problems that ask only for a proof and compile with the grader's toolchain."""
    name, text = data.get("name"), data.get("text")
    if not (isinstance(name, str) and isinstance(text, str)) or name in UNCOMPILABLE:
        return False
    parts = putnam_parts(name, text)
    return parts is not None and not asks_for_an_answer(parts[0])


def putnam_contract(data: Mapping[str, Any]) -> tuple[Contract | ImportRejection, str]:
    """The frozen contract of one PutnamBench problem and its informal statement; the target is the
    theorem the file is named after, and everything above it is the prefix."""
    name, text = data.get("name"), data.get("text")
    if not (isinstance(name, str) and isinstance(text, str) and text.strip()):
        return source_defect("missing_input", "name and text are required"), ""
    if name in UNCOMPILABLE:
        return source_defect("does_not_compile", "The statement does not compile with the grader's toolchain"), ""
    if (parts := putnam_parts(name, text)) is None:
        return source_defect("ambiguous_target", f"No theorem {name} starts a line"), ""
    prefix, statement = parts
    if asks_for_an_answer(prefix):
        return unsupported("answer_problem", "The problem asks for an answer, which the grader does not check"), ""
    docstring = DOCSTRING_BEFORE.search(prefix)
    return statement_contract(prefix, statement, name), docstring["informal"].strip() if docstring else ""


def convert_putnam(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    contract, informal = putnam_contract(row.data)
    if isinstance(contract, ImportRejection):
        return contract
    return proof_task(row, context, contract, informal, {"name": contract.target}, ("eval", "benchmark:PutnamBench"))


def sources() -> list[RlDataSource[EvalDataset]]:
    return [
        lean_eval(
            SourceInfo(
                id="GitHub:google-deepmind/miniF2F",
                title="google-deepmind/miniF2F (test)",
                origin="GitHub",
                family="math-proof",
                tags=("eval", "lean", "theorem-proving", "single-turn", "benchmark", "license:apache-2.0"),
                count=244,
                notes="The 244 miniF2F test problems as Google DeepMind maintains them, without the two shipped proofs.",
            ),
            CurationRecipe(
                name="lean-minif2f-test",
                source=UrlSource(
                    f"{MINIF2F_FILES_URL}/Test.lean",
                    MINIF2F_TEST_SHA256,
                    "Test.lean",
                    SourceFormat.GENERATED,
                    read=minif2f_rows,
                ),
                convert=convert_minif2f,
                version=EVAL_PROTOCOL_VERSION,
                intended_use=IntendedUse.EVAL,
                controls=LEAN_PROOF_CONTROLS,
                grader=LEAN_KERNEL_GRADER_PACKAGES,
            ),
            "miniF2F",
        ),
        lean_eval(
            SourceInfo(
                id="GitHub:trishullab/PutnamBench",
                title="trishullab/PutnamBench (proof-only)",
                origin="GitHub",
                family="math-proof",
                tags=("eval", "lean", "theorem-proving", "single-turn", "benchmark", "license:apache-2.0"),
                count=324,
                notes="The Putnam problems that compile and ask only for a proof, in Lean 4; proofs are not public.",
            ),
            CurationRecipe(
                name="lean-putnambench-proof",
                source=UrlSource(
                    f"https://codeload.github.com/trishullab/PutnamBench/tar.gz/{PUTNAMBENCH_REVISION}",
                    PUTNAMBENCH_ARCHIVE_SHA256,
                    "PutnamBench.tar.gz",
                    SourceFormat.GENERATED,
                    select=is_putnam_proof_problem,
                    read=putnam_rows,
                ),
                convert=convert_putnam,
                version=EVAL_PROTOCOL_VERSION,
                intended_use=IntendedUse.EVAL,
                controls=LEAN_PROOF_CONTROLS,
                grader=LEAN_KERNEL_GRADER_PACKAGES,
            ),
            "PutnamBench",
        ),
    ]
