# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lean 4 proof evals: miniF2F, PutnamBench, IMO-LeanProofBench and ProofNet-Verified, graded by kernel replay on
Lean v4.34.1.

Every task asks for a proof of one frozen statement and is graded by ``lean_grade.py``, the grader of the
Lean training data: the reply's proof is compiled with the grader image's Lean and Mathlib and Prime
Intellect's checker replays it in the kernel. A proof of that statement with the standard axioms scores 1,
any other reply scores 0, and a grader failure fails the attempt instead of scoring it.

Each benchmark has two protocols. ``greedy``, the default, takes one attempt at temperature 0. ``avg8``
samples eight attempts at temperature 1 and reports their mean score, an estimate of pass@1.

miniF2F is the test split as Google DeepMind maintains it (Apache-2.0): 244 competition problems in
``MiniF2F/Test.lean``. That file imports ``MiniF2F.ProblemImports``, which adds two FormalConjectures
modules that no test statement uses, so each prefix imports Mathlib alone. Two theorems come with their
proof (``MINIF2F_BUILTIN_PROOFS``); the converter puts ``sorry`` back in its place, so all 244 problems stay.

PutnamBench (Apache-2.0 for its Lean 4 part) has one file per Putnam problem, 672 at the commit Prime
Intellect's Lean environments pin. 346 of them ask for an answer: a definition left as ``sorry``, with the
official answer in a comment on the next line. As in PutnamBench's with-solution setting, which its
leaderboard reports, the converter writes that answer in place of ``sorry``, so the task asks for a proof
with the answer given. Five answers use real operations Lean cannot compute, so their definitions are marked
``noncomputable`` (``PUTNAMBENCH_NONCOMPUTABLE_ANSWERS``). Six statements do not compile with the grader's Mathlib;
``PORTS`` rewrites them to say the same thing, so all 672 problems are tasks.

IMO-LeanProofBench is the Lean part of Google DeepMind's IMO-Bench (CC-BY-4.0): 60 IMO-level problems in
algebra, combinatorics, number theory and geometry, from pre-IMO to IMO-hard, in the v2 file of 2026-08-19,
written for Lean and Mathlib 4.27.0. Three use ``List.Chain'``, which Mathlib renamed ``List.IsChain``;
``PORTS`` rewrites them, so all 60 problems are tasks. The file's ``Solution`` and ``Grading guidelines``
columns describe the proof; the converter never reads them.

ProofNet-Verified uses the rows and contracts of the training-side source in ``lean/proofs.py``: 365 of its 367
rows, without the repeated exercise and the row that ships its proof.

Known mistakes in the statements are left as upstream has them. In miniF2F,
``mathd_numbertheory_247`` assumes ``3 * n % 2 = 11``, which no ``n`` satisfies, and ``mathd_algebra_76`` and
``amc12_2000_p6`` ask for less than their problems do. PutnamBench's own comments note that
``putnam_1994_a3`` and ``putnam_1967_b6`` change their problems' hypotheses.
"""

import re
import tarfile
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from finestore.rl.status_taxonomy import DataRestriction
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import source_defect, source_evidence
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import (
    EvaluationItem,
    EvaluationProtocol,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.datasets.environments import LEAN_KERNEL_GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.lean.eval_overlap import MINIF2F_REVISION, PUTNAMBENCH_REVISION
from experiments.post_training.task_curation.datasets.lean.lean_grade import Contract, without_comments
from experiments.post_training.task_curation.datasets.lean.proofs import (
    LEAN_GRADE,
    LEAN_PROOF_CONTROLS,
    PROOFNET_VERIFIED_REVISION,
    REPEATED_EXERCISES,
    SORRY,
    convert_proofnet_verified,
    proof_task,
    statement_contract,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, UrlSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin

CONVERSION_REVISION = "1-" + canonical_sha256({r.path: r.source.content_base64 for r in LEAN_GRADE})[:16]
"""Names the prepared corpora. The hash follows the grade script; bump the leading counter whenever this
module's conversion output changes, such as a prompt, a contract or an answer filled in."""
PROTOCOL_VERSION = "2026.11.21"
SOURCE_TAGS = ("eval", "lean", "theorem-proving", "single-turn", "benchmark", "license:apache-2.0")

MINIF2F_TEST_URL = f"https://raw.githubusercontent.com/google-deepmind/miniF2F/{MINIF2F_REVISION}/MiniF2F/Test.lean"
MINIF2F_TEST_SHA256 = "8e135204947654a6f5234ee3da830ea7b72337f33dc29229fd8be6c69ac1c0b7"
MINIF2F_IMPORT = "import MiniF2F.ProblemImports"
MINIF2F_POPULATION = 244
MINIF2F_BUILTIN_PROOFS = {
    "mathd_numbertheory_66": ":=\n  rfl",
    "mathd_algebra_302": ":= by\n  norm_num [div_pow]",
}
"""The test theorems that ``Test.lean`` states together with their proof, mapped to that proof's exact text."""
MINIF2F_THEOREM = re.compile(
    r"^(?:/--(?P<informal>(?:(?!-/).)*)-/(?:[ \t]*\n|[ \t]*--[^\n]*\n)+)?"
    r"(?P<statement>theorem[ \t]+(?P<name>\S+).*?)(?=^/--|^theorem[ \t]|\Z)",
    re.MULTILINE | re.DOTALL,
)
"""One miniF2F theorem, with the docstring above it, through the start of the next one."""
MINIF2F_COMPETITION = re.compile(r"^(mathd_[a-z]+|amc12[ab]?|aime|imo|induction|algebra|numbertheory)")

PUTNAMBENCH_ARCHIVE_URL = f"https://codeload.github.com/trishullab/PutnamBench/tar.gz/{PUTNAMBENCH_REVISION}"
PUTNAMBENCH_ARCHIVE_SHA256 = "e1fd7eb44b51bc24430bafdbe5635f759350653e6d80c7e5bb416eea521fa9f3"
"""The digest of GitHub's tarball of the commit, checked when the archive is fetched."""
PUTNAMBENCH_POPULATION = 672
PUTNAMBENCH_FILE = re.compile(r"PutnamBench-[0-9a-f]{40}/lean4/src/(?P<name>putnam_\w+)\.lean")
PORTS: dict[str, tuple[tuple[str, str], ...]] = {
    # ``List.Chain' R l`` became ``List.IsChain R l``: every two neighbors of ``l`` are related by ``R``.
    "PB-Basic-012": ((r"\bpoints\.Chain' \(", "points.IsChain ("),),
    "PB-Advanced-018": ((r"\bc\.Chain' \(", "c.IsChain ("),),
    "PB-Advanced-023": ((r"\bP\.Chain' IsAdjacentMove\b", "P.IsChain IsAdjacentMove"),),
    "putnam_2021_a1": ((r"\bl\.Chain' fun\b", "l.IsChain fun"),),
    "putnam_2022_a5": ((r"\bg\.Chain' IsValidMove\b", "g.IsChain IsValidMove"),),
    # ``List.Chain R a l`` became ``List.IsChain R (a :: l)``.
    "putnam_2025_a3": (
        (r"\bplay\.Chain isValidMove \(initialState n\)", "(initialState n :: play).IsChain isValidMove"),
    ),
    # ``MvPolynomial.coeff m p`` became ``p.coeff m`` (``AddMonoidAlgebra.coeff``), the coefficient of ``m``.
    "putnam_1971_a4": ((r"MvPolynomial\.coeff i \((P n \S+)\)", r"(\1).coeff i"),),
    "putnam_1987_a4": ((r"\(coeff i P\)\.im", "(P.coeff i).im"),),
    # The ``if`` on ``(∃ m, ...) = True`` needs classical decidability, which does not change its value.
    "putnam_1966_a4": ((r"(?m)^(theorem putnam_1966_a4\b)", r"open Classical in\n\1"),),
}
"""Upstream statements that the grader's Mathlib rejects, mapped to rewrites that state the same thing. Each rewrite
must match exactly once, so a change upstream shows up as a source defect."""

PUTNAMBENCH_NONCOMPUTABLE_ANSWERS = frozenset(
    {"putnam_1991_b5", "putnam_2004_b5", "putnam_2014_a2", "putnam_2021_a2", "putnam_2022_a4"}
)
"""Answers that use real operations Lean cannot compute; their definitions are marked ``noncomputable``."""
PUTNAMBENCH_ANSWER_SLOT = re.compile(
    r"^(?P<head>(?:noncomputable[ \t]+)?(?:abbrev|def)[ \t]+(?P<name>putnam_\w+_solution)\b[^\n]*?:=[ \t]*)"
    r"sorry[ \t]*\n[ \t]*--(?P<answer>[^\n]*)\n",
    re.MULTILINE,
)
"""An answer definition left as ``sorry`` and the one comment line below it that holds the official answer, the
layout of every answer problem at the pinned commit."""
PUTNAMBENCH_DOCSTRING = re.compile(r"/--(?P<informal>(?:(?!-/).)*)-/\s*\Z", re.DOTALL)

IMO_BENCH_REVISION = "80b2527a0b4e4bfc6a8b28825fadbdcfdd6048a1"
IMO_BENCH_URL = (
    f"https://raw.githubusercontent.com/google-deepmind/superhuman/{IMO_BENCH_REVISION}/imobench/lean_proof_bench_v2.csv"
)
IMO_BENCH_SHA256 = "ffc48aecb1290a90582c1b0634f40ea38b86c15891ce081011c5c55905e1c2f3"
IMO_BENCH_POPULATION = 60
IMO_BENCH_TAGS = ("eval", "lean", "theorem-proving", "single-turn", "benchmark", "license:cc-by-4.0")
IMO_BENCH_EVIDENCE = {"problem_id": "Problem ID", "level": "Level", "category": "Category", "source": "Source"}
"""The columns kept as evidence; ``Solution`` and ``Grading guidelines`` describe the proof and stay out."""
IMO_BENCH_THEOREM = re.compile(r"^theorem[ \t]+(?P<name>\S+)", re.MULTILINE)

PROOFNET_POPULATION = 365
PROOFNET_LEFT_OUT = frozenset({35, *REPEATED_EXERCISES})
"""ProofNet-Verified rows the training-side converter leaves out: row 35 ships its proof and row 351 repeats 168.
The publisher counts only selected rows and allows no rejections, so these are left out before conversion."""


def minif2f_rows(path: StoragePath, _context: ConversionContext | None = None) -> Iterator[dict[str, Any]]:
    """Each theorem of ``Test.lean`` with its docstring and the file's header (its imports and ``open``s)."""
    with path.open("rb") as stream:
        text = stream.read().decode()
    theorems = list(MINIF2F_THEOREM.finditer(text))
    header = text[: theorems[0].start()].strip() if theorems else ""
    for match in theorems:
        yield {
            "name": match["name"],
            "informal": (match["informal"] or "").strip(),
            "header": header,
            "formal_statement": match["statement"].strip(),
        }


def minif2f_contract(data: Mapping[str, Any]) -> Contract | ImportRejection:
    """The frozen contract of one miniF2F theorem, importing Mathlib for the file's import, with a shipped
    proof replaced by ``sorry``."""
    name, header, statement = (data.get(key) for key in ("name", "header", "formal_statement"))
    if not all(isinstance(value, str) and value.strip() for value in (name, header, statement)):
        return source_defect("missing_input", "name, header and formal_statement are required")
    if (proof := MINIF2F_BUILTIN_PROOFS.get(name)) is not None:
        if not statement.endswith(proof):
            return source_defect("unexpected_proof", f"{name} no longer ends in its known proof")
        statement = f"{statement.removesuffix(proof)}:= by\n  sorry"
    return statement_contract(header.replace(MINIF2F_IMPORT, "import Mathlib"), statement, name)


def convert_minif2f(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    contract = minif2f_contract(row.data)
    if isinstance(contract, ImportRejection):
        return contract
    competition = MINIF2F_COMPETITION.match(contract.target)
    evidence = {"name": contract.target, "competition": competition.group(1) if competition else "other"}
    task = proof_task(row, context, contract, row.data.get("informal") or "", evidence)
    return task.model_copy(update={"tags": ("eval", "benchmark:miniF2F", *task.tags)})


def putnam_rows(path: StoragePath, _context: ConversionContext | None = None) -> Iterator[dict[str, Any]]:
    """Each PutnamBench Lean file in the archive, by name: its problem name and text. A name the archive
    holds twice is an error, not a row to replace."""
    rows: dict[str, str] = {}
    with path.open("rb") as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if member.isfile() and (match := PUTNAMBENCH_FILE.fullmatch(member.name)) is not None:
                file = archive.extractfile(member)
                if file is not None:
                    if match["name"] in rows:
                        raise ValueError(f"The archive holds {match['name']} twice")
                    rows[match["name"]] = file.read().decode()
    for name in sorted(rows):
        yield {"name": name, "text": rows[name]}


def with_official_answer(name: str, text: str) -> str:
    """``text`` with each answer left as ``sorry`` replaced by the official answer in the comment below it."""

    def fill(match: re.Match[str]) -> str:
        answer = match["answer"].strip()
        head = match["head"]
        if name in PUTNAMBENCH_NONCOMPUTABLE_ANSWERS and not head.startswith("noncomputable"):
            head = f"noncomputable {head}"
        return f"{head}{answer}\n"

    return PUTNAMBENCH_ANSWER_SLOT.sub(fill, text)


def putnam_parts(name: str, text: str) -> tuple[str, str] | None:
    """The file's text above the theorem it is named after, and that theorem, if a line starts it."""
    target = re.search(rf"^theorem[ \t]+{re.escape(name)}(?![\w'.!?])", text, re.MULTILINE)
    return None if target is None else (text[: target.start()], text[target.start() :])


def ported(name: str, text: str) -> str | None:
    """``text`` with ``name``'s rewrites from ``PORTS`` applied, or None if one no longer applies exactly once."""
    for pattern, replacement in PORTS.get(name, ()):
        text, count = re.subn(pattern, replacement, text)
        if count != 1:
            return None
    return text


def putnam_contract(data: Mapping[str, Any]) -> Contract | ImportRejection:
    """The frozen contract of one PutnamBench problem with its official answer written in. The target is the
    theorem the file is named after, and everything above it is the prefix."""
    name, text = data.get("name"), data.get("text")
    if not (isinstance(name, str) and isinstance(text, str) and text.strip()):
        return source_defect("missing_input", "name and text are required")
    if (filled := ported(name, with_official_answer(name, text))) is None:
        return source_defect("port_does_not_apply", f"A rewrite in PORTS no longer applies to {name}")
    if (parts := putnam_parts(name, filled)) is None:
        return source_defect("ambiguous_target", f"No theorem {name} starts a line")
    prefix, statement = parts
    if SORRY.search(without_comments(prefix)) is not None:
        return source_defect("unfilled_answer", "A definition above the theorem is still sorry")
    return statement_contract(prefix, statement, name)


def convert_putnam(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    contract = putnam_contract(row.data)
    if isinstance(contract, ImportRejection):
        return contract
    text = row.data["text"]
    parts = putnam_parts(contract.target, text)
    docstring = PUTNAMBENCH_DOCSTRING.search(parts[0]) if parts else None
    informal = docstring["informal"].strip() if docstring else ""
    setting = "answer given" if PUTNAMBENCH_ANSWER_SLOT.search(text) else "proof only"
    evidence = {"name": contract.target, "setting": setting}
    task = proof_task(row, context, contract, informal, evidence)
    return task.model_copy(update={"tags": ("eval", "benchmark:PutnamBench", *task.tags)})


def is_kept_proofnet_row(data: Mapping[str, Any], _context: ConversionContext | None) -> bool:
    return data.get("index") not in PROOFNET_LEFT_OUT


def imo_bench_contract(data: Mapping[str, Any]) -> Contract | ImportRejection:
    """The frozen contract of one IMO-LeanProofBench row: its Lean file split at the last theorem."""
    text = data.get("Lean Statement")
    if not isinstance(text, str) or not text.strip():
        return source_defect("missing_input", "Lean Statement is required")
    problem = data.get("Problem ID") or ""
    if (text := ported(problem, text)) is None:
        return source_defect("port_does_not_apply", f"A rewrite in PORTS no longer applies to {problem}")
    if not (theorems := list(IMO_BENCH_THEOREM.finditer(text))):
        return source_defect("ambiguous_target", "No theorem starts a line")
    target = theorems[-1]
    return statement_contract(text[: target.start()], text[target.start() :], target["name"])


def convert_imo_bench(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    contract = imo_bench_contract(row.data)
    if isinstance(contract, ImportRejection):
        return contract
    evidence = {"name": contract.target, **{key: row.data.get(column) for key, column in IMO_BENCH_EVIDENCE.items()}}
    task = proof_task(row, context, contract, row.data.get("Problem") or "", evidence)
    return task.model_copy(update={"tags": ("eval", "benchmark:IMO-LeanProofBench", *task.tags)})


def convert_proofnet(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    task = convert_proofnet_verified(row, context)
    if isinstance(task, ImportRejection):
        return task
    return task.model_copy(update={"tags": ("eval", "benchmark:ProofNet-Verified", *task.tags)})


def grouped_item(group: str) -> Callable[[TaskSpec], EvaluationItem]:
    """A chat request scored on the final answer, grouped by one evidence field."""

    def item(task: TaskSpec) -> EvaluationItem:
        evidence = source_evidence(task)
        return chat_item(task, response_selection="final_only").model_copy(
            update={"groups": {group: str(evidence.get(group, "unknown"))}}
        )

    return item


PROTOCOLS = {
    "greedy": (
        Sampling(
            temperature=0.0,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=(),
            chat_template_kwargs={"enable_thinking": True},
        ),
        TrialPolicy(count=1, seed=None),
    ),
    "avg8": (
        Sampling(
            temperature=1.0,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=(),
            chat_template_kwargs={"enable_thinking": True},
        ),
        TrialPolicy(count=8, seed=0),
    ),
}
"""Sampling and trials of each protocol; the first is the default."""


def lean_source(
    info: SourceInfo, recipe: CurationRecipe, benchmark: str, population: int, group: str, protocol: str
) -> RlDataSource[EvaluationCorpusRecipe]:
    sampling, trials = PROTOCOLS[protocol]
    return RlDataSource(
        info=info,
        config=EvaluationCorpusRecipe(
            recipe=recipe,
            protocol=EvaluationProtocol(
                benchmark=benchmark,
                version=PROTOCOL_VERSION,
                operation=OperationKind.CHAT_GENERATION,
                sampling=sampling,
                trials=trials,
                metrics=(Metrics.ACCURACY,),
                primary_metric="accuracy",
                summary=SummaryKind.MEAN,
                expected_population=population,
            ),
            item_builder=grouped_item(group),
            protocol_name=protocol,
        ),
        pipeline=prepare_evaluation,
    )


def minif2f(protocol: str) -> RlDataSource[EvaluationCorpusRecipe]:
    return lean_source(
        SourceInfo(
            data_restriction=DataRestriction.OPEN,
            build_path="Lean kernel replay",
            origin=SourceOrigin.GOOGLE_DEEPMIND,
            upstream_dataset="miniF2F",
            upstream_name="miniF2F",
            subset="test",
            family="math-proof",
            tags=SOURCE_TAGS,
            count=MINIF2F_POPULATION,
            notes="The 244 miniF2F test problems as Google DeepMind maintains them, without the two shipped proofs.",
        ),
        CurationRecipe(
            name="lean-minif2f-test",
            source=UrlSource(
                MINIF2F_TEST_URL, MINIF2F_TEST_SHA256, "Test.lean", SourceFormat.GENERATED, read=minif2f_rows
            ),
            convert=convert_minif2f,
            version=CONVERSION_REVISION,
            intended_use=IntendedUse.EVAL,
            controls=LEAN_PROOF_CONTROLS,
            grader=LEAN_KERNEL_GRADER_PACKAGES,
        ),
        "miniF2F",
        MINIF2F_POPULATION,
        "competition",
        protocol,
    )


def putnambench(protocol: str) -> RlDataSource[EvaluationCorpusRecipe]:
    return lean_source(
        SourceInfo(
            data_restriction=DataRestriction.OPEN,
            build_path="Lean kernel replay",
            origin=SourceOrigin.TRISHULLAB,
            upstream_dataset="PutnamBench",
            upstream_name="PutnamBench",
            variant="answers-given",
            family="math-proof",
            tags=SOURCE_TAGS,
            count=PUTNAMBENCH_POPULATION,
            notes="All 672 Putnam problems in Lean, with the official answers written in.",
        ),
        CurationRecipe(
            name="lean-putnambench",
            source=UrlSource(
                PUTNAMBENCH_ARCHIVE_URL,
                PUTNAMBENCH_ARCHIVE_SHA256,
                "PutnamBench.tar.gz",
                SourceFormat.GENERATED,
                read=putnam_rows,
            ),
            convert=convert_putnam,
            version=CONVERSION_REVISION,
            intended_use=IntendedUse.EVAL,
            controls=LEAN_PROOF_CONTROLS,
            grader=LEAN_KERNEL_GRADER_PACKAGES,
        ),
        "PutnamBench",
        PUTNAMBENCH_POPULATION,
        "setting",
        protocol,
    )


def proofnet_verified(protocol: str) -> RlDataSource[EvaluationCorpusRecipe]:
    return lean_source(
        SourceInfo(
            data_restriction=DataRestriction.OPEN,
            build_path="Lean kernel replay",
            origin=SourceOrigin.MARCUSM117,
            upstream_dataset="ProofNet-Verified",
            upstream_name="ProofNet-Verified",
            subset="eval",
            family="math-proof",
            tags=SOURCE_TAGS,
            count=PROOFNET_POPULATION,
            notes="Audited ProofNet statements for Lean and Mathlib v4.34; informal proofs stay out of the prompt.",
        ),
        CurationRecipe(
            name="lean-proofnet-verified-eval",
            source=HfSource(
                "marcusm117/ProofNet-Verified",
                PROOFNET_VERIFIED_REVISION,
                ("proofnet-verified.jsonl",),
                SourceFormat.JSONL,
                select=is_kept_proofnet_row,
            ),
            convert=convert_proofnet,
            version=CONVERSION_REVISION,
            intended_use=IntendedUse.EVAL,
            controls=LEAN_PROOF_CONTROLS,
            grader=LEAN_KERNEL_GRADER_PACKAGES,
        ),
        "ProofNet-Verified",
        PROOFNET_POPULATION,
        "textbook",
        protocol,
    )


def imo_lean_proof_bench(protocol: str) -> RlDataSource[EvaluationCorpusRecipe]:
    return lean_source(
        SourceInfo(
            data_restriction=DataRestriction.OPEN,
            build_path="Lean kernel replay",
            origin=SourceOrigin.IMOBENCH,
            upstream_dataset="leanproofbench",
            upstream_name="IMO-LeanProofBench",
            variant="v2",
            family="math-proof",
            tags=IMO_BENCH_TAGS,
            count=IMO_BENCH_POPULATION,
            notes="IMO-level problems in Lean, pre-IMO to IMO-hard; written solutions stay out of the prompt.",
        ),
        CurationRecipe(
            name="lean-imo-proof-bench-v2",
            source=UrlSource(
                IMO_BENCH_URL,
                IMO_BENCH_SHA256,
                "lean_proof_bench_v2.csv",
                SourceFormat.CSV,
            ),
            convert=convert_imo_bench,
            version=CONVERSION_REVISION,
            intended_use=IntendedUse.EVAL,
            controls=LEAN_PROOF_CONTROLS,
            grader=LEAN_KERNEL_GRADER_PACKAGES,
        ),
        "IMO-LeanProofBench",
        IMO_BENCH_POPULATION,
        "level",
        protocol,
    )


def benchmarks() -> list[EvaluationBenchmark]:
    """miniF2F, PutnamBench, IMO-LeanProofBench and ProofNet-Verified, each with ``greedy`` first, the default."""
    makers = (minif2f, putnambench, imo_lean_proof_bench, proofnet_verified)
    return fixed_benchmarks([make(protocol) for make in makers for protocol in PROTOCOLS])
