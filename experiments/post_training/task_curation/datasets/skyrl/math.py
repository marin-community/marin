# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL math and numeric-answer sources, graded in process by verifyit.

Symbolic answers use the verifyit math comparator and need a golden control; integer
and numeric answers use the numeric comparator, which the structural checks already exercise.
"""

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import math_answer_task, numeric_answer_task, source_defect, unsupported
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage
from taskcompendium.pipeline.controls import reference_reply
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import Controls, Converter, ImportRejection, IntendedUse, RawRow
from verifyit.modes.extract import extract_boxed

from experiments.post_training.task_curation.pipeline import HfSource, RlDataPipeline, ShellSim, UrlSource
from experiments.post_training.task_curation.source import GradingSelection, RlDataSource, SourceInfo, SourceReference

AIME_VERIFIER = SourceReference(
    "aime",
    "88660ec860e483213c2a1c5e03d7f1c483e93ab25860648ae40a656fcb876db4",
    (
        "https://github.com/marin-community/MarinSkyRL/tree/"
        "e44c4bfcb62c489286a1264094e6d9c883aaf0d2/skyrl-gym/skyrl_gym/envs/aime"
    ),
    grading=GradingSelection(
        "verifyit", (), "e44c4bfcb62c489286a1264094e6d9c883aaf0d2", "8abc63e3bdb37af1d345fcac123ef7d2122598f3"
    ),
)

MATH_CONTROLS = Controls(golden=reference_reply)
SOLUTION_FIELDS = ("solution", "answer_type", "extracted_answer", "source")
ASDIV_REVISION = "883f90a9a65bf00304ba8f37423910fe743abc47"
NUMINA_PROOF_REQUEST = re.compile(r"\b(?:prove|show)\s+that\b", re.IGNORECASE)
NUMINA_INEQUALITY = re.compile(r"[<>≤≥]|\\(?:leqslant|geqslant|leq|geq|le|ge|lt|gt)\b")
ASDIV_UNITS = re.compile(r"\s*\([^)]*\)\s*$")

REFERENCE_CHECK = (
    "Check that the reference answer solves the complete public problem. Difficulty alone is not a quality defect."
)

AIME24_RUBRIC = """
Preserve LaTeX, domains, quantifiers, and geometric assumptions. Do not demand decimal reformulation.

The source expects one integer from 0 through 999; leading zeros do not change the integer.

Check whether the premises specify a unique answer. Difficulty alone is not a quality defect.
"""

AIME_1983_2024_RUBRIC = f"""
Historical AIME answers are integers; contest year and problem number are kept as reference evidence. Reserve this
benchmark for evaluation.

{REFERENCE_CHECK}
"""

ASDIV_RUBRIC = f"""
The Body and Question jointly specify the problem. Answer parentheses contain units, which must agree with the public
quantity.

{REFERENCE_CHECK}
"""

DAPO_MATH_RUBRIC = f"""
Preserve every prompt message and the reward_model ground truth.

{REFERENCE_CHECK}
"""

DEEPSCALER_RUBRIC = """
Check the problem and reference answer for a unique mathematical result; preserve domains, LaTeX, and units.

The answer field is the key. A solution ending with an option letter does not override a numeric answer; compare the
actual arithmetic before alleging a contradiction.

This training blend includes historical AIME/AMC, Omni-MATH, and Still problems. Retain provenance and do not treat it
as uncontaminated evaluation data.

Difficulty and inability to solve immediately are not defects.
"""

GSM8K_RUBRIC = f"""
The final #### answer is the numeric key; the worked derivation is kept as reference evidence. Check its arithmetic
against the question.

{REFERENCE_CHECK}
"""

HARDMATH_RUBRIC = """
Check the question, reference solution, and ground_truths together. Symbolic regimes, approximations, boundary
conditions, and equations are part of the task.

Compare every requested regime or deliverable with the reference list. A key omitting a requested asymptotic regime is
a concrete mismatch.

Check limiting powers and coefficients before certifying asymptotic formulas; do not confuse necessary and sufficient
regimes or invent precision requirements.

This training source is not an evaluation benchmark. Hard mathematics alone is not a defect.
"""

HENDRYCKS_MATH_RUBRIC = """
Preserve the complete algebra problem, domains, quantifiers, units, and requested answer form.

The last boxed solution answer is the reference. Check short calculations and contradictions; ordered pairs and
half-open intervals are different answers.

Only the algebra train split is used. MATH test and MATH-500 remain evaluation sources and must not be merged into it.

Assess well-posedness independently of difficulty.
"""

MATH500_RUBRIC = f"""
Preserve tuple order, intervals, units, and mathematical domains. This official test subset is evaluation data.

{REFERENCE_CHECK}
"""

NUMINA_MATH_RUBRIC = f"""
The boxed solution conclusion is a generated reference; assess its derivation rather than assuming its correctness.

{REFERENCE_CHECK}
"""

RLVR_MATH_RUBRIC = f"""
Retain public few-shot worked examples and distinguish them from the final question. This selected split is training
data.

{REFERENCE_CHECK}
"""

SVAMP_RUBRIC = """
Identify the quantities and the operation the question actually requests. Check units and directionality.

Ignore irrelevant quantities. Distracting numbers alone do not make a task ambiguous.

Flag contradictions or missing quantities that prevent a unique numeric answer.
"""


def _fields(row: RawRow, keys: Sequence[str]) -> dict[str, Any]:
    return {key: row.data[key] for key in keys if key in row.data}


def _message_math_task(
    row: RawRow, messages: list[Mapping[str, Any]], answer: str, evidence: Mapping[str, Any]
) -> TaskSpec | ImportRejection:
    """Grade a math answer to every source message, not only the final one."""
    events = tuple(TextMessage(role=message["role"], content=message["content"]) for message in messages)
    task = math_answer_task(row, prompt="\n\n".join(event.content for event in events), answer=answer, evidence=evidence)
    if isinstance(task, ImportRejection):
        return task
    return task.model_copy(update={"context": ConversationInput(events=events)})


def convert_aime24(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    answer = row.data.get("answer")
    if not isinstance(answer, str) or not answer.strip().isdigit() or not 0 <= int(answer) <= 999:
        return source_defect("invalid_reference", "AIME answers must be integers from 0 through 999")
    return numeric_answer_task(row, prompt=row.data.get("problem"), answer=answer, tolerance_abs=0.0, tolerance_rel=0.0)


def convert_aime_1983_2024(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    return math_answer_task(
        row,
        prompt=row.data.get("Question"),
        answer=row.data.get("Answer"),
        evidence=_fields(row, ("ID", "Year", "Problem Number", "Part")),
    )


def asdiv_rows(path: StoragePath, _context: ConversionContext) -> Iterator[dict[str, Any]]:
    """Read ASDiv ``Problem`` elements into their attributes and child fields."""
    with path.open("rb") as stream:
        root = ET.parse(stream).getroot()
    for item in root.iter("Problem"):
        yield {**item.attrib, **{child.tag: child.text or "" for child in item}}


def convert_asdiv(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    body, question, answer = (row.data.get(key) for key in ("Body", "Question", "Answer"))
    if not all(isinstance(value, str) and value.strip() for value in (body, question, answer)):
        return source_defect("missing_prompt_or_reference", "Body, Question and Answer are required")
    return math_answer_task(
        row,
        prompt=f"{body}\n\n{question}",
        answer=ASDIV_UNITS.sub("", str(answer)),
        evidence=_fields(row, ("Answer", "Formula", "Solution-Type", "Source", "Grade", "ID")),
    )


def convert_dapo_math(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    messages, reward = row.data.get("prompt"), row.data.get("reward_model")
    if not isinstance(messages, list) or not messages or not isinstance(reward, dict) or "ground_truth" not in reward:
        return source_defect("missing_prompt_or_reference", "prompt messages and reward_model ground_truth are required")
    evidence = _fields(row, ("reward_model", "data_source", "ability", "extra_info"))
    return _message_math_task(row, messages, str(reward["ground_truth"]), evidence)


def convert_deepscaler(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    return math_answer_task(
        row, prompt=row.data.get("problem"), answer=row.data.get("answer"), evidence=_fields(row, SOLUTION_FIELDS)
    )


def convert_gsm8k(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    question, answer = row.data.get("question"), row.data.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str) or "####" not in answer:
        return source_defect("missing_prompt_or_reference", "question and answer with #### final separator are required")
    return math_answer_task(row, prompt=question, answer=answer.rsplit("####", 1)[-1], evidence={"answer": answer})


def convert_hardmath(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    return math_answer_task(
        row,
        prompt=row.data.get("question"),
        answer=row.data.get("ground_truths"),
        evidence=_fields(row, SOLUTION_FIELDS),
    )


def convert_hendrycks_math(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    solution = row.data.get("solution")
    if isinstance(solution, str) and r"\boxed" not in solution:
        return unsupported("missing_final_answer", "A boxed final answer is required in solution")
    return math_answer_task(row, prompt=row.data.get("problem"), answer=solution, evidence=_fields(row, SOLUTION_FIELDS))


def convert_math500(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    return math_answer_task(
        row,
        prompt=row.data.get("problem"),
        answer=row.data.get("answer"),
        evidence={key: row.data.get(key) for key in ("solution", "subject", "level", "unique_id")},
    )


def convert_numina_math(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    problem, solution = row.data.get("problem"), row.data.get("solution")
    if not isinstance(problem, str) or not isinstance(solution, str):
        return source_defect("missing_prompt_or_reference", "problem and solution strings are required")
    expected = extract_boxed(solution)
    if not expected:
        return unsupported("missing_final_answer", "Solution lacks a boxed conclusion")
    if NUMINA_PROOF_REQUEST.search(problem) and NUMINA_INEQUALITY.search(expected):
        # A boxed inequality concludes a requested proof; the answer comparator cannot check the derivation.
        return unsupported(
            "unsupported_proof_contract", "The requested inequality proof is outside the answer comparator"
        )
    return math_answer_task(
        row, prompt=problem, answer=expected, evidence={"solution": solution, "source": row.data.get("source")}
    )


def convert_rlvr_math(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    messages, expected = row.data.get("messages"), row.data.get("ground_truth")
    if not isinstance(messages, list) or not messages or not isinstance(expected, str):
        return source_defect("missing_prompt_or_reference", "messages and ground_truth strings are required")
    return _message_math_task(row, messages, expected, _fields(row, ("dataset", "constraint_type", "constraint")))


def convert_svamp(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    body, question = row.data.get("Body"), row.data.get("Question")
    if not isinstance(body, str) or not body.strip() or not isinstance(question, str) or not question.strip():
        return source_defect("missing_prompt", "Body and Question must be nonempty strings")
    return numeric_answer_task(
        row,
        prompt=f"{body.strip()} {question.strip()}",
        answer=row.data.get("Answer"),
        tolerance_abs=0.0,
        tolerance_rel=0.0,
    )


@dataclass(frozen=True)
class MathSource:
    ("One SkyRL math source; ``controls=None`` for numeric answers the structural checks " "cover.")

    name: str
    source: HfSource | UrlSource
    convert: Converter
    intended_use: IntendedUse
    rubric: str
    controls: Controls | None = MATH_CONTROLS
    info: SourceInfo = field(kw_only=True)


SOURCES = (
    MathSource(
        "aime24",
        HfSource(
            "HuggingFaceH4/aime_2024",
            "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
            ("data/train-*.parquet",),
            SourceFormat.PARQUET,
        ),
        convert_aime24,
        IntendedUse.EVAL,
        AIME24_RUBRIC,
        controls=None,
        info=SourceInfo(
            id="MarinSkyRL:aime24",
            title="HuggingFaceH4/aime_2024",
            origin="MarinSkyRL",
            family="math-answer",
            tags=("rlvr", "single-turn", "benchmark", "gym/aime"),
            verifier=AIME_VERIFIER,
        ),
    ),
    MathSource(
        "aime_1983_2024",
        HfSource(
            "di-zhang-fdu/AIME_1983_2024",
            "3e2cc86390666c5c756622afc0eeb9e6194496bc",
            ("AIME_Dataset_1983_2024.csv",),
            SourceFormat.CSV,
        ),
        convert_aime_1983_2024,
        IntendedUse.EVAL,
        AIME_1983_2024_RUBRIC,
        info=SourceInfo(
            id="MarinSkyRL:aime_1983_2024",
            title="di-zhang-fdu/AIME_1983_2024",
            origin="MarinSkyRL",
            family="math-answer",
            tags=("rlvr", "single-turn", "benchmark", "license:mit", "gym/aime"),
            verifier=AIME_VERIFIER,
        ),
    ),
    MathSource(
        "asdiv",
        UrlSource(
            f"https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/{ASDIV_REVISION}/dataset/ASDiv.xml",
            "ef8904068482919ac48c8eeaaf6df344b8a308ba66d048c2d4d87eab82dc4929",
            "ASDiv.xml",
            SourceFormat.XML,
            read=asdiv_rows,
        ),
        convert_asdiv,
        IntendedUse.TRAIN,
        ASDIV_RUBRIC,
        info=SourceInfo(
            id="MarinSkyRL:asdiv",
            title="chaochun/nlu-asdiv-dataset",
            origin="MarinSkyRL",
            family="math-answer",
            tags=("rlvr", "single-turn", "gym/aime"),
            verifier=AIME_VERIFIER,
        ),
    ),
    MathSource(
        "dapo_math",
        HfSource(
            "BytedTsinghua-SIA/DAPO-Math-17k",
            "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
            ("data/dapo-math-17k.parquet",),
            SourceFormat.PARQUET,
        ),
        convert_dapo_math,
        IntendedUse.TRAIN,
        DAPO_MATH_RUBRIC,
        info=SourceInfo(
            id="MarinSkyRL:dapo_math",
            title="BytedTsinghua-SIA/DAPO-Math-17k",
            origin="MarinSkyRL",
            family="math-answer",
            tags=("rlvr", "single-turn", "license:apache-2.0", "gym/aime"),
            verifier=AIME_VERIFIER,
        ),
    ),
    MathSource(
        "deepscaler",
        HfSource(
            "agentica-org/DeepScaleR-Preview-Dataset",
            "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29",
            ("deepscaler.json",),
            SourceFormat.JSON,
        ),
        convert_deepscaler,
        IntendedUse.TRAIN,
        DEEPSCALER_RUBRIC,
        info=SourceInfo(
            id="MarinSkyRL:deepscaler",
            title="agentica-org/DeepScaleR-Preview-Dataset",
            origin="MarinSkyRL",
            family="math-answer",
            tags=("rlvr", "single-turn", "license:mit", "gym/aime"),
            verifier=AIME_VERIFIER,
        ),
    ),
    MathSource(
        "gsm8k",
        HfSource(
            "openai/gsm8k",
            "740312add88f781978c0658806c59bc2815b9866",
            ("main/train-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        ),
        convert_gsm8k,
        IntendedUse.TRAIN,
        GSM8K_RUBRIC,
        info=SourceInfo(
            id="MarinSkyRL:gsm8k",
            title="openai/gsm8k",
            origin="MarinSkyRL",
            family="math-answer",
            tags=("rlvr", "single-turn", "benchmark", "license:mit", "gym/gsm8k"),
            verifier=SourceReference(
                "gsm8k",
                "60e40a6f794fad7e9527298d901ff35a7dc57f2e3d061661f386fbbea24293bc",
                (
                    "https://github.com/marin-community/MarinSkyRL/tree/"
                    "e44c4bfcb62c489286a1264094e6d9c883aaf0d2/skyrl-gym/skyrl_gym/envs/gsm8k"
                ),
                grading=GradingSelection(
                    "verifyit",
                    (),
                    "e44c4bfcb62c489286a1264094e6d9c883aaf0d2",
                    "8abc63e3bdb37af1d345fcac123ef7d2122598f3",
                ),
            ),
        ),
    ),
    MathSource(
        "hardmath",
        HfSource(
            "pafitis/HARDMath_processed_training",
            "937e9f10356e31e854f6efb9a2507f1e200c8b25",
            ("data/train-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        ),
        convert_hardmath,
        IntendedUse.TRAIN,
        HARDMATH_RUBRIC,
        info=SourceInfo(
            id="MarinSkyRL:hardmath",
            title="pafitis/HARDMath_processed_training",
            origin="MarinSkyRL",
            family="math-answer",
            tags=("rlvr", "single-turn", "gym/aime"),
            verifier=AIME_VERIFIER,
        ),
    ),
    MathSource(
        "hendrycks_math",
        HfSource(
            "EleutherAI/hendrycks_math",
            "21a5633873b6a120296cce3e2df9d5550074f4a3",
            ("algebra/train-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        ),
        convert_hendrycks_math,
        IntendedUse.TRAIN,
        HENDRYCKS_MATH_RUBRIC,
        info=SourceInfo(
            id="MarinSkyRL:hendrycks_math",
            title="EleutherAI/hendrycks_math",
            origin="MarinSkyRL",
            family="math-answer",
            tags=("rlvr", "single-turn", "license:mit", "gym/aime"),
            verifier=AIME_VERIFIER,
        ),
    ),
    MathSource(
        "math500",
        HfSource(
            "HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", ("test.jsonl",), SourceFormat.JSONL
        ),
        convert_math500,
        IntendedUse.EVAL,
        MATH500_RUBRIC,
        info=SourceInfo(
            id="MarinSkyRL:math500",
            title="HuggingFaceH4/MATH-500",
            origin="MarinSkyRL",
            family="math-answer",
            tags=("rlvr", "single-turn", "benchmark", "gym/aime"),
            verifier=AIME_VERIFIER,
        ),
    ),
    MathSource(
        "numina_math",
        HfSource(
            "AI-MO/NuminaMath-CoT",
            "9d8d210c9f6a36c8f3cd84045668c9b7800ef517",
            ("data/train-*.parquet",),
            SourceFormat.PARQUET,
        ),
        convert_numina_math,
        IntendedUse.TRAIN,
        NUMINA_MATH_RUBRIC,
        info=SourceInfo(
            id="MarinSkyRL:numina_math",
            title="AI-MO/NuminaMath-CoT",
            origin="MarinSkyRL",
            family="math-answer",
            tags=("rlvr", "single-turn", "license:apache-2.0", "gym/aime"),
            verifier=AIME_VERIFIER,
        ),
    ),
    MathSource(
        "rlvr_math",
        HfSource(
            "allenai/RLVR-MATH",
            "bd2a93551b503a395fadd1a740d957559cfe6f3c",
            ("data/train-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        ),
        convert_rlvr_math,
        IntendedUse.TRAIN,
        RLVR_MATH_RUBRIC,
        info=SourceInfo(
            id="MarinSkyRL:rlvr_math",
            title="allenai/RLVR-MATH",
            origin="MarinSkyRL",
            family="math-answer",
            tags=("rlvr", "single-turn", "gym/aime"),
            verifier=AIME_VERIFIER,
        ),
    ),
    MathSource(
        "svamp",
        HfSource(
            "ChilleD/SVAMP", "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef", ("data/train-*.parquet",), SourceFormat.PARQUET
        ),
        convert_svamp,
        IntendedUse.TRAIN,
        SVAMP_RUBRIC,
        controls=None,
        info=SourceInfo(
            id="MarinSkyRL:svamp",
            title="ChilleD/SVAMP",
            origin="MarinSkyRL",
            family="math-answer",
            tags=("rlvr", "single-turn", "license:mit", "gym/aime"),
            verifier=AIME_VERIFIER,
        ),
    ),
)


def math_source(source: MathSource) -> RlDataSource:
    return RlDataSource(
        info=source.info,
        pipeline=RlDataPipeline(
            name=source.name,
            source=source.source,
            convert=source.convert,
            version="1",
            environment=ShellSim(),
            intended_use=source.intended_use,
            rubric=source.rubric,
            controls=source.controls,
        ),
    )


def sources() -> list[RlDataSource]:
    return [math_source(source) for source in SOURCES]
