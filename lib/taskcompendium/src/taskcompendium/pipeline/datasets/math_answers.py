# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned typed math sources with their source-specific converters and recipes."""

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any, Literal

from pydantic import JsonValue
from rigging.filesystem.storage_path import StoragePath
from verifyit.modes.extract import extract_boxed

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceVisibility,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
    task_resource,
)
from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat, UrlDownload, hub_inputs
from taskcompendium.pipeline.models import (
    CheckSuite,
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    VerificationReport,
)
from taskcompendium.pipeline.verification import verify_witness
from taskcompendium.verifiers.atlas_answers import MathAnswerVerifier


def answer_type(expected: str) -> Literal["scalar", "equation", "interval", "set", "tuple", "list"]:
    """Keep structured mathematical answers distinct from scalar values."""
    if expected.startswith("[") and expected.endswith("]"):
        return "list"
    if expected.startswith(("[", "(")) and expected.endswith(("]", ")")) and "," in expected:
        return "tuple" if expected.startswith("(") and expected.endswith(")") else "interval"
    if expected.startswith(r"\{") or expected.startswith("{"):
        return "set"
    if "=" in expected or r"\approx" in expected:
        return "equation"
    return "scalar"


def normalize_math(row: RawRow, problem_field: str, reference_field: str) -> TaskSpec | ImportRejection:
    problem, reference = row.data.get(problem_field), row.data.get(reference_field)
    if not isinstance(problem, str) or not problem.strip():
        return ImportRejection(reason="missing_prompt", detail=f"{problem_field} must be a nonempty string")
    if not isinstance(reference, str) or not reference.strip():
        return ImportRejection(reason="invalid_reference", detail=f"{reference_field} must be a nonempty string")
    expected = extract_boxed(reference) or reference.strip()
    verifier = MathAnswerVerifier(expected=expected, math_type=answer_type(expected))
    private = json.dumps(
        {key: row.data[key] for key in ("solution", "answer_type", "extracted_answer", "source") if key in row.data},
        ensure_ascii=False,
    ).encode()
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=problem),)),
        environment_requirements=EnvironmentRequirements(),
        resources=(task_resource("/reference/source-evidence.json", private, ResourceVisibility.VERIFIER),),
        answer_type=AnswerType.TEXT,
        verifier=VerifierSpec(kind=VerifierKind.MATH_ANSWER, parameters_json=verifier.model_dump_json()),
    )


def math_controls(task: TaskSpec) -> VerificationReport:
    verifier = MathAnswerVerifier.model_validate_json(task.verifier.parameters_json)
    return VerificationReport(
        checks=verify_witness(task, rf"\boxed{{{verifier.expected}}}", "__incorrect_math_answer__")
    )


def math_task(
    row: RawRow, events: tuple[TextMessage, ...], expected: str, evidence: dict[str, JsonValue]
) -> TaskSpec | ImportRejection:
    """Bind an extracted reference and private evidence to public source messages."""
    problem = "\n\n".join(event.content for event in events)
    task = normalize_math(replace(row, data={"problem": problem, "answer": expected}), "problem", "answer")
    if isinstance(task, ImportRejection):
        return task
    resource = task_resource(
        "/reference/source-evidence.json", json.dumps(evidence, ensure_ascii=False).encode(), ResourceVisibility.VERIFIER
    )
    return task.model_copy(update={"context": ConversationInput(events=events), "resources": (resource,)})


def field_math_task(
    row: RawRow, problem_key: str, answer_key: str, evidence_keys: tuple[str, ...]
) -> TaskSpec | ImportRejection:
    """Extract a direct problem/reference pair while retaining source evidence."""
    problem, expected = row.data.get(problem_key), row.data.get(answer_key)
    if not isinstance(problem, str) or not isinstance(expected, str):
        return ImportRejection(
            reason="missing_prompt_or_reference", detail=f"{problem_key} and {answer_key} strings are required"
        )
    evidence = {key: row.data[key] for key in evidence_keys}
    return math_task(row, (TextMessage(role="user", content=problem),), expected, evidence)


def math_recipe(
    name: str,
    source: HFSource,
    normalize: Callable[[RawRow], TaskSpec | ImportRejection],
    intended_use: IntendedUse,
    rubric: ReviewRubric,
    inputs: RecipeInputs,
) -> DatasetRecipe:
    """Bind a typed math source to the common comparator controls."""
    return DatasetRecipe(
        name=name,
        version=f"{name}-v1",
        source=source,
        normalize=normalize,
        intended_use=intended_use,
        rubric=rubric,
        inputs=inputs,
        check_suite=CheckSuite(
            id=f"{name}-controls",
            revision="1",
            parameters={"comparator": "cleanup-math-verify"},
            run=math_controls,
        ),
    )


MATH500_DATASET = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH500_CONFIG = "default"
MATH500_SPLIT = "test"
MATH500_SOURCE_FILE = "test.jsonl"
MATH500_SOURCE_FORMAT = "jsonl"

MATH500_RUBRIC = ReviewRubric(
    id="math500-quality",
    version="1",
    criteria=(
        "Preserve tuple order, intervals, units, and mathematical domains. This official test subset is "
        "evaluation data.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_math500(row: RawRow) -> TaskSpec | ImportRejection:
    return field_math_task(row, "problem", "answer", ("answer", "solution", "subject", "level", "unique_id"))


def recipe_math500() -> DatasetRecipe:
    return math_recipe(
        "math500",
        HFSource(MATH500_DATASET, MATH500_REVISION, MATH500_CONFIG, MATH500_SPLIT),
        normalize_math500,
        IntendedUse.EVAL,
        MATH500_RUBRIC,
        hub_inputs(
            MATH500_DATASET, MATH500_REVISION, SourceFiles((MATH500_SOURCE_FILE,), SourceFormat(MATH500_SOURCE_FORMAT))
        ),
    )


AIME_1983_2024_DATASET = "di-zhang-fdu/AIME_1983_2024"
AIME_1983_2024_REVISION = "3e2cc86390666c5c756622afc0eeb9e6194496bc"
AIME_1983_2024_CONFIG = "default"
AIME_1983_2024_SPLIT = "train"
AIME_1983_2024_SOURCE_FILE = "AIME_Dataset_1983_2024.csv"
AIME_1983_2024_SOURCE_FORMAT = "csv"

AIME_1983_2024_RUBRIC = ReviewRubric(
    id="aime_1983_2024-quality",
    version="1",
    criteria=(
        "Historical AIME answers are integers; preserve contest year and problem number privately and reserve "
        "this benchmark for evaluation.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_aime_1983_2024(row: RawRow) -> TaskSpec | ImportRejection:
    return field_math_task(row, "Question", "Answer", ("Answer", "ID", "Year", "Problem Number", "Part"))


def recipe_aime_1983_2024() -> DatasetRecipe:
    return math_recipe(
        "aime_1983_2024",
        HFSource(AIME_1983_2024_DATASET, AIME_1983_2024_REVISION, AIME_1983_2024_CONFIG, AIME_1983_2024_SPLIT),
        normalize_aime_1983_2024,
        IntendedUse.EVAL,
        AIME_1983_2024_RUBRIC,
        hub_inputs(
            AIME_1983_2024_DATASET,
            AIME_1983_2024_REVISION,
            SourceFiles((AIME_1983_2024_SOURCE_FILE,), SourceFormat(AIME_1983_2024_SOURCE_FORMAT)),
        ),
    )


GSM8K_DATASET = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "train"
GSM8K_SOURCE_FILE = "main/train-00000-of-00001.parquet"
GSM8K_SOURCE_FORMAT = "parquet"

GSM8K_RUBRIC = ReviewRubric(
    id="gsm8k-quality",
    version="1",
    criteria=(
        "The final #### answer is the numeric key; retain the worked derivation privately and check its "
        "arithmetic against the question.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_gsm8k(row: RawRow) -> TaskSpec | ImportRejection:
    problem, answer = row.data.get("question"), row.data.get("answer")
    if not isinstance(problem, str) or not isinstance(answer, str) or "####" not in answer:
        return ImportRejection(
            reason="missing_prompt_or_reference", detail="question and answer with #### final separator are required"
        )
    expected = answer.rsplit("####", 1)[-1].strip()
    return math_task(row, (TextMessage(role="user", content=problem),), expected, {"answer": answer})


def recipe_gsm8k() -> DatasetRecipe:
    return math_recipe(
        "gsm8k",
        HFSource(GSM8K_DATASET, GSM8K_REVISION, GSM8K_CONFIG, GSM8K_SPLIT),
        normalize_gsm8k,
        IntendedUse.TRAIN,
        GSM8K_RUBRIC,
        hub_inputs(GSM8K_DATASET, GSM8K_REVISION, SourceFiles((GSM8K_SOURCE_FILE,), SourceFormat(GSM8K_SOURCE_FORMAT))),
    )


ASDIV_DATASET = "chaochun/nlu-asdiv-dataset"
ASDIV_REVISION = "883f90a9a65bf00304ba8f37423910fe743abc47"
ASDIV_CONFIG = "original-xml"
ASDIV_SPLIT = "train"
ASDIV_SOURCE_FILE = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/883f90a9a65bf00304ba8f37423910fe743abc47/dataset/ASDiv.xml"
ASDIV_SOURCE_FORMAT = "xml"

ASDIV_RUBRIC = ReviewRubric(
    id="asdiv-quality",
    version="1",
    criteria=(
        "The Body and Question jointly specify the problem. Answer parentheses contain units, which must "
        "agree with the public quantity.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_asdiv(row: RawRow) -> TaskSpec | ImportRejection:
    body, question, answer = (row.data.get(key) for key in ("Body", "Question", "Answer"))
    if not all(isinstance(value, str) and value.strip() for value in (body, question, answer)):
        return ImportRejection(reason="missing_prompt_or_reference", detail="Body, Question and Answer are required")
    expected = re.sub(r"\s*\([^)]*\)\s*$", "", str(answer)).strip()
    evidence = {key: row.data[key] for key in ("Answer", "Formula", "Solution-Type", "Source", "Grade", "ID")}
    return math_task(row, (TextMessage(role="user", content=f"{body}\n\n{question}"),), expected, evidence)


def asdiv_rows(path: StoragePath) -> Iterator[dict[str, Any]]:
    """Read ASDiv Problem elements into the fields used by its converter."""
    with path.open("rb") as stream:
        root = ET.parse(stream).getroot()
    yield from ({**item.attrib, **{child.tag: child.text or "" for child in item}} for item in root.iter("Problem"))


def recipe_asdiv() -> DatasetRecipe:
    return math_recipe(
        "asdiv",
        HFSource(ASDIV_DATASET, ASDIV_REVISION, ASDIV_CONFIG, ASDIV_SPLIT),
        normalize_asdiv,
        IntendedUse.TRAIN,
        ASDIV_RUBRIC,
        RecipeInputs(
            SourceFiles(("ASDiv.xml",), SourceFormat.XML, reader=asdiv_rows),
            (UrlDownload(ASDIV_SOURCE_FILE, "ASDiv.xml"),),
        ),
    )


DAPO_MATH_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
DAPO_MATH_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
DAPO_MATH_CONFIG = "default"
DAPO_MATH_SPLIT = "train"
DAPO_MATH_SOURCE_FILE = "data/dapo-math-17k.parquet"
DAPO_MATH_SOURCE_FORMAT = "parquet"

DAPO_MATH_RUBRIC = ReviewRubric(
    id="dapo_math-quality",
    version="1",
    criteria=(
        "Preserve every prompt message and reward_model ground truth; source scorer parity is not implied by "
        "matching a reference.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_dapo_math(row: RawRow) -> TaskSpec | ImportRejection:
    messages, reward = row.data.get("prompt"), row.data.get("reward_model")
    if not isinstance(messages, list) or not messages or not isinstance(reward, dict) or "ground_truth" not in reward:
        return ImportRejection(
            reason="missing_prompt_or_reference", detail="prompt messages and reward_model ground_truth are required"
        )
    events = tuple(TextMessage(role=message["role"], content=message["content"]) for message in messages)
    evidence = {key: row.data[key] for key in ("reward_model", "data_source", "ability", "extra_info")}
    return math_task(row, events, str(reward["ground_truth"]), evidence)


def recipe_dapo_math() -> DatasetRecipe:
    return math_recipe(
        "dapo_math",
        HFSource(DAPO_MATH_DATASET, DAPO_MATH_REVISION, DAPO_MATH_CONFIG, DAPO_MATH_SPLIT),
        normalize_dapo_math,
        IntendedUse.TRAIN,
        DAPO_MATH_RUBRIC,
        hub_inputs(
            DAPO_MATH_DATASET,
            DAPO_MATH_REVISION,
            SourceFiles((DAPO_MATH_SOURCE_FILE,), SourceFormat(DAPO_MATH_SOURCE_FORMAT)),
        ),
    )


RLVR_MATH_DATASET = "allenai/RLVR-MATH"
RLVR_MATH_REVISION = "bd2a93551b503a395fadd1a740d957559cfe6f3c"
RLVR_MATH_CONFIG = "default"
RLVR_MATH_SPLIT = "train"
RLVR_MATH_SOURCE_FILE = "data/train-00000-of-00001.parquet"
RLVR_MATH_SOURCE_FORMAT = "parquet"

RLVR_MATH_RUBRIC = ReviewRubric(
    id="rlvr_math-quality",
    version="1",
    criteria=(
        "Retain public few-shot worked examples and distinguish them from the final question. This selected split is "
        "training data.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_rlvr_math(row: RawRow) -> TaskSpec | ImportRejection:
    messages, expected = row.data.get("messages"), row.data.get("ground_truth")
    if not isinstance(messages, list) or not messages or not isinstance(expected, str):
        return ImportRejection(
            reason="missing_prompt_or_reference", detail="messages and ground_truth strings are required"
        )
    events = tuple(TextMessage(role=message["role"], content=message["content"]) for message in messages)
    evidence = {key: row.data[key] for key in ("ground_truth", "dataset", "constraint_type", "constraint")}
    return math_task(row, events, expected, evidence)


def recipe_rlvr_math() -> DatasetRecipe:
    return math_recipe(
        "rlvr_math",
        HFSource(RLVR_MATH_DATASET, RLVR_MATH_REVISION, RLVR_MATH_CONFIG, RLVR_MATH_SPLIT),
        normalize_rlvr_math,
        IntendedUse.TRAIN,
        RLVR_MATH_RUBRIC,
        hub_inputs(
            RLVR_MATH_DATASET,
            RLVR_MATH_REVISION,
            SourceFiles((RLVR_MATH_SOURCE_FILE,), SourceFormat(RLVR_MATH_SOURCE_FORMAT)),
        ),
    )


NUMINA_MATH_DATASET = "AI-MO/NuminaMath-CoT"
NUMINA_MATH_REVISION = "9d8d210c9f6a36c8f3cd84045668c9b7800ef517"
NUMINA_MATH_CONFIG = "default"
NUMINA_MATH_SPLIT = "train"
NUMINA_MATH_SOURCE_FILE = "data/train-*.parquet"
NUMINA_MATH_SOURCE_FORMAT = "parquet"

NUMINA_MATH_RUBRIC = ReviewRubric(
    id="numina_math-quality",
    version="1",
    criteria=(
        "The boxed solution conclusion is a generated reference; assess its derivation rather than assuming "
        "its correctness.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize_numina_math(row: RawRow) -> TaskSpec | ImportRejection:
    problem, solution = row.data.get("problem"), row.data.get("solution")
    if not isinstance(problem, str) or not isinstance(solution, str):
        return ImportRejection(reason="missing_prompt_or_reference", detail="problem and solution strings are required")
    expected = extract_boxed(solution)
    if not expected:
        return ImportRejection(reason="missing_final_answer", detail="Solution lacks a boxed conclusion")
    return math_task(
        row, (TextMessage(role="user", content=problem),), expected, {"solution": solution, "source": row.data["source"]}
    )


def recipe_numina_math() -> DatasetRecipe:
    return math_recipe(
        "numina_math",
        HFSource(NUMINA_MATH_DATASET, NUMINA_MATH_REVISION, NUMINA_MATH_CONFIG, NUMINA_MATH_SPLIT),
        normalize_numina_math,
        IntendedUse.TRAIN,
        NUMINA_MATH_RUBRIC,
        hub_inputs(
            NUMINA_MATH_DATASET,
            NUMINA_MATH_REVISION,
            SourceFiles((NUMINA_MATH_SOURCE_FILE,), SourceFormat(NUMINA_MATH_SOURCE_FORMAT)),
        ),
    )


HARDMATH_REVISION = "937e9f10356e31e854f6efb9a2507f1e200c8b25"
HARDMATH_SOURCE_FILES = SourceFiles(patterns=("data/train-00000-of-00001.parquet",), format=SourceFormat.PARQUET)


def normalize_hardmath(row: RawRow) -> TaskSpec | ImportRejection:
    return normalize_math(row, "question", "ground_truths")


def recipe_hardmath() -> DatasetRecipe:
    return DatasetRecipe(
        name="hardmath",
        version="hardmath-v1",
        source=HFSource("pafitis/HARDMath_processed_training", HARDMATH_REVISION, "default", "train"),
        inputs=hub_inputs("pafitis/HARDMath_processed_training", HARDMATH_REVISION, HARDMATH_SOURCE_FILES),
        normalize=normalize_hardmath,
        intended_use=IntendedUse.TRAIN,
        rubric=ReviewRubric(
            id="hardmath-asymptotics",
            version="1",
            criteria=(
                "Check the question, private solution, and ground_truths together. Symbolic "
                "regimes, approximations, boundary conditions, and equations are part of the "
                "task.",
                "Compare every requested regime or deliverable with the reference list. A key "
                "omitting a requested asymptotic regime is a concrete mismatch.",
                "Check limiting powers and coefficients before certifying asymptotic formulas; "
                "do not confuse necessary and sufficient regimes or invent precision "
                "requirements.",
                "The training source is not an evaluation benchmark binding. The cleanup typed "
                "math comparator is used without claiming source scorer parity; hard mathematics "
                "alone is not a defect.",
            ),
        ),
        check_suite=CheckSuite(
            id="hardmath-math-controls",
            revision="1",
            parameters={"comparator": "cleanup-math-verify"},
            run=math_controls,
        ),
    )


HENDRYCKS_MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
HENDRYCKS_MATH_SOURCE_FILES = SourceFiles(
    patterns=("algebra/train-00000-of-00001.parquet",), format=SourceFormat.PARQUET
)


def normalize_hendrycks_math(row: RawRow) -> TaskSpec | ImportRejection:
    solution = row.data.get("solution")
    if isinstance(solution, str) and r"\boxed" not in solution:
        return ImportRejection(reason="missing_final_answer", detail="A boxed final answer is required in solution")
    return normalize_math(row, "problem", "solution")


def recipe_hendrycks_math() -> DatasetRecipe:
    return DatasetRecipe(
        name="hendrycks_math",
        version="hendrycks-math-algebra-train-v1",
        source=HFSource("EleutherAI/hendrycks_math", HENDRYCKS_MATH_REVISION, "algebra", "train"),
        inputs=hub_inputs("EleutherAI/hendrycks_math", HENDRYCKS_MATH_REVISION, HENDRYCKS_MATH_SOURCE_FILES),
        normalize=normalize_hendrycks_math,
        intended_use=IntendedUse.TRAIN,
        rubric=ReviewRubric(
            id="hendrycks-math-algebra-train",
            version="1",
            criteria=(
                "Preserve the complete algebra problem, domains, quantifiers, units, and requested answer form.",
                "The last boxed solution answer is private reference evidence. Check short "
                "calculations and contradictions; ordered pairs and half-open intervals are "
                "different contracts.",
                "Only the algebra train split is bound. MATH test and MATH-500 remain evaluation "
                "sources and must not be merged through this binding.",
                "Assess well-posedness independently of difficulty. The cleanup typed math "
                "comparator is used without claiming original scorer parity.",
            ),
        ),
        check_suite=CheckSuite(
            id="hendrycks-math-controls",
            revision="1",
            parameters={"comparator": "cleanup-math-verify"},
            run=math_controls,
        ),
    )


DEEPSCALER_REVISION = "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29"
DEEPSCALER_SOURCE_FILES = SourceFiles(patterns=("deepscaler.json",), format=SourceFormat.JSON)


def normalize_deepscaler(row: RawRow) -> TaskSpec | ImportRejection:
    return normalize_math(row, "problem", "answer")


def recipe_deepscaler() -> DatasetRecipe:
    return DatasetRecipe(
        name="deepscaler",
        version="deepscaler-v1",
        source=HFSource("agentica-org/DeepScaleR-Preview-Dataset", DEEPSCALER_REVISION, "default", "train"),
        inputs=hub_inputs("agentica-org/DeepScaleR-Preview-Dataset", DEEPSCALER_REVISION, DEEPSCALER_SOURCE_FILES),
        normalize=normalize_deepscaler,
        intended_use=IntendedUse.TRAIN,
        rubric=ReviewRubric(
            id="deepscaler-math",
            version="1",
            criteria=(
                "Check the problem and private answer for a unique mathematical result; preserve "
                "domains, LaTeX, and units.",
                "The answer field is the key. A solution ending with an option letter does not "
                "override a numeric answer; compare the actual arithmetic before alleging a "
                "contradiction.",
                "This training blend includes historical AIME/AMC, Omni-MATH, and Still "
                "problems. Retain provenance and do not treat it as uncontaminated evaluation "
                "data.",
                "The cleanup typed math comparator is used; source reward-scorer parity is not "
                "claimed. Difficulty and inability to solve immediately are not defects.",
            ),
        ),
        check_suite=CheckSuite(
            id="deepscaler-math-controls",
            revision="1",
            parameters={"comparator": "cleanup-math-verify"},
            run=math_controls,
        ),
    )


RECIPES: dict[str, DatasetRecipe] = {
    "math500": recipe_math500(),
    "aime_1983_2024": recipe_aime_1983_2024(),
    "gsm8k": recipe_gsm8k(),
    "asdiv": recipe_asdiv(),
    "dapo_math": recipe_dapo_math(),
    "rlvr_math": recipe_rlvr_math(),
    "numina_math": recipe_numina_math(),
    "hardmath": recipe_hardmath(),
    "hendrycks_math": recipe_hendrycks_math(),
    "deepscaler": recipe_deepscaler(),
}
