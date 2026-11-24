# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove identity-following tasks, repaired to Marin's true identity (ruling 2026-11-23).

The source's adapter pinned a false persona into every task: the prompt, the prompt's embedded
rubric, the verifier principle and the archived judge configuration all state that Marin was
trained and developed by researchers from NVIDIA. This converter applies the versioned identity
repair from ``identity_following_repair`` to the unpacked archive before the judged-task
conversion, rewrites the source's ``/app/response.txt`` delivery onto the assistant-response
convention, and records both rewrites as normalization changes. The grading contract is the binary
full-rubric labels judge over the source's own two criteria — the repaired identity criterion and
the pinned response language — matching the tiered judged sources' MEDIUM profile, so an NVIDIA
attribution scores zero instead of a partial checklist fraction.
"""

import base64
import json
import re
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Any

from finestore.rl.curation import DatasetRelease, OracleValidation, RecipeQAHistoricalNotes, SourceQualityResult
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.delivery import rewritten_task
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizationChange, NormalizedTask, RawRow
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import JudgeSpec

from experiments.post_training.task_curation.datasets.tasktrove.archives import TaskTroveConverter
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import VERIFIER_DATA, archive_file
from experiments.post_training.task_curation.identity_following_repair import (
    CONFIG,
    GRADER_CRITERION,
    IMPORTER_REVISION,
    INSTRUCTION_FILE,
    JUDGE_CONFIG_FILE,
    PERSONA_STATEMENT,
    REPAIR_RULE,
    REPAIRED_GRADER_CRITERION,
    REPAIRED_PERSONA_STATEMENT,
    REPAIRED_TRUTHFUL_PHRASE,
    TRUTHFUL_PHRASE,
    VERIFIER_DATA_FILE,
    repair_archive,
)
from experiments.post_training.task_curation.judges import JUDGE_TIERS, JudgeTier
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_prebuilt
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

DELIVERY_REWRITE_REASON = "Replace the source's response-file delivery with the assistant response convention"
IDENTITY_REWRITE_REASON = (
    f"{REPAIR_RULE}: replace the source's false NVIDIA persona with Marin, built by researchers at "
    "Stanford and Open Athena, in the prompt, the embedded rubric, the verifier principle and the "
    "archived judge configuration"
)
RESPONSE_FILE_DELIVERY = (
    "  - Save ONLY that final answer to `/app/response.txt` (no shell transcripts, no JSON wrappers, "
    "no scratch reasoning).",
    "  - Return ONLY that final answer in the assistant response (no shell transcripts, no JSON "
    "wrappers, no scratch reasoning).",
)
TASK_COMPLETE_DELIVERY = (
    "  - Set `task_complete: true` once `/app/response.txt` is written.",
    "  - Return your complete answer in the assistant response; no file is written.",
)
DELIVERY_REPLACEMENTS: tuple[tuple[str, str], ...] = (RESPONSE_FILE_DELIVERY, TASK_COMPLETE_DELIVERY)
CRITERIA_HEADER = "Criteria:"
NUMBERED = re.compile(r"^\s*\d+[.)]\s*")
IDENTITY_CHANGE_ORIGINAL = "\n\n".join((PERSONA_STATEMENT, TRUTHFUL_PHRASE, GRADER_CRITERION))
IDENTITY_CHANGE_REPLACEMENT = "\n\n".join(
    (REPAIRED_PERSONA_STATEMENT, REPAIRED_TRUTHFUL_PHRASE, REPAIRED_GRADER_CRITERION)
)


def identity_criteria(principle: str) -> tuple[str, ...]:
    """The principle's numbered criteria under its ``Criteria:`` header.

    The principle wraps its scored criteria in evaluation boilerplate ("You will be given one or
    more evaluation criteria...") that the original judge read as guidance. Only the numbered
    items under the header are criteria; everything else must not become a yes/no checklist entry.
    """
    body = principle.partition(CRITERIA_HEADER)[2]
    return tuple(
        NUMBERED.sub("", line).strip() for line in body.splitlines() if line.strip() and NUMBERED.match(line.strip())
    )


def identity_public_instruction(instruction: str) -> str:
    """Rewrite the archived response-file delivery onto the assistant-response convention."""
    for original, replacement in DELIVERY_REPLACEMENTS:
        occurrences = instruction.count(original)
        if occurrences != 1:
            raise ValueError(f"Expected exactly one delivery line, found {occurrences}: {original[:80]!r}")
        instruction = instruction.replace(original, replacement)
    return instruction


def repair_row_data(data: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the versioned identity repair to one unpacked archive row.

    Returns the repaired row data and the per-file before/after hash ledger. The instruction text,
    the archived file payloads and the parsed verifier data all come from the repaired files, so
    every conversion input is the repaired bytes.
    """
    files = {
        name: base64.b64decode(value, validate=True) for name, value in data["files"].items() if isinstance(value, str)
    }
    repaired_files, ledger = repair_archive(dict(files))
    repaired_data = dict(data)
    repaired_data["files"] = {name: base64.b64encode(blob).decode() for name, blob in repaired_files.items()}
    repaired_data["instruction"] = repaired_files[INSTRUCTION_FILE].decode("utf-8")
    repaired_data["verifier_data"] = json.loads(repaired_files[VERIFIER_DATA_FILE])
    return repaired_data, ledger


def convert_repaired_identity(
    row: RawRow, context: ConversionContext, *, judge: JudgeSpec
) -> TaskSpec | NormalizedTask | ImportRejection:
    """Convert one identity-following row: repair the persona, then build the judged task."""
    data = row.data.get("verifier_data")
    if not isinstance(data, dict):
        return unsupported("missing_judge_data", "Source verifier_data.json is required")
    principle = data.get("principle")
    criteria = identity_criteria(principle) if isinstance(principle, str) else ()
    question = data.get("instruction")
    if not isinstance(question, str) or not question.strip() or len(criteria) != 2 or not all(criteria):
        return unsupported("invalid_rubric", "The source principle must carry exactly two numbered criteria")
    judge_toml = archive_file(row.data, JUDGE_CONFIG_FILE)
    if judge_toml is None:
        return unsupported("missing_judge_config", "Original source judge.toml is required")
    instruction = row.data["instruction"]
    if not instruction.strip():
        return source_defect("missing_instruction", "Public instruction is required")
    verifier_data = archive_file(row.data, VERIFIER_DATA)
    assert verifier_data is not None
    public_instruction = identity_public_instruction(instruction)
    rubric = json.dumps(
        [{"id": str(index + 1), "criteria": criterion} for index, criterion in enumerate(criteria)],
        ensure_ascii=False,
    )
    spec = replace(
        judge,
        question=public_instruction,
        references=(rubric,),
    )
    package = verifyit_package(
        spec,
        (inline_resource("source/judge.toml", judge_toml), inline_resource("source/verifier_data.json", verifier_data)),
    )
    prompt = TextMessage(role="user", content=public_instruction)
    task = conversation_task(row, events=(prompt,), package=package)
    task = task.model_copy(update={"tags": ("judge", "rubric", "no-reference", "identity", "freeform")})
    identity_change = NormalizationChange(
        field="instruction",
        reason=IDENTITY_REWRITE_REASON,
        original=IDENTITY_CHANGE_ORIGINAL,
        replacement=IDENTITY_CHANGE_REPLACEMENT,
    )
    delivery = rewritten_task(task, original=instruction, reason=DELIVERY_REWRITE_REASON)
    if isinstance(delivery, NormalizedTask):
        return NormalizedTask(delivery.task, (identity_change, *delivery.changes))
    return NormalizedTask(task, (identity_change,))


JUDGE_TEMPERATURE = 0.7
"""The standing judge temperature for every grading contract (worker policy)."""
JUDGE_INCOMPLETE_RETRY_TOKENS = 16384
"""The retry completion budget when the judge exhausts its primary budget without a verdict.

The MEDIUM tier's 2048-token primary budget was measured exhausting without a verdict on this
source's adaptive-thinking judge (the unix cohort recorded the same), so the repaired contract
carries an explicit incomplete-completion retry budget instead of failing the grading attempt.
"""


def identity_judge() -> JudgeSpec:
    """The repaired source's pinned judge contract: MEDIUM binary full-rubric, standing temperature,
    with the explicit incomplete-completion retry budget."""
    return replace(
        JUDGE_TIERS[JudgeTier.MEDIUM].spec,
        temperature=JUDGE_TEMPERATURE,
        incomplete_retry_tokens=JUDGE_INCOMPLETE_RETRY_TOKENS,
    )


REPAIRED_VARIANT_ID = f"TaskTrove:{CONFIG}@repaired-20261123"
REPAIRED_RELEASE_REPO = "open-athena/tasktrove-identity-following-repaired-20261123"
REPAIRED_RELEASE_REVISION = "94967dd29123f9d1274a660491b6612fbd6ebd40"
REPAIRED_TASK_COUNT = 21_660


def repaired_sources() -> list[Any]:
    """The repaired variant's catalog declarations; see :func:`repaired_variant_source`."""
    return [repaired_variant_source()]


def repaired_variant_source() -> Any:
    """The repaired population's catalog declaration, pending its admission receipts.

    The repaired revision re-enters ingestion at stage b (a converter/prompt change), so readmission
    needs fresh within-source dedup, cross-source overlap, lexical, census and admission bindings at
    the variant identity before the declaration may land — the declaration audit refuses a declared
    source without its dedup receipt, so this entry stays unwired from ``configured_sources`` until
    the campaign produces them. The QA receipts bound to the revision:
    ``s3://marin-us-east-02a/marin/rl-data/atlas/identity-following-repair/20261123/qa-r3/controls.json``
    (oracle/NoP controls, 90/90 as expected) and
    ``s3://marin-us-east-08a/marin/rl-data/atlas/identity-following-repair/20261123/qa-r3/sample-review.json``
    (SAMPLE review, 120/120 good, 0 unavailable).
    """

    cohort = PrebuiltCohort(
        intended_use=IntendedUse.TRAIN,
        name="tasktrove-identity-following",
        version=IMPORTER_REVISION,
        repo=REPAIRED_RELEASE_REPO,
        revision=REPAIRED_RELEASE_REVISION,
        expected_rows=REPAIRED_TASK_COUNT,
    )
    return RlDataSource(
        pipeline=import_taskcompendium_prebuilt,
        config=cohort,
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            id=REPAIRED_VARIANT_ID,
            upstream_dataset=CONFIG,
            upstream_name=CONFIG,
            subset=None,
            variant="repaired-20261123",
            build_path="TaskTrove identity repair",
            origin=SourceOrigin.TASKTROVE,
            upstream_url=(
                "https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/"
                "02923004846e4e73862c20962f823a6d05100e7a"
            ),
            family="llm-judge-freeform",
            tags=("judge", "rubric", "no-reference", "identity", "repair-release", "converted"),
            count=REPAIRED_TASK_COUNT,
            task_format=TaskFormat.TASKSPEC,
            current_release=DatasetRelease(
                repo=REPAIRED_RELEASE_REPO,
                revision=REPAIRED_RELEASE_REVISION,
                url=f"https://huggingface.co/datasets/{REPAIRED_RELEASE_REPO}/tree/{REPAIRED_RELEASE_REVISION}",
            ),
            oracle_validation=OracleValidation.VERIFIER_EXECUTED,
            recipe_qa_history=RecipeQAHistoricalNotes(
                campaign="identity-following repair 2026-11-23",
                recipe_revision=IMPORTER_REVISION,
                verification=(
                    "Rule identity-repair-20261123-v2 replaced the three false-persona defect strings in "
                    "every task; the pinned MEDIUM grader accepted the correct identity and rejected every "
                    "NVIDIA attribution across a 30-task stratified control panel (90/90 as expected), and "
                    "the pinned V4 GLM reviewer scored 120/120 good with zero unavailable on the stratified "
                    "SAMPLE panel."
                ),
                reviewed=120,
                admitted=0,
            ),
            quality_result=SourceQualityResult(
                reviewed=120,
                issue_count=0,
                uncertain_count=0,
                unavailable_count=0,
                issues=(),
                repair_recommendation="No repairs indicated by conclusive reviews.",
            ),
            notes=(
                "Repaired full population of TaskTrove:laion__nemotron-gym-identity-following-v4: every "
                "task carried a false 'Marin trained and developed by researchers from NVIDIA' persona in "
                "its prompt, embedded rubric, verifier principle and archived judge config; rule "
                "identity-repair-20261123-v2 replaces exactly those strings (Stanford and Open Athena "
                "attribution) with everything else byte-identical. The parent source stays Excluded "
                "(NO_SUPPORTED_BACKEND, ruling 2026-11-17) until this variant's admission receipts bind."
            ),
        ),
    )


def identity_converter() -> TaskTroveConverter:
    """The TaskTrove converter with the identity repair for the identity-following config.

    The judge contract is the pinned MEDIUM tier's binary full-rubric profile at the standing
    judge temperature with an explicit incomplete-completion retry budget, matching the tiered
    judged sources' conversion.
    """
    return TaskTroveConverter(CONFIG, partial(convert_repaired_identity, judge=identity_judge()))
