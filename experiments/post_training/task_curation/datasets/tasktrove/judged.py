# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove open-ended response sources graded by an LLM judge against the source's own rubric.

Each archive asks for a free-form answer in ``/app/response.txt`` and grades it with one holistic
numeric judgement over a short rubric (``tests/judge.toml``, ``tests/verifier_data.json``). The
task asks for the answer in the reply instead. The MEDIUM profile awards one only for full
satisfaction of the source rubric and otherwise zero. The checklist profile scores the fraction
of criteria met. The source's judge configuration and data stay beside the grader for review.
Pinned source profiles remain QA candidates until qualification and publication are complete.
"""

import json
import re
from dataclasses import dataclass, field, replace
from functools import partial

from finestore.rl.curation import DatasetRelease, OracleValidation, SourceQualityResult
from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.delivery import rewritten_task
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizedTask, RawRow
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import RUBRIC_CHECKLIST, RUBRIC_LABELS, JudgeSpec

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.tasktrove.archives import TaskTroveConverter, tasktrove_source
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import VERIFIER_DATA, archive_file
from experiments.post_training.task_curation.judges import JUDGE_TIERS, JudgeTier
from experiments.post_training.task_curation.pipeline import CurationRecipe, process_rows
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_prebuilt
from experiments.post_training.task_curation.source import (
    OversightEvidence,
    OversightLevel,
    RlDataSource,
    SourceInfo,
    SourceOrigin,
    TaskFormat,
)

REWRITE_REASON = "Replace source response-file delivery with the assistant response convention"
QA_PINNED_SOURCES = frozenset({"tasktrove-safety"})
"""Sources whose QA pinned the judge tier while keeping their own pipeline, grader and evidence."""

TIERED_JUDGE_SOURCES = frozenset(
    {
        "tasktrove-safety",
        "tasktrove-codereview",
        "tasktrove-glaive_code",
        "tasktrove-stack_overflow",
        "tasktrove-superuser",
        "tasktrove-tezos",
        "tasktrove-unix",
        "tasktrove-wizard_orca",
    }
)
MECHANICALLY_RETIRED_JUDGE_SOURCES = frozenset({"tasktrove-glaive_code", "tasktrove-tezos"})
JUDGE_QUALIFICATION_PENDING = (
    " Inventory: conversion pins MEDIUM MiniMax M3 with binary full-rubric grading and no source gold answers. "
    "Source-level qualification and a private pinned release are required before admission."
)

QUICK_COHORTS = {
    "tasktrove-codereview": PrebuiltCohort(
        intended_use=IntendedUse.TRAIN,
        name="tasktrove-codereview",
        version="3",
        repo="open-athena/tasktrove-codereview-medium-quick-20261112",
        revision="52c90a79a59262b1a25f76992430a1d81e42f0e4",
        expected_rows=10000,
    ),
    "tasktrove-stack_overflow": PrebuiltCohort(
        intended_use=IntendedUse.TRAIN,
        name="tasktrove-stack_overflow",
        version="3",
        repo="open-athena/tasktrove-stack_overflow-medium-quick-20261112",
        revision="07603684451d8db4caef836dbb07c89a4a7e0588",
        expected_rows=10000,
    ),
    "tasktrove-superuser": PrebuiltCohort(
        intended_use=IntendedUse.TRAIN,
        name="tasktrove-superuser",
        version="3",
        repo="open-athena/tasktrove-superuser-medium-quick-20261112",
        revision="01237834f87d26f4c26f92d0e876904071dd2dd1",
        expected_rows=10000,
    ),
    "tasktrove-unix": PrebuiltCohort(
        intended_use=IntendedUse.TRAIN,
        name="tasktrove-unix",
        version="3",
        repo="open-athena/tasktrove-unix-medium-quick-20261112",
        revision="2287a9698aa508aa6ae442d923d2cd68c2b7075c",
        expected_rows=10000,
    ),
    "tasktrove-wizard_orca": PrebuiltCohort(
        intended_use=IntendedUse.TRAIN,
        name="tasktrove-wizard_orca",
        version="3",
        repo="open-athena/tasktrove-wizard_orca-medium-quick-20261112",
        revision="04b33f3773ddadb6dd0f35a79d7a56b62dd4068b",
        expected_rows=9999,
    ),
}

QUALITY_RESULTS = {
    "tasktrove-codereview": SourceQualityResult(
        reviewed=8,
        issue_count=0,
        uncertain_count=12,
        unavailable_count=0,
        issues=[],
        repair_recommendation=(
            "Recover missing original inputs and review uncertain cases " "without changing the source question."
        ),
    ),
    "tasktrove-stack_overflow": SourceQualityResult(
        reviewed=17,
        issue_count=1,
        uncertain_count=3,
        unavailable_count=0,
        issues=[
            "tasktrove-stack_overflow-29d641c5f112650cb2e0b221c7f7506f69666fd4597b4a2e031c95c9f6c4507a: "
            "The critical sed expression is truncated; its pattern, quoting and pipeline tail are missing.",
        ],
        repair_recommendation=(
            "Recover missing original inputs and review uncertain cases " "without changing the source question."
        ),
    ),
    "tasktrove-superuser": SourceQualityResult(
        reviewed=17,
        issue_count=0,
        uncertain_count=3,
        unavailable_count=0,
        issues=[],
        repair_recommendation=(
            "Recover missing original inputs and review uncertain cases " "without changing the source question."
        ),
    ),
    "tasktrove-unix": SourceQualityResult(
        reviewed=20,
        issue_count=0,
        uncertain_count=0,
        unavailable_count=0,
        issues=[],
        repair_recommendation=(
            "Validate a sufficient judge completion budget and rerun the frozen unscored control; "
            "no confirmed task content issue was found."
        ),
    ),
    "tasktrove-wizard_orca": SourceQualityResult(
        reviewed=19,
        issue_count=3,
        uncertain_count=1,
        unavailable_count=0,
        issues=[
            "tasktrove-wizard_orca-d22aa7b5e903f253fb176f733d0a1b160afa886d5de2f366a782fcfefd12bb8b: "
            "The requested existing Ruby code and reword() implementation are missing.",
            "tasktrove-wizard_orca-547dae17b2c05be34efc2db58127165cde1210d6a92bacb6450ca40415a10a46: "
            "The required XML records, region data and lifespan values are missing.",
            "tasktrove-wizard_orca-6cdc5e58f06dbd27396a74f0dc0bb15db93d38404232932c4a62f8c7c142a1e8: "
            "The numerical question omits unit prices and discount rules.",
        ],
        repair_recommendation=(
            "Recover missing original inputs and review uncertain cases " "without changing the source question."
        ),
    ),
}

NUMBERED = re.compile(r"^\s*\d+[.)]\s*")
DOMAIN_TAGS = (
    ("Stack Overflow", ("stackexchange", "stackoverflow", "code")),
    ("Super User", ("stackexchange", "superuser")),
    ("Unix & Linux", ("stackexchange", "unix", "shell")),
    ("Tezos", ("stackexchange", "tezos")),
    ("Code Review", ("stackexchange", "codereview", "code")),
    ("Glaive", ("code-assistant", "code")),
    ("general instruction-following", ("general-assistant",)),
)
DELIVERY_SENTENCE = (
    "After you have completed your analysis and formulated your answer, you MUST write your final, "
    "comprehensive response into a file named 'response.txt' in the current directory.",
    "Return your final, comprehensive assistant response.",
)
DELIVERY_HEADER = "\n\n## Submitting your answer (IMPORTANT)\n"
TERMINAL_FOOTER = "You are a terminal agent."
FOOTER_REPLACEMENT = "\n\nReturn your complete answer in the assistant response."
PREAMBLE_SEPARATOR = "\n---\n"
HEREDOC_GUIDANCE = re.compile(
    r"  - To write the response from a shell, use a heredoc, e\.g\.:\n"
    r"        cat > /app/response\.txt <<'EOF'\n        [^\n]+\n        EOF\n"
)
PREAMBLE_PHRASES = (
    ("  - Verify with `cat /app/response.txt` before marking the task complete. ", "  - "),
    ("  - Verify with `cat /app/response.txt` before marking the task complete.\n", ""),
    ("to the file `/app/response.txt` inside the sandbox", "in the assistant response"),
    ("to `/app/response.txt`", "in the assistant response"),
    ("Empty or missing files score 0.", "An empty response scores 0."),
)

STACK_EXCHANGE_CRITERIA = """
Require a complete public request and any code, prior turns, or external passages needed to answer it.

Check each hidden criterion against the public request; flag invented constraints or incorrect premises.

The source used one holistic numeric judge over four criteria and has no gold answer. The converted MEDIUM profile
uses binary full-rubric grading; the original judge configuration is retained as source evidence."""

HOLISTIC_SOURCE_CRITERIA = """
Reject missing inputs, contradictions, and rubric requirements absent from the public request.

The source graded one holistic numeric reward; its rubric, judge prompt and threshold are kept beside the grader.

No reference answer is supplied; do not invent one.

An unavailable judge is a verification limitation, not evidence that the content is bad."""

CODEREVIEW_RUBRIC = f"""{STACK_EXCHANGE_CRITERIA}

Require the code under review and its intended behavior; inspect flattened source for lost comparisons, markup, links,
and abrupt truncation before judging it complete.
"""

GLAIVE_CODE_RUBRIC = f"""{STACK_EXCHANGE_CRITERIA}

Check required code, table schemas, input, output, dependencies, and runtime assumptions; generic programming questions
can be answerable, but references to absent specific inputs are defects.
"""

SAFETY_RUBRIC = f"""
Compare each safety principle with the actual request; refusal on a benign request is a rubric mismatch.
{HOLISTIC_SOURCE_CRITERIA}
"""

STACK_OVERFLOW_RUBRIC = f"""{STACK_EXCHANGE_CRITERIA}

Check that error reports include relevant code, versions, input, and observed behavior; distinguish plausible advice
from an answer justified by supplied context.
"""

SUPERUSER_RUBRIC = f"""{STACK_EXCHANGE_CRITERIA}

Check operating system, application, privileges, and device assumptions; missing environment details can make a
checklist demand impossible or unsafe.
"""

TEZOS_RUBRIC = f"""
Check that Tezos questions supply necessary code, transaction details, versions, and error context.
{HOLISTIC_SOURCE_CRITERIA}
"""

UNIX_RUBRIC = f"""{STACK_EXCHANGE_CRITERIA}

Check shell, distribution, filesystem, quoting, permissions, and tool assumptions; different valid commands must not
be excluded by an arbitrary checklist.
"""

WIZARD_ORCA_RUBRIC = f"""
Trace supplied code with each stated example, including actual printed strings, divisions, return values, and
arithmetic. A purported correct example that disagrees with the code is a defect unless the public task explicitly asks
to debug or correct that discrepancy.

Check the complete instruction, facts, and requested reasoning against the hidden source rubric.
{HOLISTIC_SOURCE_CRITERIA}
"""


def response_instruction(instruction: str) -> str:
    """Rewrite the source's ``/app/response.txt`` delivery wording, leaving the request after the preamble intact."""
    instruction = instruction.replace(*DELIVERY_SENTENCE)
    body, marker, footer = instruction.rpartition(DELIVERY_HEADER)
    if marker and footer.startswith(TERMINAL_FOOTER):
        instruction = body + FOOTER_REPLACEMENT
    preamble, separator, request = instruction.partition(PREAMBLE_SEPARATOR)
    if not separator:
        return instruction
    preamble = HEREDOC_GUIDANCE.sub("", preamble)
    for original, replacement in PREAMBLE_PHRASES:
        preamble = preamble.replace(original, replacement)
    return preamble + separator + request


def source_criteria(data: dict) -> tuple[str, ...]:
    """The rubric's criteria, or the numbered lines of its principle when it has no rubric."""
    rubric = data.get("rubric")
    if isinstance(rubric, list):
        return tuple(entry.get("criteria", "").strip() for entry in rubric if isinstance(entry, dict))
    principle = data.get("principle")
    if not isinstance(principle, str):
        return ()
    return tuple(NUMBERED.sub("", line).strip() for line in principle.splitlines() if line.strip())


def convert_judged(
    row: RawRow, context: ConversionContext, *, judge: JudgeSpec
) -> TaskSpec | NormalizedTask | ImportRejection:
    data = row.data.get("verifier_data")
    if not isinstance(data, dict):
        return unsupported("missing_judge_data", "Source verifier_data.json is required")
    criteria, question = source_criteria(data), data.get("instruction")
    if not isinstance(question, str) or not question.strip() or not criteria or not all(criteria):
        return unsupported("invalid_rubric", "Source question and nonempty criteria are required")
    judge_toml = archive_file(row.data, "tests/judge.toml")
    if judge_toml is None:
        return unsupported("missing_judge_config", "Original source judge.toml is required")
    instruction = row.data["instruction"]
    if not instruction.strip():
        return source_defect("missing_instruction", "Public instruction is required")
    verifier_data = archive_file(row.data, VERIFIER_DATA)
    assert verifier_data is not None
    public_instruction = response_instruction(instruction)
    if judge.rubric == RUBRIC_LABELS:
        rubric = data.get("rubric")
        if not isinstance(rubric, list) or len(rubric) != len(criteria):
            return unsupported("invalid_rubric", "The binary judge profile requires a complete source criterion list")
        spec = replace(
            judge,
            question=public_instruction,
            references=(json.dumps(rubric, ensure_ascii=False),),
        )
        environment = None
    elif judge.rubric == RUBRIC_CHECKLIST:
        spec = replace(judge, criteria=criteria, question=question)
        environment = required_grader_environment(context)
    else:
        raise ValueError(f"Unsupported source judge rubric: {judge.rubric}")
    package = verifyit_package(
        spec,
        (inline_resource("source/judge.toml", judge_toml), inline_resource("source/verifier_data.json", verifier_data)),
        environment=environment,
    )
    prompt = TextMessage(role="user", content=public_instruction)
    task = conversation_task(row, events=(prompt,), package=package)
    domain = (
        ("safety",)
        if isinstance(data.get("principle"), str)
        else next(
            (tags for name, tags in DOMAIN_TAGS if name in str(data.get("judge_system_prompt", ""))), ("freeform",)
        )
    )
    task = task.model_copy(update={"tags": ("judge", "rubric", "no-reference", *domain)})
    return rewritten_task(task, original=instruction, reason=REWRITE_REASON)


@dataclass(frozen=True)
class JudgedSource:
    name: str
    config: str
    rubric: str
    info: SourceInfo = field(kw_only=True)


SOURCES = (
    JudgedSource(
        "tasktrove-codereview",
        "laion__stackexchange-codereview-sandboxes-verified-v2",
        CODEREVIEW_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:laion__stackexchange-codereview-sandboxes-verified-v2",
            upstream_dataset="laion__stackexchange-codereview-sandboxes-verified-v2",
            upstream_name="laion__stackexchange-codereview-sandboxes-verified-v2",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="llm-judge-freeform",
            tags=("agentic", "multi-turn"),
            count=10000,
            notes=(
                "Judge-only code review with a four-criterion rubric and no reference; the rubric "
                "ships under tests/, which the agent never sees. Kept as rubric-only (tag no-"
                "reference)."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    JudgedSource(
        "tasktrove-glaive_code",
        "laion__glaive-code-assistant-sandboxes-verified-v2",
        GLAIVE_CODE_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:laion__glaive-code-assistant-sandboxes-verified-v2",
            upstream_dataset="laion__glaive-code-assistant-sandboxes-verified-v2",
            upstream_name="laion__glaive-code-assistant-sandboxes-verified-v2",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="llm-judge-freeform",
            tags=("agentic", "multi-turn"),
            count=10000,
            notes=(
                "Judge-only over code that is never run, four-criterion rubric, no reference. Kept as "
                "rubric-only (tag no-reference)."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    JudgedSource(
        "tasktrove-safety",
        "laion__nemotron-gym-safety-v3",
        SAFETY_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:laion__nemotron-gym-safety-v3",
            upstream_dataset="laion__nemotron-gym-safety-v3",
            upstream_name="laion__nemotron-gym-safety-v3",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="llm-judge-freeform",
            tags=("agentic", "multi-turn"),
            count=89066,
            notes=(
                "Judge-only against a two-line safety principle, no reference. Kept as rubric-only "
                "(tag no-reference): the principle is specific to each prompt (refuse the harmful "
                "ones, help with the benign ones). QA (RL Data Atlas): every row re-converted with "
                "zero fidelity discrepancies; 44,127 exact duplicates collapse under the pipeline "
                "dedup key (44,939 published); MEDIUM judge e2e on the seeded sample through the "
                "production grading path, agreement 59/60 across the MEDIUM pair (one conservative "
                "0.5 disagreement on this source)."
            ),
            oversight=OversightEvidence(
                campaign="RL Data Atlas",
                recipe_revision="datasets/tasktrove/judged.py version 3 "
                "(QA_PINNED_SOURCES: MEDIUM tier settings under RUBRIC_CHECKLIST); "
                "archive open-thoughts/TaskTrove @ 02923004846e",
                verification=(
                    "All 89,066 rows re-converted through the production recipe and compared against the "
                    "archive row-by-row with zero discrepancies; 44,127 exact duplicates collapse under "
                    "the pipeline dedup key, 44,939 retained. MEDIUM-tier checklist judge e2e on the "
                    "10-task seeded sample through the production grading boundary at temperature 0.7: "
                    "this source graded 29/30 as expected (the one disagreement is a conservative 0.5 on "
                    "a wrong submission judged against the expected 0.0). Self-judged verdicts recorded "
                    "before any judge run. Evidence: atlas-env-audit qa-knowledge/"
                ),
                level=OversightLevel.SAMPLE,
                reviewed=10,
                admitted=44939,
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    JudgedSource(
        "tasktrove-stack_overflow",
        "laion__stackexchange-overflow-sandboxes-verified-v2",
        STACK_OVERFLOW_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:laion__stackexchange-overflow-sandboxes-verified-v2",
            upstream_dataset="laion__stackexchange-overflow-sandboxes-verified-v2",
            upstream_name="laion__stackexchange-overflow-sandboxes-verified-v2",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="llm-judge-freeform",
            tags=("agentic", "multi-turn"),
            count=10000,
            notes=(
                "Judge-only Stack Overflow answers, four-criterion rubric, no reference. Kept as "
                "rubric-only (tag no-reference)."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    JudgedSource(
        "tasktrove-superuser",
        "laion__stackexchange-superuser-sandboxes-verified-v2",
        SUPERUSER_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:laion__stackexchange-superuser-sandboxes-verified-v2",
            upstream_dataset="laion__stackexchange-superuser-sandboxes-verified-v2",
            upstream_name="laion__stackexchange-superuser-sandboxes-verified-v2",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="llm-judge-freeform",
            tags=("agentic", "multi-turn"),
            count=10000,
            notes=(
                "Judge-only Super User answers, four-criterion rubric, no reference. Kept as rubric-"
                "only (tag no-reference)."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    JudgedSource(
        "tasktrove-tezos",
        "laion__stackexchange-tezos-sandboxes-verified-v2",
        TEZOS_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:laion__stackexchange-tezos-sandboxes-verified-v2",
            upstream_dataset="laion__stackexchange-tezos-sandboxes-verified-v2",
            upstream_name="laion__stackexchange-tezos-sandboxes-verified-v2",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="llm-judge-freeform",
            tags=("agentic", "multi-turn"),
            count=10000,
            notes=(
                "Judge-only Tezos answers, four-criterion rubric, no reference; the _copyN duplicates "
                "fall to the dedup step. Kept as rubric-only (tag no-reference)."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    JudgedSource(
        "tasktrove-unix",
        "laion__stackexchange-unix-sandboxes-verified-v2",
        UNIX_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:laion__stackexchange-unix-sandboxes-verified-v2",
            upstream_dataset="laion__stackexchange-unix-sandboxes-verified-v2",
            upstream_name="laion__stackexchange-unix-sandboxes-verified-v2",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="llm-judge-freeform",
            tags=("agentic", "multi-turn"),
            count=10000,
            notes=(
                "Judge-only Unix & Linux answers, four-criterion rubric, no reference. Kept as rubric-"
                "only (tag no-reference)."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    JudgedSource(
        "tasktrove-wizard_orca",
        "laion__wizardlm-orca-v4",
        WIZARD_ORCA_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:laion__wizardlm-orca-v4",
            upstream_dataset="laion__wizardlm-orca-v4",
            upstream_name="laion__wizardlm-orca-v4",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="llm-judge-freeform",
            tags=("agentic", "multi-turn"),
            count=10000,
            notes=(
                "Judge-only over Orca-style instructions, three-criterion rubric, no reference. Kept "
                "as rubric-only (tag no-reference)."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
)


def candidate_sources() -> list[RlDataSource[CurationRecipe]]:
    """Retain raw conversion recipes for explicit QA, including mechanically retired candidates."""
    return [
        RlDataSource(
            pipeline=None if source.name in MECHANICALLY_RETIRED_JUDGE_SOURCES else process_rows,
            info=replace(
                source.info,
                task_format=TaskFormat.TASKCOMPENDIUM,
                notes=(
                    source.info.notes + JUDGE_QUALIFICATION_PENDING
                    if source.name in TIERED_JUDGE_SOURCES - QA_PINNED_SOURCES
                    else source.info.notes
                ),
            ),
            config=CurationRecipe(
                name=source.name,
                source=tasktrove_source(source.config),
                convert=TaskTroveConverter(
                    source.config,
                    partial(
                        convert_judged,
                        judge=(
                            # Safety grades a source-principle checklist, so it takes the tier's
                            # model and completion settings under the checklist rubric; the
                            # reference-label sources keep the tier's own rubric.
                            replace(JUDGE_TIERS[JudgeTier.MEDIUM].spec, rubric=RUBRIC_CHECKLIST)
                            if source.name in QA_PINNED_SOURCES
                            else (
                                JUDGE_TIERS[JudgeTier.MEDIUM].spec
                                if source.name in TIERED_JUDGE_SOURCES
                                else JudgeSpec(rubric=RUBRIC_CHECKLIST)
                            )
                        ),
                    ),
                ),
                version="3" if source.name in TIERED_JUDGE_SOURCES else "2",
                intended_use=IntendedUse.TRAIN,
                rubric=source.rubric,
                grader=None if source.name in TIERED_JUDGE_SOURCES - QA_PINNED_SOURCES else GRADER_PACKAGES,
            ),
        )
        for source in SOURCES
    ]


def published_reply(source: RlDataSource[CurationRecipe]) -> RlDataSource[PrebuiltCohort]:
    """Consume the private converted population while preserving its upstream identity."""
    cohort = QUICK_COHORTS[source.name]
    upstream = source.dataset
    assert upstream is not None
    control_coverage = (
        "Unix has 19/20 parsed manual controls matching both reviewers; ed8810b16c9fb4c6 exhausted "
        "the 2048-token completion budget without a verdict. Twenty unrelated and twenty empty "
        "submissions score zero. The unscored completion is an execution issue, not a content defect."
        if source.name == "tasktrove-unix"
        else "Ten unique exact-spec manual controls match the blinded verdicts; unrelated and empty controls score zero."
    )
    return RlDataSource(
        config=cohort,
        pipeline=import_taskcompendium_prebuilt,
        info=replace(
            source.info,
            count=cohort.expected_rows,
            current_release=DatasetRelease(repo=cohort.repo, revision=cohort.revision, url=cohort.dataset.url),
            upstream_url=upstream.url,
            build_path="TaskTrove MEDIUM v3/private QUICK import",
            quality_result=QUALITY_RESULTS[source.name],
            tags=(
                *(tag for tag in source.info.tags if tag not in {"agentic", "multi-turn"}),
                "single-turn",
                "reply-only",
            ),
            oracle_validation=OracleValidation.VERIFIER_EXECUTED,
            oversight=OversightEvidence(
                level=OversightLevel.NONE,
                campaign="RL Data Atlas",
                recipe_revision="TaskTrove MEDIUM conversion version 3",
                verification="Private QUICK import; no SAMPLE/FULL qualification is claimed. "
                + control_coverage
                + " Receipts: "
                + cohort.dataset.url
                + "/production-controls.json",
                reviewed=20,
                admitted=0,
            ),
            notes=source.info.notes + " Private QUICK release reuses the complete converted TaskSpecs. "
            "QC Result is informational; uncertain review coverage and judge execution issues remain explicit. "
            "Raw upstream conversion counts and drops remain in the pinned conversion manifest and rejection ledger.",
        ),
    )


def sources() -> list[RlDataSource]:
    return [
        published_reply(source) if source.name in QUICK_COHORTS else source
        for source in candidate_sources()
        if source.pipeline is not None
    ]
