# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct generated Reasoning Gym entries with their pinned native scorer contract."""

from pydantic import BaseModel, ValidationError

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat, UrlDownload
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    VerificationReport,
)
from taskcompendium.pipeline.verification import verify_task
from taskcompendium.verifiers.source_contract import SourceContractVerifier

DATASET = "open-thought/reasoning-gym"
REVISION = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"
VERIFIER_REVISION = "bb6494e678ffaa1e6bd3967e221d1a67b038757e"
RUBRIC = ReviewRubric(
    id="direct-reasoning-gym-answerability",
    version="1",
    criteria=(
        (
            "Read the complete generated question and verify that every grid, rule, sequence, or example "
            "required to solve it is present."
        ),
        (
            "Independently check the private generated answer against the public problem where feasible; "
            "passing native scoring does not establish correctness."
        ),
        (
            "Preserve the task-native partial reward and answer parsing; a label or approximate substring "
            "match is not a substitute for that scorer."
        ),
        (
            "Judge default generated difficulty and unusual puzzle formats on their actual content, without "
            "assuming they are defects."
        ),
        (
            "Check non-discrimination when a scorer accepts incorrect alternatives, and record concrete "
            "semantic mismatches rather than rejecting an unbound runtime alone."
        ),
    ),
)


class RecordedReward(BaseModel):
    candidate: str
    reward: float


class RecordedControls(BaseModel):
    generator_revision: str
    positive: RecordedReward
    negative: RecordedReward
    execution: str


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    try:
        entry = row.data["entry"]
        generation = row.data["generation"]
        task_name = entry["metadata"]["source_dataset"]
        if task_name != generation["task"] or not isinstance(entry["answer"], str):
            raise ValueError("Generated entry and generation provenance must identify the same native task")
        recorded = row.data.get("recorded_pinned_generator_controls")
        controls = RecordedControls.model_validate(recorded) if recorded is not None else None
        context = ConversationInput(events=(TextMessage(role="user", content=entry["question"]),))
        verifier = SourceContractVerifier(
            evaluator="MarinSkyRL:skyrl_gym.envs.reasoning_gym.scoring.score_response",
            source_revision=VERIFIER_REVISION,
            contract={
                "task": task_name,
                "entry": entry,
                "generation": generation,
                "generator_revision": REVISION,
                "answer_extraction": "Text after the last Answer: marker, stripped; otherwise stripped whole response",
                "reward": "float(reasoning_gym.get_score_answer_fn(task)(answer, entry))",
                "recorded_pinned_generator_controls": controls.model_dump(mode="json") if controls is not None else None,
            },
            runtime_requirements=(f"Native reasoning-gym scorer at generator revision {REVISION}",),
        )
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        return ImportRejection(reason="invalid_generated_reasoning_entry", detail=str(error))
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=context,
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=VerifierSpec(kind=VerifierKind.SOURCE_CONTRACT, parameters_json=verifier.model_dump_json()),
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    verifier = SourceContractVerifier.model_validate_json(task.verifier.parameters_json)
    controls = verifier.contract.get("recorded_pinned_generator_controls")
    checks = verify_task(task)
    if controls is None:
        return VerificationReport(checks=checks)
    controls = RecordedControls.model_validate(controls)
    if controls.generator_revision != REVISION:
        checks.append(
            CheckResult(
                check="recorded_pinned_generator_controls",
                status=CheckStatus.FAIL,
                detail="Recorded native controls do not identify the pinned generator revision",
            )
        )
        return VerificationReport(checks=checks)
    passed = controls.positive.reward == 1.0 and controls.negative.reward < controls.positive.reward
    checks.append(
        CheckResult(
            check="recorded_pinned_generator_controls",
            status=CheckStatus.PASS if passed else CheckStatus.FAIL,
            detail="Recorded acquisition-time native controls, not current bound runtime: " + controls.model_dump_json(),
        )
    )
    return VerificationReport(checks=checks)


def recipe() -> DatasetRecipe:
    return DatasetRecipe(
        name="reasoning_gym_generated",
        version="reasoning-gym-direct-v1",
        source=HFSource(DATASET, REVISION, "generated", "generated"),
        inputs=RecipeInputs(
            files=SourceFiles(("generator.tar.gz",), SourceFormat.GENERATED),
            downloads=(UrlDownload(f"https://api.github.com/repos/{DATASET}/tarball/{REVISION}", "generator.tar.gz"),),
        ),
        normalize=normalize,
        rubric=RUBRIC,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(
            id="recorded-pinned-generator-controls", revision="1", parameters={}, run=verification_report
        ),
    )
