# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-population Harbor oracle receipts bound to the converted task contents."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec, decode_json_value
from taskcompendium.pipeline.fingerprints import recipe_code_identity
from taskcompendium.pipeline.models import SourceRecipe
from taskcompendium.pipeline.sources import source_files_identity


def has_oracle_solution(task: TaskSpec) -> bool:
    resources = task.resources
    return any(
        resource.path.startswith("solution/")
        for resource in (*resources.all, *resources.worker, *resources.verifier, *resources.oracle)
    )


def oracle_recipe_identity(recipe: SourceRecipe) -> str:
    return canonical_sha256(
        {
            "name": recipe.name,
            "version": recipe.version,
            "source": source_files_identity(recipe.source),
            "code": recipe_code_identity(recipe),
            "grader": recipe.grader_environment.model_dump(mode="json") if recipe.grader_environment else None,
            "oracle_solutions": recipe.oracle_solutions,
        }
    )


class OracleTrialReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    path: str
    result: dict[str, Any]
    verdict: dict[str, Any] | None
    oracle_exit_code: int | None
    result_sha256: str

    @model_validator(mode="after")
    def validate_result_hash(self):
        if canonical_sha256(self.result) != self.result_sha256:
            raise ValueError("Oracle trial result hash differs from its receipt")
        return self

    def passed(self, task_checksum: str) -> bool:
        result = self.result
        verdict = self.verdict or {}
        reward = verdict.get("reward")
        return (
            result.get("agent_info", {}).get("name") == "oracle"
            and result.get("config", {}).get("agent", {}).get("name") == "oracle"
            and result.get("config", {}).get("environment", {}).get("type") in {"local", "daytona"}
            and result.get("task_checksum") == task_checksum
            and result.get("exception_info") is None
            and result.get("finished_at") is not None
            and result.get("completion") is not None
            and self.oracle_exit_code == 0
            and verdict.get("status") == "scored"
            and not isinstance(reward, bool)
            and reward == 1
            and ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward") == 1
        )


class OracleTaskReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_sha256: str
    runtime_sha256: str | None = None
    harbor_task_checksum: str
    has_oracle: bool
    trials: tuple[OracleTrialReceipt, ...]
    exclusion_reason: str | None = None

    @property
    def passed(self) -> bool:
        return not self.has_oracle or any(trial.passed(self.harbor_task_checksum) for trial in self.trials)


class OracleFilterReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1] = 1
    recipe_identity: str | None
    input_cohort_sha256: str
    input_rows: int = Field(ge=0)
    conversion_rejections: int = Field(ge=0)
    tasks: dict[str, OracleTaskReceipt]

    @model_validator(mode="after")
    def validate_population(self):
        if len(self.tasks) + self.conversion_rejections != self.input_rows:
            raise ValueError("Oracle filter receipt does not cover the full input population")
        for task_id, entry in self.tasks.items():
            if entry.has_oracle and not entry.passed and not entry.exclusion_reason:
                raise ValueError(f"Oracle exclusion has no reason: {task_id}")
            if not entry.has_oracle and entry.trials:
                raise ValueError(f"Non-oracle task carries oracle trials: {task_id}")
        return self

    def admits(self, task: TaskSpec, runtime: dict | None = None) -> bool:
        if task.id not in self.tasks:
            raise ValueError(f"Oracle filter receipt omits task {task.id}")
        entry = self.tasks[task.id]
        if entry.task_sha256 != canonical_sha256(task.model_dump(mode="json")):
            raise ValueError(f"Task changed after oracle filtering: {task.id}")
        if runtime is not None and entry.runtime_sha256 != canonical_sha256(runtime):
            raise ValueError(f"Runtime changed after oracle filtering: {task.id}")
        if entry.has_oracle != has_oracle_solution(task):
            raise ValueError(f"Oracle presence changed after filtering: {task.id}")
        return entry.passed


def load_oracle_filter(path: str) -> OracleFilterReceipt:
    return OracleFilterReceipt.model_validate(decode_json_value(StoragePath(path).read_text()))


def require_oracle_filter(task: TaskSpec, receipt: OracleFilterReceipt | None, runtime: dict | None = None) -> bool:
    if not has_oracle_solution(task):
        return True
    if receipt is None:
        raise ValueError(f"SAMPLE/FULL admission requires a full-population Harbor oracle filter receipt: {task.id}")
    return receipt.admits(task, runtime)
