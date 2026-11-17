import json
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from harbor.models.trial.config import TrialConfig


class TrialPlanMismatchKind(StrEnum):
    """Reason an existing trial has no slot in a planned job."""

    EXCESS_EXISTING_ATTEMPT = "excess_existing_attempt"
    SEMANTIC_CONFIG_MISMATCH = "semantic_config_mismatch"
    TASK_NOT_IN_PLAN = "task_not_in_plan"


@dataclass(frozen=True)
class TrialPlanMismatch:
    """One existing trial that cannot be reused by the planned job."""

    trial_name: str
    task_name: str
    kind: TrialPlanMismatchKind
    differing_fields: tuple[str, ...] = ()


class TrialPlanMismatchError(ValueError):
    """Raised when preserved trials are incompatible with a planned job."""

    def __init__(
        self,
        mismatches: list[TrialPlanMismatch],
        *,
        compatible_existing_count: int,
    ) -> None:
        self.mismatches = tuple(mismatches)
        self.compatible_existing_count = compatible_existing_count

        details = []
        for mismatch in self.mismatches:
            fields = (
                f"; differing_fields={','.join(mismatch.differing_fields)}"
                if mismatch.differing_fields
                else ""
            )
            details.append(
                f"- {mismatch.trial_name}: task={mismatch.task_name}; "
                f"kind={mismatch.kind.value}{fields}"
            )

        message = (
            "Existing trial config does not match planned job config: "
            f"{len(self.mismatches)} on-disk trial(s) are incompatible; "
            f"{compatible_existing_count} existing trial(s) matched safely.\n"
            "Mismatch details:\n"
            + "\n".join(details)
            + "\nThis is a SEMANTIC mismatch; Harbor will not combine the plans. "
            "Preserve the existing job directory. Resume it only with the "
            "original task plan. To migrate compatible results deliberately, "
            "archive (do not delete) only the listed trial directories outside "
            "the job root, then rerun planning. Otherwise use a new job directory "
            "for the changed plan."
        )
        super().__init__(message)


def _canonical_config_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def trial_identity_key(config: TrialConfig) -> str:
    """Return the canonical key used by resume identity matching."""
    return _canonical_config_key(config.identity_fingerprint())


def _task_identity_key(config: TrialConfig) -> str:
    return _canonical_config_key(config.identity_fingerprint()["task"])


def _differing_config_fields(
    existing: object,
    planned: object,
    *,
    prefix: str = "",
) -> tuple[str, ...]:
    """Return dotted paths that differ, or an empty tuple for equal values.

    A scalar mismatch at the top level is represented by an empty-string path.
    """
    if existing == planned:
        return ()
    if not isinstance(existing, dict) or not isinstance(planned, dict):
        return (prefix,)

    differing: list[str] = []
    for key in sorted(existing.keys() | planned.keys()):
        field = f"{prefix}.{key}" if prefix else str(key)
        if key not in existing or key not in planned:
            differing.append(field)
            continue
        differing.extend(
            _differing_config_fields(existing[key], planned[key], prefix=field)
        )
    return tuple(differing)


def classify_trial_plan_mismatches(
    unmatched: list[TrialConfig],
    planned: list[TrialConfig],
) -> list[TrialPlanMismatch]:
    """Classify preserved trials that have no slot in the planned job."""
    planned_identity_keys = {trial_identity_key(config) for config in planned}
    planned_by_task: dict[str, list[TrialConfig]] = defaultdict(list)
    for config in planned:
        planned_by_task[_task_identity_key(config)].append(config)

    mismatches: list[TrialPlanMismatch] = []
    for existing in unmatched:
        identity_key = trial_identity_key(existing)
        task_candidates = planned_by_task.get(_task_identity_key(existing), [])
        differing_fields: tuple[str, ...] = ()
        if identity_key in planned_identity_keys:
            kind = TrialPlanMismatchKind.EXCESS_EXISTING_ATTEMPT
        elif task_candidates:
            kind = TrialPlanMismatchKind.SEMANTIC_CONFIG_MISMATCH
            existing_identity = existing.identity_fingerprint()
            candidate_diffs = [
                _differing_config_fields(
                    existing_identity,
                    candidate.identity_fingerprint(),
                )
                for candidate in task_candidates
            ]
            differing_fields = min(
                candidate_diffs, key=lambda fields: (len(fields), fields)
            )
        else:
            kind = TrialPlanMismatchKind.TASK_NOT_IN_PLAN

        mismatches.append(
            TrialPlanMismatch(
                trial_name=existing.trial_name,
                task_name=existing.task.get_task_id().get_name(),
                kind=kind,
                differing_fields=differing_fields,
            )
        )
    return mismatches
