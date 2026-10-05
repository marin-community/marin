# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a sealed replay schedule for later Russell RSI pilots."""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.russell_rsi.bootstrap_loop import (
    ATTEMPTS_PER_TASK,
    MAX_GLM_RESPONSES,
    MAX_PILOTS,
    PILOT_UPDATES,
    Measurement,
    QualifiedTask,
    RoundPlan,
)
from experiments.post_training.russell_rsi.repair_tasks import canonical_sha256
from experiments.post_training.russell_rsi.sources import compact_json_sha256

REPLAY_SEED = 9528
GROUPS_PER_UPDATE = 16
REPLAY_GROUPS = 12
TARGETED_GROUPS = 2
EXPLORATION_GROUPS = 2
ROLLOUTS_PER_GROUP = 4
REQUIRED_WEIGHTED_Q4 = 0.5
REPLAY_DATA_SHUFFLE = False
REPLAY_EPOCHS = 1
REPLAY_BATCH_POLICY = "full_batch"
REPLAY_MAX_STALENESS_STEPS = 0
REPLAY_ROWS = PILOT_UPDATES * GROUPS_PER_UPDATE


@dataclass(frozen=True)
class ReplayOccurrence:
    row_index: int
    occurrence_id: str
    update: int
    group: int
    category: str
    task_id: str
    task_sha256: str
    admission_sha256: str
    source_id: str
    contract_id: str
    q4: float


def mixed_group_probability(successes: int) -> float:
    """Calculate the chance that four draws include reward variation."""
    if not 0 <= successes <= ATTEMPTS_PER_TASK:
        raise ValueError(f"Calibration success count must be between zero and {ATTEMPTS_PER_TASK}")

    def choose_four(value: int) -> int:
        return math.comb(value, ROLLOUTS_PER_GROUP) if value >= ROLLOUTS_PER_GROUP else 0

    return 1 - (choose_four(successes) + choose_four(ATTEMPTS_PER_TASK - successes)) / math.comb(
        ATTEMPTS_PER_TASK, ROLLOUTS_PER_GROUP
    )


def calibration_signal_failure(measurements: tuple[Measurement, ...]) -> str | None:
    """Return why the binary calibration cannot support replay sampling."""
    if any(reward not in (0, 1, 0.0, 1.0) for item in measurements for reward in item.rewards):
        return "calibration_rewards_not_binary"
    if not any(len(set(item.rewards)) > 1 for item in measurements):
        return "no_calibration_reward_variation"
    return None


def _family_groups(
    tasks: tuple[QualifiedTask, ...], family_by_task: dict[str, str]
) -> tuple[tuple[QualifiedTask, ...], ...]:
    families = defaultdict(list)
    for task in tasks:
        families[family_by_task[task.task_id]].append(task)
    return tuple(tuple(families[key]) for key in sorted(families))


def replay_plan(
    plan: RoundPlan,
    measurements: tuple[Measurement, ...],
    targeted_tasks: tuple[QualifiedTask, ...],
    *,
    pilot_number: int,
    bank_identity: str,
    calibration_identity: str,
    frozen_identity: str,
    parent_identity: str,
    model_identity: str,
    family_by_task: dict[str, str] | None = None,
) -> dict:
    """Seal four 12/2/2 schedules and their calibration signal estimate."""
    if pilot_number < 2 or pilot_number > MAX_PILOTS:
        raise ValueError("Replay applies only to pilots two and three")
    measured = {item.task_sha256: item for item in measurements}
    if len(measured) != len(measurements) or set(measured) != {task.task_sha256 for task in plan.task_bank}:
        raise ValueError("Replay calibration must measure the complete qualified bank")
    if any(reward not in (0, 1, 0.0, 1.0) for item in measurements for reward in item.rewards):
        raise ValueError("Replay q4 requires binary calibration rewards")
    if any(len(item.rewards) != ATTEMPTS_PER_TASK for item in measurements):
        raise ValueError("Replay q4 requires eight calibration attempts per task")
    successes = {key: sum(reward > 0 for reward in item.rewards) for key, item in measured.items()}
    q4 = {key: mixed_group_probability(count) for key, count in successes.items()}
    replay_pool = tuple(task for task in plan.task_bank if 1 <= successes[task.task_sha256] <= ATTEMPTS_PER_TASK - 1)
    targeted = tuple(targeted_tasks)
    if len({task.contract_id for task in targeted}) != len(targeted):
        raise ValueError("Targeted replay tasks require distinct independent contracts")
    if any(task not in plan.task_bank for task in targeted):
        raise ValueError("Targeted replay tasks must belong to the calibrated bank")
    if not replay_pool or not targeted:
        raise ValueError("Replay requires eligible replay tasks and independent targeted additions")
    if any(task.relation in {"variant", "replacement", "alias"} for task in targeted):
        raise ValueError("Targeted replay tasks must be independent contract additions")
    family_groups = {}
    probabilities = {}
    if family_by_task is not None:
        if set(family_by_task) != {task.task_id for task in plan.task_bank}:
            raise ValueError("Family sampling must identify every calibrated task")
        roots = {task.contract_id for task in plan.task_bank if task.relation not in {"variant", "replacement", "alias"}}
        if not set(family_by_task.values()) <= roots or any(
            family_by_task[task.task_id] != task.contract_id
            for task in plan.task_bank
            if task.relation not in {"variant", "replacement", "alias"}
        ):
            raise ValueError("Task families must identify their original independent contracts")
        if len({family_by_task[task.task_id] for task in targeted}) != len(targeted):
            raise ValueError("Targeted additions must belong to distinct independent families")
        family_groups = {
            "replay": _family_groups(replay_pool, family_by_task),
            "exploration": _family_groups(plan.task_bank, family_by_task),
        }
        probabilities = {
            category: {task.task_id: 1 / len(groups) / len(family) for family in groups for task in family}
            for category, groups in family_groups.items()
        }
        probabilities["targeted"] = {task.task_id: 1 / len(targeted) for task in targeted}
    elif any(task.relation in {"variant", "replacement", "alias"} for task in plan.task_bank):
        raise ValueError("A bank with variants requires an explicit original family mapping")

    family_expected_q4 = None
    if family_by_task is not None:
        q4_by_id = {task.task_id: q4[task.task_sha256] for task in plan.task_bank}
        family_expected_q4 = (
            sum(
                allocation
                * sum(probability * q4_by_id[task_id] for task_id, probability in probabilities[category].items())
                for category, allocation in (
                    ("replay", REPLAY_GROUPS),
                    ("targeted", TARGETED_GROUPS),
                    ("exploration", EXPLORATION_GROUPS),
                )
            )
            / GROUPS_PER_UPDATE
        )

    rng = random.Random(REPLAY_SEED)
    occurrences: list[ReplayOccurrence] = []
    update_q4: list[float] = []
    for update in range(1, PILOT_UPDATES + 1):
        selected_targeted = [targeted[0], targeted[0]] if len(targeted) == 1 else rng.sample(targeted, TARGETED_GROUPS)
        if family_by_task is not None:
            groups = [
                *(("replay", rng.choice(rng.choice(family_groups["replay"]))) for _ in range(REPLAY_GROUPS)),
                *(("targeted", task) for task in selected_targeted),
                *(
                    ("exploration", rng.choice(rng.choice(family_groups["exploration"])))
                    for _ in range(EXPLORATION_GROUPS)
                ),
            ]
        else:
            groups = [
                *(("replay", rng.choice(replay_pool)) for _ in range(REPLAY_GROUPS)),
                *(("targeted", task) for task in selected_targeted),
                *(("exploration", rng.choice(plan.task_bank)) for _ in range(EXPLORATION_GROUPS)),
            ]
        for group, (category, task) in enumerate(groups, start=1):
            occurrences.append(
                ReplayOccurrence(
                    row_index=len(occurrences),
                    occurrence_id=f"pilot-{pilot_number}-update-{update}-group-{group}",
                    update=update,
                    group=group,
                    category=category,
                    task_id=task.task_id,
                    task_sha256=task.task_sha256,
                    admission_sha256=task.admission_sha256,
                    source_id=task.source_id,
                    contract_id=task.contract_id,
                    q4=q4[task.task_sha256],
                )
            )
        if family_expected_q4 is not None:
            update_q4.append(family_expected_q4)
        else:
            # Preserve the estimate and random draw order of sealed v1 cohorts.
            update_q4.append(
                (
                    REPLAY_GROUPS * sum(q4[task.task_sha256] for task in replay_pool) / len(replay_pool)
                    + sum(q4[task.task_sha256] for task in selected_targeted)
                    + EXPLORATION_GROUPS * sum(q4.values()) / len(plan.task_bank)
                )
                / GROUPS_PER_UPDATE
            )

    counts = Counter((entry.task_id, entry.category) for entry in occurrences)
    sampling_spec = {
        "seed": REPLAY_SEED,
        "updates": PILOT_UPDATES,
        "groups_per_update": GROUPS_PER_UPDATE,
        "rollouts_per_group": ROLLOUTS_PER_GROUP,
        "allocation": {"replay": REPLAY_GROUPS, "targeted": TARGETED_GROUPS, "exploration": EXPLORATION_GROUPS},
        "replay_success_range": [1, ATTEMPTS_PER_TASK - 1],
        "q4_formula": f"1 - (C(k,4) + C({ATTEMPTS_PER_TASK}-k,4)) / C({ATTEMPTS_PER_TASK},4), " "with C(n,4)=0 for n<4",
        "data_shuffle": REPLAY_DATA_SHUFFLE,
        "epochs": REPLAY_EPOCHS,
        "batch_policy": REPLAY_BATCH_POLICY,
        "max_staleness_steps": REPLAY_MAX_STALENESS_STEPS,
        "signal_gate_weighted_q4": REQUIRED_WEIGHTED_Q4,
    }
    if family_by_task is not None:
        sampling_spec["selection"] = "uniform_original_family_then_uniform_eligible_member"
        sampling_spec["task_probabilities"] = probabilities
        sampling_spec["targeted_task_ids"] = [task.task_id for task in targeted]
    payload = {
        "protocol": "pilot2-calibrated-replay-v1",
        "pilot_number": pilot_number,
        "parent_identity": parent_identity,
        "model_identity": model_identity,
        "bank_identity": bank_identity,
        "calibration_identity": calibration_identity,
        "frozen_identity": frozen_identity,
        "round_plan_sha256": compact_json_sha256(asdict(plan)),
        "task_support": [asdict(task) for task in plan.task_bank],
        "sampling_spec": sampling_spec,
        "sampling_spec_sha256": compact_json_sha256(sampling_spec),
        "successes_by_task": {task.task_id: successes[task.task_sha256] for task in plan.task_bank},
        "q4_by_task": {task.task_id: q4[task.task_sha256] for task in plan.task_bank},
        "schedule": [asdict(entry) for entry in occurrences],
        "per_task_counts": {
            task_id: {
                "total": sum(counts[(task_id, category)] for category in ("replay", "targeted", "exploration")),
                **{category: counts[(task_id, category)] for category in ("replay", "targeted", "exploration")},
            }
            for task_id in sorted(task.task_id for task in plan.task_bank)
        },
        "weighted_q4_by_update": update_q4,
        "weighted_mean_q4": sum(update_q4) / len(update_q4),
        "expected_mixed_groups_per_update": GROUPS_PER_UPDATE * sum(update_q4) / len(update_q4),
        "expected_mixed_groups_for_pilot": GROUPS_PER_UPDATE * sum(update_q4),
        "signal_gate_passed": sum(update_q4) / len(update_q4) >= REQUIRED_WEIGHTED_Q4,
        "signal_gate_interpretation": "A predeclared compute allocation target, not a confidence bound.",
        "expected_signal_note": "A calibration estimate with sampling uncertainty, not a forecast or causal claim.",
        "experiment_limits": {
            "maximum_additional_pilots": MAX_PILOTS - 1,
            "maximum_additional_updates": (MAX_PILOTS - 1) * PILOT_UPDATES,
            "max_glm_responses_per_round": MAX_GLM_RESPONSES,
        },
        "limits": [
            "Replay reduces task diversity and does not create independent contracts.",
            "Calibration selection can overstate future informative-group rates.",
            "This curriculum amendment does not support a causal comparison with pilot one.",
        ],
    }
    if family_by_task is not None:
        payload["protocol"] = "pilot2-calibrated-family-replay-v2"
        payload["family_by_task"] = dict(sorted(family_by_task.items()))
    payload["schedule_sha256"] = compact_json_sha256(payload)
    return payload


def validate_replay_plan(
    sealed_plan: dict,
    round_plan: RoundPlan,
    *,
    pilot_number: int,
    bank_identity: str,
    calibration_identity: str,
    frozen_identity: str,
    parent_identity: str,
    model_identity: str,
    family_by_task: dict[str, str] | None = None,
) -> None:
    """Verify that a resumed schedule still identifies its frozen inputs."""
    value = dict(sealed_plan)
    schedule_sha256 = value.pop("schedule_sha256")
    if schedule_sha256 != compact_json_sha256(value):
        raise ValueError("Replay schedule digest mismatch")
    expected = {
        "pilot_number": pilot_number,
        "bank_identity": bank_identity,
        "calibration_identity": calibration_identity,
        "frozen_identity": frozen_identity,
        "parent_identity": parent_identity,
        "model_identity": model_identity,
        "round_plan_sha256": compact_json_sha256(asdict(round_plan)),
    }
    if any(sealed_plan[key] != expected_value for key, expected_value in expected.items()):
        raise ValueError("Resumed replay schedule input identity changed")
    if family_by_task is None:
        if "family_by_task" in sealed_plan:
            raise ValueError("Resumed family replay requires the declared family mapping")
        return
    if sealed_plan.get("family_by_task") != family_by_task:
        raise ValueError("Resumed replay task families changed")
    targeted_ids = sealed_plan["sampling_spec"]["targeted_task_ids"]
    by_id = {task.task_id: task for task in round_plan.task_bank}
    reconstructed = replay_plan(
        round_plan,
        tuple(
            Measurement(
                model_identity,
                task.task_sha256,
                (1.0,) * sealed_plan["successes_by_task"][task.task_id]
                + (0.0,) * (ATTEMPTS_PER_TASK - sealed_plan["successes_by_task"][task.task_id]),
            )
            for task in round_plan.task_bank
        ),
        tuple(by_id[task_id] for task_id in targeted_ids),
        **{key: item for key, item in expected.items() if key != "round_plan_sha256"},
        family_by_task=family_by_task,
    )
    if reconstructed != sealed_plan:
        raise ValueError("Resumed family replay differs from its declared sampling protocol")


@dataclass(frozen=True)
class ReplayDatasetConfig:
    bank_path: str
    output_path: str
    tasks: tuple[QualifiedTask, ...]
    replay_plan: dict


def freeze_replay_dataset(config: ReplayDatasetConfig) -> None:
    """Write unchanged task rows in sealed schedule order, including replay duplicates."""
    from taskcompendium.parquet import read_tasks, write_tasks  # noqa: PLC0415

    plan_value = dict(config.replay_plan)
    plan_sha256 = plan_value.pop("schedule_sha256")
    if plan_sha256 != compact_json_sha256(plan_value):
        raise ValueError("Replay schedule digest mismatch")
    rows = list(read_tasks(prefix_join(config.bank_path, "train.parquet")))
    by_id = {task.id: task for task in rows}
    if len(by_id) != len(rows):
        raise ValueError("The qualified bank contains duplicate task IDs")
    evidence_by_id = {task.task_id: task for task in config.tasks}
    output = []
    families = config.replay_plan.get("family_by_task")
    for row_index, entry in enumerate(config.replay_plan["schedule"]):
        if entry["row_index"] != row_index:
            raise ValueError("Replay occurrence UID differs from its row index")
        task_record = evidence_by_id.get(entry["task_id"])
        task = by_id.get(entry["task_id"])
        if task_record is None or task is None:
            raise ValueError("Replay schedule task is absent from the qualified bank")
        if task_record.task_sha256 != entry["task_sha256"] or task_record.admission_sha256 != entry["admission_sha256"]:
            raise ValueError("Replay schedule identity differs from the qualified bank")
        if canonical_sha256(task.model_dump(mode="json")) != task_record.task_sha256:
            raise ValueError("Replay TaskSpec content differs from its sealed bank")
        admission = StoragePath(
            prefix_join(config.bank_path, f"evidence/{task_record.admission_sha256}/proposal.json")
        ).read_bytes()
        if hashlib.sha256(admission).hexdigest() != task_record.admission_sha256:
            raise ValueError("Replay admission report differs from its sealed bank")
        report = json.loads(admission)
        if report["task_sha256"] != task_record.task_sha256 or report["source_group"] != task_record.source_id:
            raise ValueError("Replay task differs from its sealed admission evidence")
        if families is not None and (
            families[task.id] != report.get("original_family_id", task_record.contract_id)
            or report.get("relation", task_record.relation) != task_record.relation
        ):
            raise ValueError("Replay task family differs from its sealed admission evidence")
        output.append(task)
    occurrences = config.replay_plan["schedule"]
    if len(output) != REPLAY_ROWS or len({entry["occurrence_id"] for entry in occurrences}) != REPLAY_ROWS:
        raise ValueError("Replay artifact requires sixty-four distinct occurrences")
    write_tasks(prefix_join(config.output_path, "train.parquet"), output)
    StoragePath(prefix_join(config.output_path, "replay-plan.json")).write_text(
        json.dumps(config.replay_plan, sort_keys=True, indent=2) + "\n"
    )
