# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the requirement-backfill static rules over one retained Harbor-archive population.

Issue #11976 step 2 pilot: measure the static rules' per-capability outcomes, the
ambiguous residual, and the hand-check sample on one frozen converted population,
entirely in region. Reads the population artifact the actor census measured
(``specs/*.parquet`` with ``(id, task_spec, runtime)`` rows), pins it by its
``task_specs_sha256``, and writes the receipt, the ambiguous-task set and the
hand-check sheet under the pilot's atlas prefix. The population never leaves the
region; only the bounded sample sheet is pulled for human review.
"""

import argparse
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.requirements import (
    REQUIREMENTS_REVISION,
    RULE_CAPABILITIES,
    RULES,
    BackfillPolicy,
    TaskBackfill,
    backfill_task,
    is_command_file,
)
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.lexical_contamination import RENDERER_VERSION

LOGGER = logging.getLogger(__name__)

PILOT_POLICY_REVISION = "harbor-v2"
PILOT_RESIDUAL_THRESHOLD = 0.3
HAND_CHECK_SIZE = 100
HAND_CHECK_SEED = 0
HAND_CHECK_FILE_TEXT_BYTES = 8192
HAND_CHECK_MAX_FILE_BYTES = 262144
OUTCOMES = ("declared", "unchanged", "ambiguous", "not_stated")


@dataclass(frozen=True)
class TaskSummary:
    """The compact static-pass outcome of one task, without retaining the task's bytes."""

    task_id: str
    summary: TaskBackfill
    capabilities_before: tuple[str, ...]


def spec_rows(population: str) -> Iterator[tuple[str, str]]:
    """Yield (task id, serialized spec) in stored order across the population's spec shards."""
    shards = sorted(str(path) for path in (StoragePath(population) / "specs/*.parquet").glob())
    if not shards:
        raise ValueError(f"The frozen population holds no spec shards: {population}")
    for shard in shards:
        with StoragePath(shard).open("rb") as stream:
            table = pq.read_table(stream, columns=["id", "task_spec"])
        ids = table.column("id").to_pylist()
        specs = table.column("task_spec").to_pylist()
        yield from zip(ids, specs, strict=True)


def verify_population(population: str, expected_sha256: str) -> int:
    """Recompute the population's serialized-spec digest in stored order and return the task count."""
    digest = hashlib.sha256()
    count = 0
    for _, encoded in spec_rows(population):
        digest.update(encoded.encode())
        count += 1
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"The frozen population's task specs changed: {actual} != {expected_sha256}")
    return count


def sheet_files(task: TaskSpec) -> list[dict[str, Any]]:
    """The task's command files with bounded text, for hand-labeling ground truth."""
    files = []
    for resource in task.resources.all + task.resources.worker + task.resources.oracle + task.resources.verifier:
        if not is_command_file(resource.path):
            continue
        data = resource_bytes(resource)
        text = data.decode("utf-8", errors="replace") if len(data) <= HAND_CHECK_MAX_FILE_BYTES else ""
        files.append(
            {
                "path": resource.path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "byte_count": len(data),
                "text": text[:HAND_CHECK_FILE_TEXT_BYTES],
                "text_truncated": len(text) > HAND_CHECK_FILE_TEXT_BYTES,
            }
        )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--population", required=True, help="The frozen population artifact (specs/*.parquet)")
    parser.add_argument("--output", required=True, help="The pilot output prefix, in region")
    parser.add_argument("--expected-task-specs-sha256", required=True)
    parser.add_argument("--task-count", type=int, required=True, help="The frozen population's declared task count")
    parser.add_argument("--policy-revision", default=PILOT_POLICY_REVISION)
    parser.add_argument("--residual-threshold", type=float, default=PILOT_RESIDUAL_THRESHOLD)
    parser.add_argument("--hand-check-size", type=int, default=HAND_CHECK_SIZE)
    parser.add_argument("--hand-check-seed", type=int, default=HAND_CHECK_SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    configure_coreweave_s3()

    count = verify_population(args.population, args.expected_task_specs_sha256)
    if count != args.task_count:
        raise ValueError(f"The frozen population holds {count} tasks, not the declared {args.task_count}")
    LOGGER.info("Pinned the frozen population: %d tasks, task_specs_sha256 %s", count, args.expected_task_specs_sha256)

    policy = BackfillPolicy(args.policy_revision, args.residual_threshold)
    counts: Counter[str] = Counter()
    ambiguous: list[dict[str, str]] = []
    summaries: list[TaskSummary] = []

    def summarized_rows() -> Iterator[dict[str, Any]]:
        for task_id, encoded in spec_rows(args.population):
            task = TaskSpec.model_validate_json(encoded)
            counts["task:total"] += 1
            counts["task:normalized"] += 1
            result = backfill_task(task, policy)
            counts.update(result.rule_hits)
            if result.machine_less:
                counts["task:machine_less"] += 1
            else:
                for capability, outcome in result.outcomes.items():
                    counts[f"cap:{capability}:{outcome}"] += 1
                counts[f"task:{result.outcome}"] += 1
            if result.ambiguous:
                ambiguous.append({"task_id": task_id})
                counts["task:ambiguous_set"] += 1
            summaries.append(TaskSummary(task_id, result, task.environment_requirements.capabilities))
            yield {"task_id": task_id}

    rows = list(summarized_rows())
    if len(rows) != count or len(summaries) != count:
        raise ValueError(f"The static pass covered {len(summaries)} of {count} tasks")

    panel_count, panel = seeded_sample(
        summaries, size=args.hand_check_size, key=lambda row: seeded_order(row.task_id, args.hand_check_seed)
    )
    if panel_count != count:
        raise ValueError(f"The sample covered {panel_count} of {count} tasks")
    sampled_ids = {row.task_id for row in panel}
    sampled: list[dict[str, Any]] = []
    for row in panel:
        result = row.summary
        sampled.append(
            {
                "task_id": row.task_id,
                "outcome": result.outcome,
                "outcomes": result.outcomes,
                "evidence": result.written,
                "rule_hits": dict(result.rule_hits),
                "capabilities_before": list(row.capabilities_before),
                "files": [],
            }
        )
    # Second pass over the specs fetches only the sampled tasks' files, for hand-labeling.
    filled = 0
    for task_id, encoded in spec_rows(args.population):
        if task_id not in sampled_ids:
            continue
        task = TaskSpec.model_validate_json(encoded)
        for entry in sampled:
            if entry["task_id"] == task_id:
                entry["files"] = sheet_files(task)
                entry["command_semantics"] = task.environment_requirements.command_semantics
                filled += 1
        if filled == len(sampled_ids):
            break
    if filled != len(sampled_ids):
        raise ValueError(f"The sample sheet lacks files for {len(sampled_ids) - filled} tasks")

    normalized = counts["task:normalized"]
    machine_ful = normalized - counts["task:machine_less"]
    ambiguous_count = counts["task:ambiguous_set"]
    share = ambiguous_count / machine_ful if machine_ful else 0.0
    receipt = {
        "source_id": args.source_id,
        "population": args.population,
        "task_specs_sha256": args.expected_task_specs_sha256,
        "revision": REQUIREMENTS_REVISION,
        "renderer_version": RENDERER_VERSION,
        "policy": {
            "rules_revision": policy.rules_revision,
            "residual_threshold": policy.residual_threshold,
            "hand_check_size": args.hand_check_size,
            "hand_check_seed": args.hand_check_seed,
        },
        "counts": {
            "tasks": {
                "total": counts["task:total"],
                "normalized": normalized,
                "machine_less": counts["task:machine_less"],
            },
            "outcomes": {outcome: counts[f"task:{outcome}"] for outcome in OUTCOMES},
            "capabilities": {
                capability: {outcome: counts[f"cap:{capability}:{outcome}"] for outcome in OUTCOMES}
                for capability in RULE_CAPABILITIES
            },
            "rules": {rule.name: counts[f"rule:{rule.name}"] for rule in RULES},
        },
        "residual": {
            "ambiguous_count": ambiguous_count,
            "machine_ful": machine_ful,
            "ambiguous_share": share,
            "over_threshold": share > policy.residual_threshold,
        },
        "hand_check": {"size": len(sampled), "seed": args.hand_check_seed},
    }
    output = StoragePath(args.output)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "ambiguous.json").write_text(json.dumps(ambiguous, indent=1) + "\n")
    (output / "hand-check-sample.json").write_text(json.dumps(sampled, indent=1) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
