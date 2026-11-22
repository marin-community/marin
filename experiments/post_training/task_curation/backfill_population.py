# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Produce a requirement-backfilled population from one frozen retained population.

Issue #11976 step 3: read the frozen population the actor census measured, apply the
static requirement rules under one policy revision, and write the backfilled tasks as a
standard text population through the shared exact-deduplication writer, so every later
gate (census, lexical, oracle) consumes it like any intake population. Runs entirely in
region; the parent population never moves.
"""

import argparse
import hashlib
import json
import logging
from collections import Counter
from typing import Any

import pyarrow.parquet as pq
from marin.execution.artifact import write_artifact
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import task_spec_json
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.requirements import (
    LABEL_KEY,
    REQUIREMENTS_REVISION,
    RULE_CAPABILITIES,
    RULES,
    BackfillPolicy,
    backfill_task,
)

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import (
    RENDERER_VERSION,
    write_text_population,
)

LOGGER = logging.getLogger(__name__)

OUTCOMES = ("declared", "unchanged", "ambiguous", "not_stated")


def _binding(old: TaskSpec, old_encoded: str, result: Any) -> dict[str, Any]:
    """Prove the backfilled spec differs from the parent only in declared requirements and labels.

    The carry-forward receipt binds every parent task hash to its backfilled hash; the diff
    check rejects any change beyond the added capabilities and the evidence label, so prior
    oracle and QC verdicts carry forward without re-running them.
    """
    new = result.task
    old_dump = old.model_dump(mode="json")
    new_dump = new.model_dump(mode="json")
    old_environment = old_dump.pop("environment_requirements")
    new_environment = new_dump.pop("environment_requirements")
    old_labels = old_dump.pop("labels")
    new_labels = new_dump.pop("labels")
    if old_dump != new_dump:
        raise ValueError(f"Backfill changed fields beyond requirements and labels: {old.id}")
    added = tuple(
        capability for capability in new_environment["capabilities"] if capability not in old_environment["capabilities"]
    )
    if new_environment != {**old_environment, "capabilities": new_environment["capabilities"]}:
        raise ValueError(f"Backfill changed requirement fields other than capabilities: {old.id}")
    if set(new_labels) - set(old_labels) != {LABEL_KEY} or new_labels != {
        **old_labels,
        LABEL_KEY: new_labels.get(LABEL_KEY),
    }:
        raise ValueError(f"Backfill changed labels beyond the evidence key: {old.id}")
    return {
        "task_id": old.id,
        "parent_task_sha256": hashlib.sha256(old_encoded.encode()).hexdigest(),
        "backfilled_task_sha256": hashlib.sha256(task_spec_json(new).encode()).hexdigest(),
        "capabilities_added": list(added),
        "outcome": result.outcome,
    }


def spec_rows(population: str) -> list[tuple[str, str, str]]:
    """Every (task id, serialized spec, runtime) in stored order across the population's spec shards."""
    rows = []
    shards = sorted(str(path) for path in (StoragePath(population) / "specs/*.parquet").glob())
    if not shards:
        raise ValueError(f"The frozen population holds no spec shards: {population}")
    for shard in shards:
        with StoragePath(shard).open("rb") as stream:
            table = pq.read_table(stream, columns=["id", "task_spec", "runtime"])
        ids = table.column("id").to_pylist()
        specs = table.column("task_spec").to_pylist()
        runtimes = table.column("runtime").to_pylist()
        rows.extend(zip(ids, specs, runtimes, strict=True))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--parent-population", required=True, help="The frozen retained population artifact")
    parser.add_argument("--expected-task-specs-sha256", required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--output", required=True, help="The backfilled population's versioned artifact prefix")
    parser.add_argument("--policy-revision", required=True)
    parser.add_argument("--residual-threshold", type=float, default=None)
    parser.add_argument("--dataset-id", required=True, help="The declared dataset pin the census and lexical gates bind")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--converter-revision", required=True, help="The landing commit the producer ran at")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    configure_coreweave_s3()

    digest = hashlib.sha256()
    rows = spec_rows(args.parent_population)
    for _, encoded, _ in rows:
        digest.update(encoded.encode())
    actual = digest.hexdigest()
    if actual != args.expected_task_specs_sha256:
        raise ValueError(f"The frozen population's task specs changed: {actual} != {args.expected_task_specs_sha256}")
    if len(rows) != args.task_count:
        raise ValueError(f"The frozen population holds {len(rows)} tasks, not the declared {args.task_count}")
    LOGGER.info("Pinned the parent population: %d tasks", len(rows))

    policy = BackfillPolicy(args.policy_revision, args.residual_threshold)
    counts: Counter[str] = Counter()
    documents: list[tuple[dict[str, Any], dict[str, Any]]] = []
    bindings: list[dict[str, Any]] = []
    for _task_id, encoded, runtime in rows:
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
                counts["task:ambiguous_set"] += 1
        bindings.append(_binding(task, encoded, result))
        documents.append((json.loads(task_spec_json(result.task)), json.loads(runtime) if runtime else {}))

    files = write_text_population(documents, args.output)
    if files.task_count != len(documents):
        raise ValueError(f"The backfilled population holds {files.task_count} of {len(documents)} tasks")
    LOGGER.info("Wrote the backfilled population: %s (task_specs_sha256 %s)", args.output, files.task_specs_sha256)

    machine_ful = counts["task:normalized"] - counts["task:machine_less"]
    ambiguous_count = counts["task:ambiguous_set"]
    receipt = {
        "source_id": args.source_id,
        "parent_population": args.parent_population,
        "parent_task_specs_sha256": args.expected_task_specs_sha256,
        "population": args.output,
        "task_specs_sha256": files.task_specs_sha256,
        "task_count": files.task_count,
        "revision": REQUIREMENTS_REVISION,
        "policy": {"rules_revision": policy.rules_revision, "residual_threshold": policy.residual_threshold},
        "counts": {
            "tasks": {
                "total": counts["task:total"],
                "normalized": counts["task:normalized"],
                "machine_less": counts["task:machine_less"],
            },
            "outcomes": {outcome: counts[f"task:{outcome}"] for outcome in OUTCOMES},
            "capabilities": {
                capability: {outcome: counts[f"cap:{capability}:{outcome}"] for outcome in OUTCOMES}
                for capability in RULE_CAPABILITIES
            },
            "rules": {rule.name: counts[f"rule:{rule.name}"] for rule in RULES},
            "ambiguous_set": ambiguous_count,
        },
        "residual": {
            "ambiguous_share": ambiguous_count / machine_ful if machine_ful else 0.0,
            "over_threshold": (
                policy.residual_threshold is not None
                and machine_ful > 0
                and ambiguous_count / machine_ful > policy.residual_threshold
            ),
        },
        "carry_forward": {
            "bindings": len(bindings),
            "proof": (
                "every backfilled task differs from its parent only in environment_requirements.capabilities "
                "(extended) and the requirements.backfill.capabilities label"
            ),
        },
    }
    output = StoragePath(args.output)
    (output / "backfill-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "carry-forward-bindings.json").write_text(json.dumps(bindings, indent=1) + "\n")
    population_record = SourceTextPopulation(
        path=args.output,
        source_id=args.source_id,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        source_version=policy.rules_revision,
        converter_revision=args.converter_revision,
        text_renderer_version=RENDERER_VERSION,
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        converter_rejections={},
        admission_rejections={},
        within_source_dedup={
            "raw_count": files.raw_task_count,
            "unique_count": files.task_count,
            "within_source_duplicates": files.within_source_duplicates,
            "drop_manifest_uri": files.drop_manifest_uri,
        },
    )
    # Write the artifact-store record, so the gates resolve the population through
    # Artifact.raw_load and read its value fields instead of a path-only handle. The
    # payload excludes ``path``: raw_load passes the location itself as the handle.
    write_artifact(population_record.model_dump(mode="json", exclude={"path"}), args.output)
    (output / "population.json").write_text(population_record.model_dump_json(indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
