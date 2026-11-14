# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select task identities while retaining their published evaluation repetitions."""

import json
from collections import Counter

import pyarrow as pa

from experiments.post_training.long_loop.eval_runtime.profile import SKIPPED_BENCHMARKS


def selected_rollouts(unique: pa.Table, expanded: pa.Table, tasks_per_benchmark: int) -> tuple[pa.Table, dict]:
    selected = {}
    counts = Counter()
    for row in unique.to_pylist():
        benchmark = row["benchmark"]
        if benchmark in SKIPPED_BENCHMARKS or counts[benchmark] >= tasks_per_benchmark:
            continue
        if row["task_id"] in selected:
            raise ValueError("Unique evaluation table repeats a task identity")
        selected[row["task_id"]] = row
        counts[benchmark] += 1
    expected_paths = set(unique["benchmark"].to_pylist()) - set(SKIPPED_BENCHMARKS)
    if set(counts) != expected_paths or any(count != tasks_per_benchmark for count in counts.values()):
        raise ValueError("Evaluation selection does not cover the requested tasks in every supported path")
    chosen = []
    repetitions = {identity: set() for identity in selected}
    attempts = Counter()
    agents = Counter()
    sources = Counter()
    identities = []
    for index, row in enumerate(expanded.to_pylist()):
        if row["task_id"] not in selected:
            continue
        source = selected[row["task_id"]]
        for key in ("benchmark", "task_spec", "task_rollout_spec", "agent", "repetitions"):
            if source[key] != row[key]:
                raise ValueError(f"Expanded evaluation row changed {key}: {row['task_id']}")
        repetition = row["repetition_index"]
        seen = repetitions[row["task_id"]]
        if repetition in seen:
            raise ValueError("Expanded evaluation table duplicates an attempt")
        seen.add(repetition)
        attempts[row["benchmark"]] += 1
        agents[row["agent"]] += 1
        data_source = json.loads(row["task_spec"])["source"]["dataset"]
        sources[data_source] += 1
        identities.append(
            {
                "dataset_row_uid": str(len(chosen)),
                "task_id": row["task_id"],
                "repetition_index": repetition,
                "benchmark": row["benchmark"],
                "data_source": data_source,
            }
        )
        chosen.append(index)
    for identity, source in selected.items():
        if repetitions[identity] != set(range(source["repetitions"])):
            raise ValueError(f"Published evaluation attempts are incomplete: {identity}")
    return expanded.take(pa.array(chosen)), {
        "unique_tasks": len(selected),
        "attempts": len(chosen),
        "paths": len(counts),
        "tasks_per_benchmark": dict(sorted(counts.items())),
        "attempts_per_benchmark": dict(sorted(attempts.items())),
        "attempts_per_data_source": dict(sorted(sources.items())),
        "attempts_per_agent": dict(sorted(agents.items())),
        "identities": identities,
        "task_repetitions": {identity: sorted(values) for identity, values in repetitions.items()},
    }
