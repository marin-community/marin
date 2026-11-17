# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build one deterministic, fixed-row replacement comparison from admitted inputs."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.fingerprints import deduplication_key
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample

from experiments.post_training.task_curation.inventory import _panel_keys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_rows(path: Path):
    if path.suffix == ".jsonl":
        with path.open() as stream:
            for line in stream:
                task = TaskSpec.model_validate_json(line)
                yield line.strip(), task
        return
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["task_spec"]):
        for row in batch.to_pylist():
            value = row["task_spec"]
            yield value, TaskSpec.model_validate_json(value)


def eligible_rows(path: Path, excluded: set, counts: Counter):
    seen_ids = set()
    seen_contracts = set()
    for value, task in task_rows(path):
        counts["input"] += 1
        contract = deduplication_key(task)
        if not excluded.isdisjoint(_panel_keys(task).exclusion):
            counts["heldout_overlap"] += 1
        elif task.id in seen_ids or contract in seen_contracts:
            counts["duplicate"] += 1
        else:
            seen_ids.add(task.id)
            seen_contracts.add(contract)
            counts["eligible"] += 1
            yield value, task, contract


def build_pair(
    *,
    baseline: Path,
    candidate: Path,
    admitted_ids: Path,
    validation: Path,
    exclusions: list[Path],
    output: Path,
    stem_datasets: set[str],
    baseline_uri: str,
    candidate_uri: str,
    baseline_sampling_manifest: Path | None = None,
    size: int = 2000,
    replace: int = 200,
    seed: int = 20261020,
) -> dict:
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if not 0 < replace < size:
        raise ValueError("Replacement count must be positive and smaller than the control")
    excluded = {
        key for path in [validation, *exclusions] for _, task in task_rows(path) for key in _panel_keys(task).exclusion
    }
    control_counts = Counter()
    baseline_ids = set()
    baseline_contracts = set()

    def control_rows():
        for row in eligible_rows(baseline, excluded, control_counts):
            baseline_ids.add(row[1].id)
            baseline_contracts.add(row[2])
            yield row

    _, control = seeded_sample(control_rows(), size=size, key=lambda row: seeded_order(row[1].id, seed))
    if len(control) != size:
        raise ValueError(f"Control has {len(control)} eligible rows, needs {size}")
    displaced_positions = [index for index, row in enumerate(control) if row[1].source.dataset in stem_datasets][
        :replace
    ]
    if len(displaced_positions) != replace:
        raise ValueError(f"Control has only {len(displaced_positions)} replaceable STEM rows")

    admitted = set(admitted_ids.read_text().splitlines())
    if "" in admitted:
        raise ValueError("Admission ID list contains an empty ID")
    candidates_counts = Counter()
    source_rows = list(eligible_rows(candidate, excluded, candidates_counts))
    source_ids = {row[1].id for row in source_rows}
    if not admitted <= source_ids:
        raise ValueError("Admission IDs include candidate tasks absent after deduplication and heldout exclusion")
    candidate_rows = []
    for row in source_rows:
        if row[1].id not in admitted:
            candidates_counts["not_admitted"] += 1
        elif row[1].id in baseline_ids or row[2] in baseline_contracts:
            candidates_counts["baseline_overlap"] += 1
        else:
            candidate_rows.append(row)
    selected = sorted(candidate_rows, key=lambda row: seeded_order(row[1].id, seed))[:replace]
    if len(selected) != replace:
        raise ValueError(f"Candidate has {len(selected)} admitted unique rows, needs {replace}")
    treatment = list(control)
    for position, row in zip(displaced_positions, selected, strict=True):
        treatment[position] = row

    changed = [i for i, pair in enumerate(zip(control, treatment, strict=True)) if pair[0][0] != pair[1][0]]
    if changed != displaced_positions:
        raise AssertionError("Unexpected changed control rows")
    for arm in (control, treatment):
        if len({row[1].id for row in arm}) != size or len({row[2] for row in arm}) != size:
            raise AssertionError("An arm contains duplicate task IDs or contracts")

    inputs = {
        "baseline": {"uri": baseline_uri, "sha256": sha256(baseline)},
        "candidate": {"uri": candidate_uri, "sha256": sha256(candidate)},
        "admitted_ids": {"sha256": sha256(admitted_ids), "count": len(admitted)},
        "validation": {
            "sha256": sha256(validation),
            "task_specs_sha256": canonical_sha256([task.model_dump(mode="json") for _, task in task_rows(validation)]),
        },
        "exclusions": [{"sha256": sha256(path)} for path in exclusions],
    }
    if baseline_sampling_manifest is not None:
        inputs["baseline"]["sampling"] = json.loads(baseline_sampling_manifest.read_text())
        inputs["baseline"]["sampling_manifest_sha256"] = sha256(baseline_sampling_manifest)
    manifest = {
        "seed": seed,
        "rows_per_arm": size,
        "replacement_rows": replace,
        "replacement_share": replace / size,
        "stem_datasets": sorted(stem_datasets),
        "inputs": inputs,
        "control_counts": dict(control_counts),
        "candidate_counts": dict(candidates_counts),
        "dataset_counts": {
            name: dict(Counter(row[1].source.dataset for row in rows))
            for name, rows in (("control", control), ("candidate", treatment))
        },
        "control_ids": [row[1].id for row in control],
        "added_ids": [row[1].id for row in selected],
        "displaced_ids": [control[index][1].id for index in displaced_positions],
        "changed_positions": changed,
    }
    schema = pa.schema([pa.field("task_spec", pa.string(), nullable=False)])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="taskforge-pair-", dir=output.parent) as temporary:
        staged = Path(temporary) / "pair"
        for name, rows in (("control", control), ("candidate", treatment)):
            arm_dir = staged / name
            arm_dir.mkdir(parents=True)
            pq.write_table(
                pa.Table.from_pylist([{"task_spec": row[0]} for row in rows], schema=schema),
                arm_dir / "train.parquet",
            )
            shutil.copyfile(validation, arm_dir / "validation.parquet")
            if len(exclusions) == 1:
                shutil.copyfile(exclusions[0], arm_dir / "full_validation.parquet")
        manifest["training_sha256"] = {
            name: sha256(staged / name / "train.parquet") for name in ("control", "candidate")
        }
        (staged / "comparison.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(output)

    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "admitted-ids", "validation", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--exclusion", dest="exclusions", type=Path, action="append", required=True)
    parser.add_argument("--stem-dataset", dest="stem_datasets", action="append", required=True)
    parser.add_argument("--baseline-sampling-manifest", type=Path)
    parser.add_argument("--baseline-uri", required=True)
    parser.add_argument("--candidate-uri", required=True)
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--replace", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261020)
    args = vars(parser.parse_args())
    args["stem_datasets"] = set(args["stem_datasets"])
    manifest = build_pair(**args)
    print(json.dumps({key: manifest[key] for key in ("rows_per_arm", "replacement_rows", "replacement_share")}))


if __name__ == "__main__":
    main()
