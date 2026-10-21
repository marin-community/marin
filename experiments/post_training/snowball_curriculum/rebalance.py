# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebalance a frozen pool and add faithful historical native closure contracts."""

import argparse
import hashlib
import heapq
import json
import shutil
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.sampling import seeded_order

from experiments.post_training.snowball_curriculum.bundle import verify_manifest, write_manifest, write_pool
from experiments.post_training.snowball_curriculum.native import NativeSelection, select_native_closure_tasks
from experiments.post_training.snowball_curriculum.selection import (
    TASK_SPEC_CONTRACT_KIND,
    allocate_quotas,
    source_weights,
)


def canonical_source_quotas(
    previous: dict, families: Mapping[str, str], family_quotas: Mapping[str, int]
) -> dict[str, int]:
    """Preserve scarce calendar and structural rows, reducing other source quotas."""
    quotas = {}
    for family, target in family_quotas.items():
        capacities = {
            source: count for source, count in previous["eligible_counts"].items() if families[source] == family
        }
        if family in {"calendar", "structural"}:
            quotas.update({source: previous["source_quotas"][source] for source in capacities})
            continue
        quotas.update(allocate_quotas(capacities, target, source_weights(capacities, equal_fraction=0)))
    if any(quota > previous["source_quotas"][source] for source, quota in quotas.items()):
        raise ValueError("Requested source quota exceeds the frozen predecessor sample")
    return dict(sorted(quotas.items()))


def retained_contracts(
    previous_root: Path, source_quotas: Mapping[str, int]
) -> tuple[set[str], dict[str, dict[str, int]]]:
    """Keep the smaller seeded prefix in each original source/difficulty bin."""
    report = json.loads((previous_root / "selection.json").read_text())
    quotas = {
        source: allocate_quotas(counts, source_quotas[source], dict.fromkeys(counts, 1.0))
        for source, counts in report["eligible_difficulty_counts"].items()
    }
    selected = set()
    seen: Counter[tuple[str, str]] = Counter()
    previous_order = None
    parquet = pq.ParquetFile(previous_root / "train.parquet")
    for batch in parquet.iter_batches(batch_size=1024, columns=["source_id", "contract_id", "difficulty"]):
        for row in batch.to_pylist():
            order = seeded_order(row["contract_id"], report["seed"])
            if previous_order is not None and order <= previous_order:
                raise ValueError("Frozen predecessor is not in unique seeded contract order")
            previous_order = order
            source, difficulty = row["source_id"], row["difficulty"]
            if seen[source, difficulty] < quotas[source][difficulty]:
                selected.add(row["contract_id"])
            seen[source, difficulty] += 1
    expected = {
        (source, difficulty): count
        for source, counts in report["difficulty_quotas"].items()
        for difficulty, count in counts.items()
    }
    if dict(seen) != expected or len(selected) != sum(source_quotas.values()):
        raise ValueError("Frozen predecessor does not contain the required bin counts")
    return selected, quotas


def canonical_records(
    root: Path, selected: set[str], bin_quotas: Mapping[str, Mapping[str, int]], recipe: dict, report: dict
) -> Iterator[dict[str, Any]]:
    with (root / "tasks.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["contract_id"] not in selected:
                continue
            source, difficulty = row["source_id"], row["difficulty"]
            quota = bin_quotas[source][difficulty]
            row.update(
                contract_kind=TASK_SPEC_CONTRACT_KIND,
                lowered_task_spec=None,
                family=recipe["resolved_source_families"][source],
                previous_pool_inclusion_probability=row["inclusion_probability"],
                retention_probability=quota / report["difficulty_quotas"][source][difficulty],
                inclusion_probability=quota / report["eligible_difficulty_counts"][source][difficulty],
                sampling_stratum=f"{source}/{difficulty}",
                native_format=None,
                native_family=None,
                native_control=None,
                native_prefix=None,
                tokenizer_model=None,
                tokenizer_revision=None,
                runtime_compatibility="schema-validated-not-executed",
            )
            yield row


def native_records(selection: NativeSelection) -> Iterator[dict[str, Any]]:
    for task in selection.selected:
        yield {
            "source_id": task.source_id,
            "task_id": task.task_id,
            "contract_id": task.contract_id,
            "public_content_id": task.public_content_id,
            "contract_kind": task.contract_kind,
            "task_json": None,
            "lowered_task_spec": task.lowered_task_spec,
            "family": "structural",
            "difficulty": "unknown",
            "intended_use": "train",
            "population": "reviewed-native-import",
            "prior_admission_status": task.prior_admission_status,
            "inclusion_probability": selection.stratum_quotas[task.stratum] / selection.eligible_counts[task.stratum],
            "previous_pool_inclusion_probability": None,
            "retention_probability": None,
            "sampling_stratum": task.stratum,
            "native_format": task.format,
            "native_family": task.family,
            "native_control": task.control,
            "native_prefix": task.prefix,
            "tokenizer_model": task.tokenizer_model,
            "tokenizer_revision": task.tokenizer_revision,
            "runtime_compatibility": task.runtime_compatibility,
            "teacher_route": task.teacher_route,
        }


def write_rebalanced_pool(
    previous_root: Path,
    output: Path,
    plan: dict,
    native: NativeSelection,
    *,
    native_train: Path,
    native_validation: Path,
    expected_manifest_sha256: str,
) -> None:
    """Write a mixed pool without retaining canonical task payloads in memory."""
    if output.exists():
        raise FileExistsError(output)
    verify_manifest(StoragePath(str(previous_root)), expected_manifest_sha256)
    previous = json.loads((previous_root / "selection.json").read_text())
    recipe = json.loads((previous_root / "recipe.json").read_text())
    for field in ("catalog_uri", "catalog_snapshot_id", "migration_receipt"):
        if recipe.get(field) is None:
            raise ValueError(f"Original pinned recipe lacks {field}")
    seed = plan["seed"]
    if seed != previous["seed"] or len(native.selected) != plan["native_rows"] or native.shortfall:
        raise ValueError("Predecessor seed or native sample differs from the approved plan")
    quotas = canonical_source_quotas(previous, recipe["resolved_source_families"], plan["family_quotas"])
    selected, bin_quotas = retained_contracts(previous_root, quotas)
    sources = json.loads((previous_root / "sources.json").read_text())
    snapshots = {source["catalog_snapshot_id"] for source in sources["sources"] if source["population"] == "admitted"}
    if snapshots != {recipe["catalog_snapshot_id"]}:
        raise ValueError("Admitted source snapshot differs from the original pinned recipe")
    native_source = native.selected[0].source_id
    sources["sources"].append(
        {
            "source_id": native_source,
            "intended_use": "train",
            "population": "reviewed-native-import",
            "family": "structural",
            "contract_kind": native.selected[0].contract_kind,
            "task_paths": [str(native_train)],
            "validation_paths": [str(native_validation)],
            "dataset_repository": "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data",
            "dataset_revision": "cb728a7636caad3fc6fa6b6ba85a49194d9d69a3",
            "file_sha256": native.input_sha256,
            "tokenizer_model": native.selected[0].tokenizer_model,
            "tokenizer_revision": native.selected[0].tokenizer_revision,
            "runtime_compatibility": "historical-source-only",
        }
    )
    recipe.update(
        version=plan["version"],
        predecessor_version=plan["previous_version"],
        predecessor_artifact_manifest_sha256=hashlib.sha256(
            (previous_root / "artifact-manifest.json").read_bytes()
        ).hexdigest(),
        requested_family_weights={
            family: quota / sum(plan["family_quotas"].values()) for family, quota in plan["family_quotas"].items()
        },
        resolved_family_weights={
            family: quota / sum(plan["family_quotas"].values()) for family, quota in plan["family_quotas"].items()
        },
        requested_family_quotas=plan["family_quotas"],
        resolved_family_quotas=plan["family_quotas"],
        canonical_source_quotas=quotas,
        native_structural_quota=len(native.selected),
        native_stratification=(
            "Equal mass across native format/family groups, then controls, capped by eligible capacity"
        ),
        population_policy="Production-admitted canonical TaskSpecs plus reviewed historical native closure imports",
        structural_policy="Keep admitted structural-output rows and add reviewed historical closure contracts",
        difficulty_basis=plan["difficulty_policy"],
        selection_policy=plan["selection_policy"],
        native_runtime_readiness="Historical source audit only; export is a separate semantic conversion",
        native_input_sha256=native.input_sha256,
    )
    recipe["resolved_source_families"][native_source] = "structural"
    recipe["extensions"][native_source] = "structural"
    recipe["difficulty_by_source"] = {**recipe.get("difficulty_by_source", {}), native_source: "unknown"}
    output.mkdir(parents=True)
    provenance = output / "provenance"
    provenance.mkdir()
    for name in ["selection.json", "recipe.json", "sources.json", "artifact-manifest.json"]:
        shutil.copyfile(previous_root / name, provenance / f"predecessor-{name}")
    if (previous_root / "provenance").exists():
        for path in (previous_root / "provenance").iterdir():
            if path.is_file():
                shutil.copyfile(path, provenance / path.name)
    code = provenance / "code"
    code.mkdir()
    for name in ["rebalance.py", "native.py", "selection.py", "bundle.py"]:
        shutil.copyfile(Path(__file__).parent / name, code / name)
    (output / "sources.json").write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    counts: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    families: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    contracts = set()
    records = heapq.merge(
        canonical_records(previous_root, selected, bin_quotas, recipe, previous),
        native_records(native),
        key=lambda row: seeded_order(row["contract_id"], seed),
    )

    def checked_records() -> Iterator[dict[str, Any]]:
        for row in records:
            if row["contract_id"] in contracts:
                raise ValueError("Duplicate full contract in rebalanced pool")
            contracts.add(row["contract_id"])
            counts[row["source_id"]] += 1
            difficulties[f"{row['source_id']}/{row['difficulty']}"] += 1
            families[row["family"]] += 1
            kinds[row["contract_kind"]] += 1
            yield row

    write_pool(checked_records(), StoragePath(str(output)))
    if dict(families) != plan["family_quotas"] or sum(counts.values()) != sum(plan["family_quotas"].values()):
        raise ValueError("Rebalanced pool differs from the approved family allocation")
    with (output / "selected-aliases.jsonl").open("w") as stream:
        with (previous_root / "selected-aliases.jsonl").open() as previous_aliases:
            for line in previous_aliases:
                if json.loads(line)["contract_id"] in selected:
                    stream.write(line)
        for contract, aliases in native.selected_aliases.items():
            stream.write(json.dumps({"contract_id": contract, "aliases": [asdict(alias) for alias in aliases]}) + "\n")
    report = {
        **previous,
        "version": plan["version"],
        "selected_count": sum(counts.values()),
        "shortfall": 0,
        "source_quotas": dict(counts),
        "family_quotas": plan["family_quotas"],
        "family_counts": dict(families),
        "contract_kind_counts": dict(kinds),
        "difficulty_counts": dict(difficulties),
        "difficulty_quotas": {**bin_quotas, native_source: {"unknown": len(native.selected)}},
        "eligible_counts": {**previous["eligible_counts"], native_source: native.counts["eligible"]},
        "candidate_counts": {**previous["candidate_counts"], native_source: native.counts["train"]},
        "eligible_difficulty_counts": {
            **previous["eligible_difficulty_counts"],
            native_source: {"unknown": native.counts["eligible"]},
        },
        "native_counts": native.counts,
        "native_stratum_quotas": native.stratum_quotas,
        "native_family_quotas": native.family_quotas,
        "native_eligible_stratum_counts": native.eligible_counts,
        "source_snapshot_sha256": canonical_sha256(sources),
        "probability_policy": (
            "Canonical probability uses original full eligible source/difficulty census; "
            "native probability uses eligible format/family/control stratum census"
        ),
    }
    (output / "selection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    verify_manifest(StoragePath(str(previous_root)), expected_manifest_sha256)
    write_manifest(StoragePath(str(output)))
    print(json.dumps({"output": str(output), "selected_count": sum(counts.values()), "family_counts": dict(families)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--previous-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--native-train", type=Path, required=True)
    parser.add_argument("--native-validation", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    previous_recipe = json.loads((args.previous / "recipe.json").read_text())
    native = select_native_closure_tasks(
        str(args.native_train),
        str(args.native_validation),
        target=plan["native_rows"],
        seed=plan["seed"],
        excluded_task_ids=previous_recipe["heldout_task_ids"],
    )
    print(json.dumps({"stage": "native_selected", "counts": native.counts}), flush=True)
    write_rebalanced_pool(
        args.previous,
        args.output,
        plan,
        native,
        native_train=args.native_train,
        native_validation=args.native_validation,
        expected_manifest_sha256=args.previous_manifest_sha256,
    )


if __name__ == "__main__":
    main()
