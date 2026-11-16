# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply exact Atlas duplicate and measured lexical dispositions to bound cache rows."""

import argparse
import hashlib
import json

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.exp3_runtime import configure_storage, implementation_identity, read_pinned
from experiments.sft_new_mix.final_receipts import collect_final_inventory
from experiments.sft_new_mix.finalize_shards import FINALIZE_IMPLEMENTATION_FILES, DropStage, execute_plan
from experiments.sft_new_mix.prepare_overlay_gates import pinned_json


def bind_disposition(identifier: str, stage: DropStage, reason: str, shards: dict, rows: dict) -> None:
    prefix, index = identifier.split(":")
    if prefix not in shards or not 0 <= int(index) < shards[prefix]["rows"]:
        raise ValueError("Atlas disposition lies outside its bound candidate cache")
    dispositions = rows[prefix].setdefault(str(int(index)), {})
    if stage in dispositions:
        raise ValueError("An Atlas disposition repeats the same stage for one source row")
    dispositions[stage] = reason


def propagate_lexical_duplicates(duplicate_of: dict[str, str], lexical_ids: set[str], shards: dict, rows: dict) -> int:
    """Keep decontamination effective when exact-deduplication is rolled back.

    Returns the number of rows that received a new lexical-drop disposition.
    """
    count = 0
    for identifier, kept in duplicate_of.items():
        if kept not in lexical_ids:
            continue
        bind_disposition(identifier, DropStage.LEXICAL, "atlas_measured_lexical_overlap_duplicate", shards, rows)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-sha256", required=True)
    args = parser.parse_args()
    configure_storage()
    recipe = pinned_json({"uri": args.recipe, "sha256": args.recipe_sha256})
    inventory = pinned_json(recipe["inventory"])
    report = pinned_json(recipe["report"])
    populations = pinned_json(recipe["population_pins"])
    sources = {
        source["source"]: source
        for source in inventory["sources"]
        if source["loss_mode"] == "chat_assistant_only" and source["shards"]
    }
    if inventory["complete"] is not True or set(report["source_ids"]) != {"exp3-sft/" + name for name in sources}:
        raise ValueError("Atlas report does not cover the exact complete candidate SFT population")
    if set(populations) != set(sources):
        raise ValueError("Atlas population pins omit candidate sources")
    jobs, source_loss, pins = [], {}, []
    for name, source in sources.items():
        population_pin = populations[name]
        population = json.loads(
            read_pinned(population_pin["path"] + "/.artifact.json", population_pin["metadata_sha256"])
        )["result"]
        digest = hashlib.sha256(
            json.dumps(
                {key: source[key] for key in ("source", "source_revision", "loss_mode", "shards")}, sort_keys=True
            ).encode()
        ).hexdigest()
        if population["source_version"] != digest or population_pin["source_input_sha256"] != digest:
            raise ValueError("Atlas population and candidate source disagree")
        shards = {hashlib.sha256(shard["cache_uri"].encode()).hexdigest(): shard for shard in source["shards"]}
        rows = {prefix: {} for prefix in shards}
        dedup = population["within_source_dedup"]
        expected_raw = source["counts"]["output_documents"]
        if dedup["raw_count"] != expected_raw:
            raise ValueError("Atlas duplicate accounting omits candidate documents")
        inputs = [
            (dedup["drop_manifest_uri"], DropStage.EXACT_DUPLICATE, "task_id", dedup["within_source_duplicates"]),
            (
                report["drop_manifests"]["exp3-sft/" + name],
                DropStage.LEXICAL,
                "id",
                report["measurements"]["exp3-sft/" + name]["contaminated_tasks"],
            ),
        ]
        duplicate_of, lexical_ids = {}, set()
        for uri, stage, column, expected in inputs:
            with StoragePath(uri).open("rb") as stream:
                sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            count = 0
            with StoragePath(uri).open("rb") as stream:
                columns = [column, "kept_id"] if stage == DropStage.EXACT_DUPLICATE else [column]
                for batch in pq.ParquetFile(stream).iter_batches(batch_size=1024, columns=columns):
                    for record in batch.to_pylist():
                        identifier = record[column]
                        if stage == DropStage.EXACT_DUPLICATE:
                            duplicate_of[identifier] = record["kept_id"]
                        else:
                            lexical_ids.add(identifier)
                        bind_disposition(
                            identifier,
                            stage,
                            (
                                "exact_duplicate"
                                if stage == DropStage.EXACT_DUPLICATE
                                else "atlas_measured_lexical_overlap"
                            ),
                            shards,
                            rows,
                        )
                        count += 1
            if count != expected:
                raise ValueError("Atlas drop manifest and measured accounting disagree")
            pins.append({"source": name, "stage": stage, "uri": uri, "sha256": sha256, "documents": count})
        propagated = propagate_lexical_duplicates(duplicate_of, lexical_ids, shards, rows)
        pins.append({"source": name, "stage": "atlas_lexical_duplicate_propagation", "documents": propagated})
        source_loss[name] = {
            key: source[key]
            for key in ("loss_mode", "loss_weights_key", "loss_weights_indexing", "teacher_only", "source_revision")
        }
        for prefix, shard in shards.items():
            jobs.append(
                {
                    "source": name,
                    "base_shard": shard,
                    "loss_mode": source["loss_mode"],
                    "teacher_only": source["teacher_only"],
                    "passed": False,
                    "output_prefix": str(
                        StoragePath(recipe["output_prefix"]) / "components" / hashlib.sha256(name.encode()).hexdigest()
                    ),
                    "implementation_sha256": implementation_identity(FINALIZE_IMPLEMENTATION_FILES),
                    "overlays": [],
                    "overlay_enabled": False,
                    "real_tools_enabled": False,
                    "drop_rows": rows[prefix],
                    "enabled_drop_stages": recipe["enabled_drop_stages"],
                    "tokenizer_uri": inventory["tokenizer_uri"],
                    "tokenizer_sha256": inventory["tokenizer_sha256"],
                    **recipe["filter_config"],
                }
            )
    plan = {"jobs": jobs, "output_prefix": recipe["output_prefix"], "scope": "Complete candidate SFT Atlas dispositions"}
    root = StoragePath(recipe["output_prefix"])
    (root / "plan.json").write_text(json.dumps(plan, sort_keys=True))
    (root / "disposition-pins.json").write_text(json.dumps(pins, sort_keys=True))
    execute_plan(plan, recipe["workers"])
    result = collect_final_inventory(plan, source_loss, recipe["receipt_workers"])
    if result["complete"] is not True:
        raise ValueError("Atlas disposition application did not produce every expected receipt")
    (root / "inventory.json").write_text(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
