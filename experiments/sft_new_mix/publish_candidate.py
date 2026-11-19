# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze candidate catalog, explicit supervision, mixture and retention artifacts."""

import argparse
import csv
import hashlib
import io
import json

from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.candidate_receipts import catalog_from_inventory
from experiments.sft_new_mix.exp3_runtime import configure_storage
from experiments.sft_new_mix.mixture_weights import candidate_weights
from experiments.sft_new_mix.prepare_overlay_gates import pinned_json


def csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def source_selection_rows(inventory: dict) -> list[dict]:
    """Account for frozen input replacements before scripted source transforms."""
    sources = {source["source"]: source for source in inventory["sources"]}
    rows = []
    for replacement in inventory["source_replacements"]:
        source = sources[replacement["source"]]
        science = replacement["transform"] == "accepted_science_swap"
        output_documents = replacement.get("after_rows", source["counts"]["input_documents"])
        output_tokens = replacement.get("after_tokens", source["counts"]["input_tokens"])
        rows.append(
            {
                "source": source["source"],
                "component": source["component"],
                "loss_mode": source["loss_mode"],
                "phase": "source_selection",
                "transform": replacement["transform"],
                "input_documents": replacement["before_rows"],
                "input_tokens": replacement["before_tokens"],
                "output_documents": output_documents,
                "output_tokens": output_tokens,
                "dropped_documents": replacement["before_rows"] - output_documents,
                "dropped_tokens": None if science else replacement["before_tokens"] - output_tokens,
                "net_serialized_token_change": output_tokens - replacement["before_tokens"],
                "drop_reason": "outside_accepted_science" if science else "missing_or_failed_recorded_outcome",
                "accounting_note": (
                    "Science repairs change text; excluded original-row token counts are unavailable. "
                    "Net serialized token change is measured, not a dropped-original-token count."
                    if science
                    else "Recorded-pass input selection precedes all scripted trace cleanup."
                ),
            }
        )
    return rows


def write_pinned(root: StoragePath, name: str, content: bytes) -> dict:
    """Refuse to change published bytes and return the written pin."""
    path = root / name
    if path.exists() and path.read_bytes() != content:
        raise ValueError("Published candidate bytes changed; use a new release version")
    path.write_bytes(content)
    return {"uri": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-sha256", required=True)
    args = parser.parse_args()
    configure_storage()
    recipe = pinned_json({"uri": args.recipe, "sha256": args.recipe_sha256})
    inventory = pinned_json(recipe["inventory"])
    scripted = pinned_json(recipe["scripted_plan"])
    components = pinned_json(recipe["control_components"])
    schedule = pinned_json(recipe["control_schedule"])
    proposal = pinned_json(recipe["mixture_proposal"])
    weights = candidate_weights(inventory, scripted, components, schedule, proposal["family_shares"], recipe["updates"])
    catalog, sidecar = catalog_from_inventory(inventory, recipe["cache_root"])
    root = StoragePath(recipe["output_prefix"])
    catalog_uri = str(root / "cache_catalog.json")
    control_reservations = recipe["control_validation_reservations"]
    reservations = {name: count for name, count in control_reservations.items() if weights.weights.get(name, 0) > 0}
    dropped_reservations = {name: count for name, count in control_reservations.items() if name not in reservations}
    train_weights = [[update, weights.weights] for update in recipe["schedule_updates"]]
    data = {
        "tokenizer": recipe["runtime_tokenizer_uri"],
        "cache_catalog": catalog_uri,
        "cache_dir": None,
        "auto_build_caches": False,
        "enforce_eos": False,
        "shuffle": True,
        "permutation_type": "feistel",
        "block_cross_document_attention": True,
        "stop_strategy": "restart",
        "mixture_block_size": recipe["mixture_block_size"],
        "num_validation_sequences": reservations,
        "shuffle_before_trainval_split": True,
        "components": {
            name: {
                "type": "cached",
                "cache_dir": recipe["cache_root"],
                "flat_cache": True,
                "split": "train",
                "format": {"type": "prebuilt", "input_ids_key": "input_ids", "loss_weights_key": "loss_weights"},
                "pack": recipe["packing_cap"],
            }
            for name in sorted(weights.weights)
        },
        "train_weights": train_weights,
    }
    source_rows = []
    for source in inventory["sources"]:
        row = {
            "source": source["source"],
            "component": source["component"],
            "loss_mode": source["loss_mode"],
            "source_revision": source["source_revision"],
            "teacher_only": source["teacher_only"],
            "weight": weights.weights.get(source["component"], 0),
            **source["counts"],
        }
        row["retained_serialized_positions"] = source["counts"]["output_tokens"]
        source_rows.append(row)
    transform_rows = []
    for row in source_selection_rows(inventory) + inventory["transforms"]:
        transform_rows.append(
            {
                **row,
                "input_serialized_positions": row.get("input_tokens"),
                "output_serialized_positions": row.get("output_tokens"),
            }
        )
    payloads = {
        "cache_catalog.json": catalog.to_json().encode(),
        "source_loss.json": json.dumps(sidecar, sort_keys=True).encode(),
        "data.json": json.dumps(data, sort_keys=True).encode(),
        "train_weights.json": json.dumps(train_weights, sort_keys=True).encode(),
        "retention-by-source.csv": csv_bytes(source_rows),
        "retention-by-transform.csv": csv_bytes(transform_rows),
        "mixture-rationale.json": (
            json.dumps(
                {
                    "proposal": proposal,
                    "source_families": weights.families,
                    "control_schedule_average": weights.control_average,
                    "replay_child_multiplicities": weights.replay_multiplicities,
                    "supplemental_source_lineage": inventory["supplemental_source_lineage"],
                    "validation_reservations": reservations,
                    "removed_validation_reservations": dropped_reservations,
                    "position_accounting": (
                        "Serialized positions equal retained IDs. Supervised prediction counts use measured masks. "
                        "Sampled context positions and padding are measured by the separate native consumer receipt."
                    ),
                },
                sort_keys=True,
            ).encode()
        ),
    }
    pins = {name: write_pinned(root, name, content) for name, content in payloads.items()}
    measurement = {
        "data_uri": pins["data.json"]["uri"],
        "data_sha256": pins["data.json"]["sha256"],
        "catalog_uri": catalog_uri,
        "catalog_sha256": pins["cache_catalog.json"]["sha256"],
        "source_loss_uri": pins["source_loss.json"]["uri"],
        "source_loss_sha256": pins["source_loss.json"]["sha256"],
        "steps": recipe["updates"],
        "batch_size": recipe["global_batch"],
        "sequence_length": recipe["sequence_length"],
        "trainer_seed": recipe["seed"],
        "data_seed": None,
        "jax_threefry_partitionable": True,
        "packing_cap": recipe["packing_cap"],
        "lookahead": recipe["lookahead"],
        "report_update_prefixes": recipe["report_update_prefixes"],
        "consumer_runtime_revision": recipe["consumer_runtime_revision"],
        "body_probe_sequences_per_source": 1,
        "expected_source_populations": {
            source["component"]: {
                "documents": source["counts"]["output_documents"],
                "tokens": source["counts"]["output_tokens"],
            }
            for source in inventory["sources"]
            if weights.weights.get(source["component"], 0) > 0
        },
    }
    pins["native-measurement-job.json"] = write_pinned(
        root, "native-measurement-job.json", json.dumps(measurement, sort_keys=True).encode()
    )
    manifest = {
        "training_ready": False,
        "readiness": "Data artifacts frozen; native consumer resolve and padding receipt required",
        "publication_recipe": {"uri": args.recipe, "sha256": args.recipe_sha256},
        "inventory": recipe["inventory"],
        "artifacts": pins,
        "geometry": {
            key: recipe[key]
            for key in ("updates", "global_batch", "sequence_length", "seed", "packing_cap", "lookahead")
        },
    }
    # The manifest cannot pin itself; the same write guard freezes its bytes.
    write_pinned(root, "manifest.json", json.dumps(manifest, sort_keys=True).encode())


if __name__ == "__main__":
    main()
