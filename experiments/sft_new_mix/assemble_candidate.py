# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble complete source cohorts while preserving control and transform lineage."""

import argparse
import hashlib
import json

from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.candidate_receipts import catalog_from_inventory
from experiments.sft_new_mix.exp3_runtime import configure_storage
from experiments.sft_new_mix.prepare_overlay_gates import pinned_json

SUPERVISION_FIELDS = ("loss_mode", "loss_weights_key", "loss_weights_indexing", "teacher_only")


def correct_replay_inventory(base: dict, corrected: dict) -> dict:
    """Replace the complete replay cohort while preserving prepared SFT sources."""
    if not base["complete"] or not corrected["complete"] or base["tokenizer_sha256"] != corrected["tokenizer_sha256"]:
        raise ValueError("Replay correction requires complete populations with the same tokenizer")
    previous = {row["source"]: row for row in base["sources"] if row["source"].startswith("replay/")}
    replay = {row["source"]: row for row in corrected["sources"] if row["source"].startswith("replay/")}
    if previous.keys() != replay.keys():
        raise ValueError("Replay correction changed the source scope")
    for name, source in replay.items():
        before = previous[name]
        if any(source[field] != before[field] for field in (*SUPERVISION_FIELDS, "source_revision")):
            raise ValueError("Replay correction changed source provenance or loss mode")
        if any(source["counts"][field] != before["counts"][field] for field in ("input_documents", "input_tokens")):
            raise ValueError("Replay correction changed the original input population")
        if source["counts"]["output_documents"] > source["counts"]["input_documents"]:
            raise ValueError("Replay correction created more rows than its original source")
    return {
        **base,
        "sources": [replay.get(row["source"], row) for row in base["sources"]],
        "transforms": (
            [row for row in base["transforms"] if row["source"] not in replay]
            + [
                {**row, "phase": "scripted_replay_corrected"}
                for row in corrected["transforms"]
                if row["source"] in replay
            ]
        ),
        "replay_restoration_recipe": corrected["replay_restoration_recipe"],
        "replay_line_length_rejection_enabled": corrected["replay_line_length_rejection_enabled"],
    }


def combine_source_inventories(
    base: dict, cohorts: list[dict], overrides: dict[str, str], additions: dict[str, str]
) -> dict:
    """Require every nonempty chat source and every declared addition exactly once."""
    if base["complete"] is not True or any(not source["complete"] for source in base["sources"]):
        raise ValueError("Cannot assemble a candidate from an incomplete scripted population")
    originals = {source["source"]: source for source in base["sources"]}
    replacements = {}
    transforms = [
        {
            **row,
            "phase": row.get("phase", "scripted"),
            "superseded_by_source_rebuild": row.get("superseded_by_source_rebuild", False) or row["source"] in overrides,
        }
        for row in base["transforms"]
    ]
    for cohort in cohorts:
        if cohort["complete"] is not True or cohort["tokenizer_sha256"] != base["tokenizer_sha256"]:
            raise ValueError("A final source cohort is incomplete or uses a different tokenizer")
        for source in cohort["sources"]:
            name = source["source"]
            if not source["complete"] or name in replacements:
                raise ValueError("Final source cohorts are incomplete or overlap")
            if name not in originals and name not in additions:
                raise ValueError("Final source cohort contains an undeclared addition")
            if name in originals:
                original = originals[name]
                if any(source[field] != original[field] for field in SUPERVISION_FIELDS):
                    raise ValueError("Source supervision changed during final assembly")
                revision = overrides.get(name, original["source_revision"])
            else:
                revision = additions[name]
            if source["source_revision"] != revision:
                raise ValueError("Source revision differs from its explicit replacement or addition pin")
            replacements[name] = source
        transforms.extend({**row, "phase": row.get("phase", "final")} for row in cohort["transforms"])
    required = (
        {name for name, source in originals.items() if source["loss_mode"] == "chat_assistant_only" and source["shards"]}
        | set(overrides)
        | set(additions)
    )
    if required - replacements.keys():
        raise ValueError(f"Final source cohorts are missing: {sorted(required - replacements.keys())}")
    sources = []
    for name in sorted(originals.keys() | replacements.keys()):
        if name not in replacements:
            sources.append(originals[name])
            continue
        final = replacements[name]
        if name not in originals:
            sources.append(final)
            continue
        original = originals[name]
        counts = {
            **original["counts"],
            "final_stage_input_documents": final["counts"]["input_documents"],
            "final_stage_input_tokens": final["counts"]["input_tokens"],
            **{
                field: final["counts"][field]
                for field in ("output_documents", "output_tokens", "supervised_prediction_positions")
            },
        }
        sources.append({**final, "counts": counts, "prior_stage_counts": original["counts"]})
    return {
        "complete": True,
        "training_ready": False,
        "readiness": (
            "Complete data population; decontamination, weights and consumer receipts must be pinned separately"
        ),
        "tokenizer_uri": base["tokenizer_uri"],
        "tokenizer_sha256": base["tokenizer_sha256"],
        "sources": sources,
        "transforms": transforms,
        "source_revision_overrides": {**base.get("source_revision_overrides", {}), **overrides},
        "source_additions": {**base.get("source_additions", {}), **additions},
        "source_replacements": base["source_replacements"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-sha256", required=True)
    args = parser.parse_args()
    configure_storage()
    recipe = pinned_json({"uri": args.recipe, "sha256": args.recipe_sha256})
    base = pinned_json(recipe["base_inventory"])
    if recipe["replay_correction"] is not None:
        base = correct_replay_inventory(base, pinned_json(recipe["replay_correction"]))
    inventory = combine_source_inventories(
        base,
        [pinned_json(pin) for pin in recipe["cohorts"]],
        recipe["source_revision_overrides"],
        recipe["source_additions"],
    )
    inventory["assembly_recipe"] = {"uri": args.recipe, "sha256": args.recipe_sha256}
    inventory["replay_correction_inventory"] = recipe["replay_correction"]
    if recipe["replay_correction"] is not None:
        inventory["replay_restoration_recipe"] = base["replay_restoration_recipe"]
        inventory["replay_line_length_rejection_enabled"] = base["replay_line_length_rejection_enabled"]
    supplemental = pinned_json(recipe["supplemental_source_receipts"])
    inventory["transforms"].extend(supplemental["transforms"])
    inventory["supplemental_source_lineage"] = supplemental["source_lineage"]
    root = StoragePath(recipe["output_prefix"])
    payloads = {"inventory.json": json.dumps(inventory, sort_keys=True).encode()}
    if recipe["output_kind"] == "catalog":
        catalog, sidecar = catalog_from_inventory(inventory, recipe["cache_root"])
        payloads.update(
            {
                "cache_catalog.json": catalog.to_json().encode(),
                "source_loss.json": json.dumps(sidecar, sort_keys=True).encode(),
            }
        )
    elif recipe["output_kind"] != "inventory":
        raise ValueError("Unknown candidate assembly output kind")
    pins = {}
    for name, content in payloads.items():
        path = root / name
        path.write_bytes(content)
        pins[name] = {"uri": str(path), "sha256": hashlib.sha256(content).hexdigest()}
    (root / "pins.json").write_text(json.dumps(pins, sort_keys=True))


if __name__ == "__main__":
    main()
