# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inventory permanent Hero snapshots and pin their saved configuration without reading weights."""

import argparse
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wandb
from levanter.checkpoint_manifest import CheckpointManifest
from rigging.filesystem.buckets import filesystem_for

from experiments.grug.moe_hero_ep.averaging import CheckpointPin, ProviderRegionProbe
from experiments.grug.moe_hero_ep.checkpoints import hero_checkpoint_paths
from experiments.grug.moe_hero_ep.current_run import CURRENT_HERO_RUN_ID
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest


def inventory_run(
    run_id: str,
    output_directory: Path,
    *,
    source_revision: str,
    tokenizer_revision: str,
    tokenizer_revision_evidence: str,
    checkpoint_count: int,
) -> dict:
    """Pin a compatible window within the current run's saved 16K configuration.

    Ancestry discovery is retained in the inventory, but only snapshots under
    this run's exact root are selected. Older phases require their own saved
    configurations and compatibility review. An explicit tokenizer-evidence
    reference is required because historical saved run configs omit its SHA.
    """
    if checkpoint_count < 2 or not tokenizer_revision_evidence:
        raise ValueError("Declare at least two checkpoints and the source tokenizer revision evidence")
    for revision in (source_revision, tokenizer_revision):
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError("Source and tokenizer revisions must be immutable git SHAs")
    if output_directory.exists():
        raise FileExistsError("Choose a fresh inventory directory")
    run = wandb.Api(timeout=30).run(f"marin-community/marin_moe/{run_id}")
    saved = run.config
    trainer = saved["trainer"]["trainer"]
    if saved["model"]["max_seq_len"] != 16384:
        raise ValueError("This format-SFT inventory expects the 16K source regime")
    checkpointer = trainer["checkpointer"]
    root = checkpointer["base_path"].rstrip("/")
    if checkpointer["append_run_id_to_base_path"]:
        root += "/" + trainer["id"]
    paths = hero_checkpoint_paths(run_id)
    region_probe = ProviderRegionProbe()
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.startswith(root + "/"):
            continue
        fs, plain = filesystem_for(path)
        with fs.open(f"{plain.rstrip('/')}/metadata.json") as stream:
            metadata = json.load(stream)
        with fs.open(f"{plain.rstrip('/')}/manifest.json") as stream:
            raw_manifest = json.load(stream)
        manifest = CheckpointManifest.model_validate(raw_manifest)
        if metadata.get("is_temporary") is not False:
            raise ValueError(f"Snapshot is no longer permanent: {path}")
        records.append(
            {
                "path": path,
                "metadata": metadata,
                "metadata_digest": digest(metadata),
                "manifest_digest": digest(raw_manifest),
                "arrays": len(manifest.arrays),
                "storage_region": region_probe.storage_region(path),
            }
        )
    records.sort(key=lambda row: (row["metadata"]["step"], row["path"]))
    if len({row["metadata"]["step"] for row in records}) != len(records):
        raise ValueError("Distinct paths share a step; select and record their lineage explicitly")
    selected = records[-checkpoint_count:]
    if len({row["storage_region"] for row in selected}) > 1:
        raise ValueError("Selected checkpoint window spans storage regions")
    output_directory.mkdir(parents=True)
    config_path = output_directory / "saved-config.json"
    config_path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    pins = [
        CheckpointPin(
            path=row["path"],
            metadata_digest=row["metadata_digest"],
            config_path=str(config_path.resolve()),
            config_digest=digest(saved),
            tokenizer_revision=tokenizer_revision,
            manifest_digest=row["manifest_digest"],
            tokenizer_revision_evidence=tokenizer_revision_evidence,
        )
        for row in selected
    ]
    result = {
        "run_id": run_id,
        "recorded_at": datetime.now(UTC).isoformat(),
        "source_revision": source_revision,
        "launch_revision": saved.get("hero_launch_commit"),
        "saved_config_digest": digest(saved),
        "model_digest": digest(saved["model"]),
        "optimizer_digest": digest(saved["optimizer"]),
        "tokenizer": saved["data"]["tokenizer"],
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_revision_evidence": tokenizer_revision_evidence,
        "ancestry_paths": paths,
        "current_phase_inventory": records,
        "requested_checkpoint_count": checkpoint_count,
        "selected": [dataclasses.asdict(pin) for pin in pins],
        "averaging_ready": len(pins) >= 2,
        "shorter_window": len(pins) < checkpoint_count,
        "readability": "Metadata and manifests read; model-array readability remains unverified",
        "averaging_placement": "All workers, temporary storage and output must match the verified source region",
    }
    result["inventory_digest"] = digest(result)
    (output_directory / "inventory.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--run-id", default=CURRENT_HERO_RUN_ID)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--tokenizer-revision-evidence", required=True)
    parser.add_argument("--checkpoint-count", type=int, default=3)
    args = parser.parse_args()
    inventory_run(
        args.run_id,
        args.output_directory,
        source_revision=args.source_revision,
        tokenizer_revision=args.tokenizer_revision,
        tokenizer_revision_evidence=args.tokenizer_revision_evidence,
        checkpoint_count=args.checkpoint_count,
    )


if __name__ == "__main__":
    main()
