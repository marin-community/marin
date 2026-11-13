# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and export one arm of the #11188 data-patch experiment.

Each arm is the historical 262,144-context cold-start SFT recipe stopped after a
fixed number of updates. Arms differ only in the cache catalog, which maps each
mixture component to a tokenized store. ``train`` runs inside a CPU coordinator
job and dispatches the GPU gang; ``export`` converts one permanent checkpoint to
the Hugging Face layout that vLLM serves; ``catalog`` writes a patched catalog.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import draccus
import yaml
from levanter.store.cache import CacheLedger
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.post_training.long_loop.serving import SNOWBALL_CONTEXT_LIMIT
from experiments.post_training.long_loop.sft import TOKENIZER_URI, read_json, restore_frozen, sft_run, validate_handoff
from experiments.post_training.long_loop.sft_runtime.export_writer import ExportConfig, export_weights
from experiments.post_training.long_loop.sft_runtime.model import GrugModelConfig

logger = logging.getLogger(__name__)

EXPERIMENT_DIR = Path(__file__).parent / "patch_experiment"
LONG_LOOP_SPECS = Path(__file__).parents[1] / "post_training" / "long_loop" / "specs"
RECIPE_PATH = EXPERIMENT_DIR / "sft-262k-legacy.yaml"
INITIALIZER = "s3://marin-us-east-02a/marin/long-loop/base/2026.11.08.8"
CONTROL_CATALOG = "s3://marin-us-east-02a/marin/grug-sft/sft-pipe-repro/full-mixture/2026.10.24.2/cache_catalog.json"
WANDB_PROJECT = "sft-data-patch-11188"
GENERATION_CONFIG = {"bos_token_id": 128000, "eos_token_id": [128001, 128009], "pad_token_id": 128001}


def arm_recipe(catalog: str, steps: int, replicas: int, arm: str) -> dict:
    """Return the historical recipe bound to one catalog, step budget and gang size."""
    recipe = yaml.safe_load(RECIPE_PATH.read_text())
    data = yaml.safe_load((LONG_LOOP_SPECS / "sft-data.yaml").read_text())["data"]
    data["components"] = json.loads((LONG_LOOP_SPECS / "sft-components.json").read_text())
    data["train_weights"] = json.loads((LONG_LOOP_SPECS / "sft-train_weights.json").read_text())
    data["cache_catalog"] = catalog
    recipe["data"] = data
    recipe["stop_after_steps"] = steps
    recipe["resources"]["replicas"] = replicas
    tracker = recipe["trainer"]["trainer"]["tracker"]
    tracker["project"] = WANDB_PROJECT
    tracker["tags"] = ["sft-data-patch-11188", arm, f"steps-{steps}"]
    return recipe


def train(arm: str, catalog: str, output: str, steps: int, replicas: int) -> None:
    recipe = arm_recipe(catalog, steps, replicas, arm)
    sft_run(output=output, initializer=INITIALIZER, recipe=recipe, run_id=f"sft-patch-11188-{arm}")


def export(trained: str, step: int, output: str, source_revision: str) -> None:
    """Export ``checkpoints/step-<step>`` of a trained arm to ``<output>/hf``."""
    configure_coreweave_s3()
    checkpoint = prefix_join(trained, f"checkpoints/step-{step}")
    metadata = read_json(prefix_join(checkpoint, "metadata.json"))
    model = read_json(prefix_join(trained, "consumed-config.json"))["model"]
    config = ExportConfig(
        checkpoint,
        digest(metadata),
        draccus.decode(GrugModelConfig, model),
        prefix_join(output, "hf"),
        source_revision,
        max_position_embeddings=SNOWBALL_CONTEXT_LIMIT,
    )
    export_weights(config, restore_frozen, weights_kind="snowball_sft_frozen_router")
    source_fs, source = filesystem_for(TOKENIZER_URI)
    destination_fs, destination = filesystem_for(prefix_join(output, "hf"))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        destination_fs.pipe_file(prefix_join(destination, name), source_fs.cat_file(prefix_join(source, name)))
    destination_fs.pipe_file(prefix_join(destination, "generation_config.json"), json.dumps(GENERATION_CONFIG).encode())
    manifest = read_json(prefix_join(output, "hf/export-manifest.json"))
    for shard in manifest["shards"]:
        with destination_fs.open(prefix_join(destination, shard["filename"]), "rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != shard["sha256"]:
                raise ValueError(f"Export hash mismatch: {shard['filename']}")
    validate_handoff(output)


def write_catalog(output: str, replacements: dict[str, str]) -> None:
    """Copy the control catalog with some components pointed at other stores.

    ``replacements`` maps a component name to the store directory that replaces
    it; the entry's ledger is read from that store.
    """
    catalog = json.loads(StoragePath(CONTROL_CATALOG).read_text())
    entries = catalog["splits"]["train"]
    for name, cache_dir in replacements.items():
        if name not in entries:
            raise ValueError(f"{name} is not a train component of the control catalog")
        ledger = json.loads(CacheLedger.load(cache_dir).to_json())
        logger.info(
            "%s: %s rows -> %s rows, %s tokens -> %s tokens",
            name,
            entries[name]["ledger"]["total_num_rows"],
            ledger["total_num_rows"],
            entries[name]["ledger"]["field_counts"]["input_ids"],
            ledger["field_counts"]["input_ids"],
        )
        entries[name] = {"cache_dir": cache_dir, "ledger": ledger}
    StoragePath(output).write_text(json.dumps(catalog))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--arm", required=True)
    train_parser.add_argument("--catalog", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--steps", type=int, required=True)
    train_parser.add_argument("--replicas", type=int, required=True, help="Eight-H100 nodes in the training gang")

    export_parser = commands.add_parser("export")
    export_parser.add_argument("--trained", required=True)
    export_parser.add_argument("--step", type=int, required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--source-revision", required=True)

    catalog_parser = commands.add_parser("catalog")
    catalog_parser.add_argument("--output", required=True)
    catalog_parser.add_argument(
        "--replace", nargs=2, action="append", required=True, metavar=("COMPONENT", "STORE"), help="Repeatable"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "train":
        train(args.arm, args.catalog, args.output, args.steps, args.replicas)
    elif args.command == "export":
        export(args.trained, args.step, args.output, args.source_revision)
    else:
        write_catalog(args.output, dict(args.replace))


if __name__ == "__main__":
    main()
