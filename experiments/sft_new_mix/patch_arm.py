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
import dataclasses
import hashlib
import json
import logging
from pathlib import Path

import draccus
import yaml
from levanter.store.cache import LEDGER_FILE_NAME
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.post_training.long_loop.serving import SNOWBALL_CONTEXT_LIMIT
from experiments.post_training.long_loop.sft import (
    TOKENIZER_URI,
    fresh_sft_config,
    read_json,
    restore_frozen,
    validate_handoff,
)
from experiments.post_training.long_loop.sft_runtime.export_writer import ExportConfig, export_weights
from experiments.post_training.long_loop.sft_runtime.model import GrugModelConfig
from experiments.post_training.long_loop.sft_runtime.train import run_grug

logger = logging.getLogger(__name__)

EXPERIMENT_DIR = Path(__file__).parent / "patch_experiment"
LONG_LOOP_SPECS = Path(__file__).parents[1] / "post_training" / "long_loop" / "specs"
RECIPE_PATH = EXPERIMENT_DIR / "sft-262k-legacy.yaml"
INITIALIZER = "s3://marin-us-east-02a/marin/long-loop/base/2026.11.08.8"
CONTROL_CATALOG = "s3://marin-us-east-02a/marin/grug-sft/sft-pipe-repro/full-mixture/2026.10.24.2/cache_catalog.json"
WANDB_PROJECT = "sft-data-patch-11188"
PREEMPTION_RETRIES = 20
FAILURE_RETRIES = 2
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
    """Train one arm, resuming from its own latest checkpoint after a preemption.

    The long-loop's ``sft_run`` never resumes, so one evicted node restarts the run
    from the base. Arms here run at interactive priority beside production work,
    so the gang gets a preemption retry budget and restores from the checkpoints
    under ``output`` when any exist.
    """
    configure_coreweave_s3()
    recipe = arm_recipe(catalog, steps, replicas, arm)
    recipe["max_retries_preemption"] = PREEMPTION_RETRIES
    recipe["max_retries_failure"] = FAILURE_RETRIES
    # One evicted node fails its coscheduled siblings, and each counts as a task failure.
    recipe["max_task_failures"] = replicas * (PREEMPTION_RETRIES + FAILURE_RETRIES)
    config = fresh_sft_config(output=output, initializer=INITIALIZER, recipe=recipe, run_id=f"sft-patch-11188-{arm}")
    resumable = dataclasses.replace(config.trainer.trainer, load_checkpoint=None, load_checkpoint_path=None)
    config = dataclasses.replace(config, trainer=dataclasses.replace(config.trainer, trainer=resumable))
    (StoragePath(output) / "consumed-config.json").write_text(json.dumps(draccus.encode(config), indent=2))
    run_grug(config)
    metadata = read_json(prefix_join(output, f"checkpoints/step-{steps}/metadata.json"))
    if metadata.get("step") != steps or metadata.get("is_temporary") is not False:
        raise ValueError(f"SFT did not commit permanent step {steps}")


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
    configure_coreweave_s3()
    catalog = json.loads(StoragePath(CONTROL_CATALOG).read_text())
    entries = catalog["splits"]["train"]
    for name, cache_dir in replacements.items():
        if name not in entries:
            raise ValueError(f"{name} is not a train component of the control catalog")
        ledger = json.loads((StoragePath(cache_dir) / LEDGER_FILE_NAME).read_text())
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
