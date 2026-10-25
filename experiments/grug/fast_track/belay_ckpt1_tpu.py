# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the frozen Ckpt 1 d512 recipe to Larry's GCS caches and TPU execution."""

import argparse
import dataclasses
import json
import logging
import os
import uuid
from pathlib import Path

import draccus
from fray.cluster import ResourceConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import StepContext

from experiments.grug.fast_track import belay_ckpt1, launch, train

SOURCE_COMMIT = "8415c3ce8411051c1e22bc13184c272b88ec2b6c"
ISSUE_URL = "https://github.com/WhenWen/astra-d8-research/issues/253"
DATA_ROOT = "gs://marin-us-central2"
TRAIN_CACHE = DATA_ROOT + "/datakit/hero_tok/v16384_shuf/train"
FULL_STEPS = belay_ckpt1.LADDER_RUNGS["d512"][1]


def settings(items: tuple[str, ...]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in items)


def build_config(run_id: str, output_path: str, tokenizer: str, tpu: str, seed: int) -> train.GrugRunConfig:
    """Materialize the upstream recipe, changing only documented TPU bindings."""
    model_settings = settings((*belay_ckpt1.MODEL_SETTINGS, *belay_ckpt1.LADDER_MODEL_SETTINGS))
    model_settings.update(mla_num_heads="4", intermediate_dim="384", shared_expert_intermediate_dim="384")
    recipe = launch.build_h100_ladder_run(
        run_id=run_id,
        size="d512",
        recipe=launch.Recipe.KMA,
        version=belay_ckpt1.VERSION,
        num_steps=FULL_STEPS,
        batch_size=128,
        seed=seed,
        train_cache_dir=TRAIN_CACHE,
        save_checkpoints=True,
        model_settings=model_settings,
        optimizer_settings=settings(belay_ckpt1.OPTIMIZER_SETTINGS),
    )
    resources = ResourceConfig.with_tpu(tpu, cpu=16, ram="96g", disk="64g", regions=["us-central2"])
    # These are already-built caches. Context resolution only reads their records;
    # it never runs the raw download/tokenization dependencies.
    context = StepContext.for_run(
        output_path=output_path,
        prefix=DATA_ROOT,
        region="us-central2",
        runtime_args={"train_resources": resources},
        deps=recipe.deps,
    )
    original = recipe.build_config(context)
    chips = {"v4-32": 16, "v4-64": 32}[tpu]
    trackers = [
        JsonLoggerConfig(),
        WandbConfig(
            entity="marin-community",
            project="auto-understanding-arch",
            name=run_id,
            # A restarted process gets a separate curve; the native checkpoint
            # loader still restores the model/optimizer from the fixed run root.
            id=f"{run_id}-{os.environ.get('IRIS_ATTEMPT_UID', uuid.uuid4().hex[:12])}",
            mode="online",
            resume="never",
            group="belay-ckpt1-tpu-20261025",
            tags=["belay-ckpt1", "d512", tpu, f"seed-{seed}"],
            replicate_path=output_path,
            save_code=False,
        ),
    ]
    trainer = dataclasses.replace(
        original.trainer.trainer,
        tracker=trackers,
        progress_watchdog=ProgressWatchdogConfig(),
    )
    return dataclasses.replace(
        original,
        data=dataclasses.replace(original.data, tokenizer=tokenizer),
        # Receiver capacity and accepted token prefixes depend on the expert
        # group. Keep the source eight ranks instead of changing drop semantics.
        trainer=dataclasses.replace(
            original.trainer, trainer=trainer, expert_axis_size=8, replica_axis_size=1, model_axis_size=chips // 8
        ),
        # Sonic is a GPU kernel. Scatter preserves dropless evaluation semantics;
        # equivalence remains a required TPU gate before claiming reproduction.
        # Upstream evaluates every example (max_examples_per_dataset=None).
        eval=dataclasses.replace(original.eval, dropless_eval_moe_implementation="scatter"),
        processes_per_task=1,
    )


def validate_recipe(config: train.GrugRunConfig) -> None:
    model = config.model
    trainer = config.trainer.trainer
    expected = {
        "hidden_dim": 512,
        "num_layers": 6,
        "vocab_size": 16384,
        "max_seq_len": 4096,
        "num_experts": 384,
        "num_experts_per_token": 10,
        "num_shared_experts": 1,
        "mla_num_heads": 4,
        "intermediate_dim": 384,
        "shared_expert_intermediate_dim": 384,
    }
    for name, value in expected.items():
        if getattr(model, name) != value:
            raise ValueError(f"Ckpt 1 recipe drift: {name}")
    if (
        trainer.train_batch_size != 128
        or trainer.num_train_steps != 8821
        or trainer.progress_watchdog.create() is not None
    ):
        raise ValueError("Ckpt 1 budget or no-wall-clock-limit policy drift")
    if len(config.data.components) != 24 or config.data.train_weights["train"] != 1.0:
        raise ValueError("Expected one flat train cache and 23 evaluation components")
    if any(weight != 0 for name, weight in config.data.train_weights.items() if name != "train"):
        raise ValueError("Evaluation data entered the training mixture")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tpu", choices=("v4-32", "v4-64"), default="v4-32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-output", type=Path, required=True)
    # Training deliberately uses the upstream complete schedule, not a shortened
    # optimizer schedule hidden inside a smoke configuration.
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = build_config(args.run_id, args.output_path, args.tokenizer, args.tpu, args.seed)
    validate_recipe(config)
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    with args.config_output.open("w") as file:
        json.dump(draccus.encode(config), file, indent=2)
        file.write("\n")
    print(
        json.dumps(
            {
                "source_commit": SOURCE_COMMIT,
                "issue_url": ISSUE_URL,
                "config": str(args.config_output),
                "training_requested": args.run,
            }
        ),
        flush=True,
    )
    if not args.run:
        return
    if os.environ.get("JAX_PLATFORMS") != "tpu":
        raise RuntimeError("Training requires explicit JAX_PLATFORMS=tpu")
    log_dir = Path(os.environ["IRIS_OUTPUT_DIR"]) / args.run_id / os.environ.get("HOSTNAME", "host")
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "native-metrics.jsonl")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("levanter.json_logger").addHandler(handler)
    train._run_grug_local(config)


if __name__ == "__main__":
    main()
