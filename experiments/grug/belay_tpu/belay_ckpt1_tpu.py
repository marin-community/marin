# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the frozen Ckpt 1 d512 recipe to Larry's GCS caches and TPU execution."""

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import uuid
from pathlib import Path

import draccus
import gcsfs
import levanter.tracker as tracker
from fray.cluster import ResourceConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import _initialize_global_tracker
from marin.execution.lazy import StepContext

from experiments.grug.belay_tpu import belay_ckpt1, launch, train
from experiments.grug.belay_tpu.tpu_kernel_profiles import KERNEL_PROFILES, OWNER_ATTENTION_PROFILES

SOURCE_COMMIT = "8415c3ce8411051c1e22bc13184c272b88ec2b6c"
ISSUE_URL = "https://github.com/WhenWen/astra-d8-research/issues/253"
DATA_ROOT = "gs://marin-us-central2"
TRAIN_CACHE = DATA_ROOT + "/datakit/hero_tok/v16384_shuf/train"
FULL_STEPS = belay_ckpt1.LADDER_RUNGS["d512"][1]
TOKENIZER_ROOT = DATA_ROOT + "/tokenizers/hero-bpe-v16384/hf-hub-1.21.0"
TPU_ROUNDING_FLAG = "--xla_allow_excess_precision=false"
TOKENIZER_HASHES = {
    "tokenizer.json": "adaba3dc92dff6eed642f10101e4183e0525437f3432d2ce004cd14232b350ae",
    "tokenizer_config.json": "1ff46b7c04e1ef0335286b733db65c663261132d0a9866842fd48020255ea05e",
}


def require_tpu_rounding() -> None:
    """Preserve declared BF16 rounding across forward and backward fusions."""
    if TPU_ROUNDING_FLAG not in os.environ.get("XLA_FLAGS", "").split():
        raise RuntimeError(f"Ckpt 1 TPU execution requires XLA_FLAGS={TPU_ROUNDING_FLAG}")


def verified_tokenizer(source: str = TOKENIZER_ROOT) -> str:
    """Load the frozen tokenizer on this execution host and verify every file."""
    remote = source.startswith("gs://")
    destination = Path("/tmp/belay-ckpt1-tokenizer") if remote else Path(source)
    destination.mkdir(parents=True, exist_ok=True)
    fs = gcsfs.GCSFileSystem(project="hai-gcp-models") if remote else None
    for name, expected in TOKENIZER_HASHES.items():
        path = destination / name
        data = fs.cat(f"{source}/{name}") if fs is not None else path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Ckpt 1 tokenizer checksum mismatch: {name}")
        if remote:
            path.write_bytes(data)
    return str(destination)


def settings(items: tuple[str, ...]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in items)


def build_config(
    run_id: str,
    output_path: str,
    tokenizer: str,
    tpu: str,
    seed: int,
    *,
    expert_axis_size: int = 16,
    local_experts: bool = False,
    kernel_profile: str = "source",
    pipeline_depth: int = 0,
) -> train.GrugRunConfig:
    """Materialize the upstream recipe, changing only documented TPU bindings."""
    if kernel_profile not in KERNEL_PROFILES or pipeline_depth not in (0, 1):
        raise ValueError("Unsupported Ckpt 1 kernel profile or pipeline depth")
    if kernel_profile != "source" and not local_experts:
        raise ValueError("Optimized kernel profiles require local experts")
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
    if local_experts:
        expert_axis_size = chips
    elif expert_axis_size not in (8, 16):
        raise ValueError("Ckpt 1 supports the verified 8x2 or candidate 16x1 expert-pool layout")
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
    attention = "pallas" if kernel_profile in OWNER_ATTENTION_PROFILES else "xla"
    if kernel_profile == "kda-owner-causal-sort-v1":
        attention = "causal"
    return dataclasses.replace(
        original,
        model=dataclasses.replace(
            original.model,
            moe_ragged_chunks=max(1, 16 // expert_axis_size),
            moe_local_gathered_experts=local_experts,
            tpu_attention=attention,
            # Save attention activations for backward.
            attn_res_remat_attention=False,
        ),
        data=dataclasses.replace(original.data, tokenizer=tokenizer),
        # EP8/chunks2 and EP16/chunks1 retain the same sixteen capacity pools
        # of 24 experts and global token order within each expert.
        trainer=dataclasses.replace(
            original.trainer,
            trainer=trainer,
            expert_axis_size=expert_axis_size,
            replica_axis_size=1,
            model_axis_size=chips // expert_axis_size,
            pipeline_depth=pipeline_depth,
        ),
        # Sonic is a GPU kernel. Scatter preserves dropless evaluation semantics;
        # equivalence remains a required TPU gate before claiming reproduction.
        # Upstream evaluates every example (max_examples_per_dataset=None).
        # Dropless examples are independent; grouping 16 instead of 8 examples
        # changes neither the evaluated examples nor their weights.
        eval=dataclasses.replace(
            original.eval, dropless_eval_moe_implementation="scatter", eval_batch_size=expert_axis_size
        ),
        processes_per_task=1,
        tpu_kernel_profile=kernel_profile,
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
    if model.moe_local_gathered_experts:
        if config.trainer.model_axis_size != 1 or config.trainer.expert_axis_size not in (16, 32):
            raise ValueError("Local-token Ckpt 1 requires 16 or 32 token shards with model axis one")
    elif config.trainer.expert_axis_size * model.moe_ragged_chunks != 16:
        raise ValueError("Ckpt 1 must retain sixteen capacity pools of 24 experts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--tokenizer", default=TOKENIZER_ROOT)
    parser.add_argument("--tpu", choices=("v4-32", "v4-64"), default="v4-32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-experts", action="store_true")
    parser.add_argument("--kernel-profile", choices=KERNEL_PROFILES, default="source")
    parser.add_argument("--pipeline-depth", type=int, choices=(0, 1), default=0)
    parser.add_argument("--resume-from", help="Restore complete training state; keep this run's checkpoints separate")
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--profile-start-step", type=int)
    parser.add_argument("--profile-steps", type=int, default=10)
    # Training deliberately uses the upstream complete schedule, not a shortened
    # optimizer schedule hidden inside a smoke configuration.
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--logging-only", action="store_true", help="Check online tracking without optimizer updates")
    args = parser.parse_args()
    tokenizer = verified_tokenizer(args.tokenizer)
    config = build_config(
        args.run_id,
        args.output_path,
        tokenizer,
        args.tpu,
        args.seed,
        local_experts=args.local_experts,
        kernel_profile=args.kernel_profile,
        pipeline_depth=args.pipeline_depth,
    )
    if args.resume_from:
        trainer = config.trainer.trainer
        trainer = dataclasses.replace(
            trainer,
            load_checkpoint=True,
            load_checkpoint_path=[*trainer.checkpoint_search_paths(args.run_id), args.resume_from],
            allow_partial_checkpoint=False,
        )
        config = dataclasses.replace(config, trainer=dataclasses.replace(config.trainer, trainer=trainer))
    if args.profile_start_step is not None:
        if args.profile_start_step < 0 or args.profile_steps <= 0:
            parser.error("Profiling requires a nonnegative start step and a positive number of steps")
        trainer = config.trainer.trainer
        trainer = dataclasses.replace(
            trainer,
            log_dir=args.config_output.parent / "logs",
            profiler=dataclasses.replace(
                trainer.profiler, enabled=True, start_step=args.profile_start_step, num_steps=args.profile_steps
            ),
        )
        config = dataclasses.replace(
            config,
            trainer=dataclasses.replace(
                config.trainer, trainer=trainer, hlo_dump_path=f"{args.output_path}/train-step.mlir"
            ),
        )
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
    if args.logging_only:
        _initialize_global_tracker(config.trainer.trainer.tracker, args.run_id)
        tracker.log_configuration(config)
        tracker.log_summary({"execution/logging_only_pass": True, "execution/optimizer_updates": 0})
        tracker.current_tracker().finish()
        print(json.dumps({"logging_only_pass": True, "optimizer_updates": 0}), flush=True)
        return
    if not args.run:
        return
    if os.environ.get("JAX_PLATFORMS") != "tpu,cpu":
        raise RuntimeError("Training requires JAX_PLATFORMS=tpu,cpu; CPU is needed for data shuffling")
    require_tpu_rounding()
    log_dir = Path(os.environ["IRIS_OUTPUT_DIR"]) / args.run_id / os.environ.get("HOSTNAME", "host")
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "native-metrics.jsonl")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("levanter.json_logger").addHandler(handler)
    train._run_grug_local(config)


if __name__ == "__main__":
    main()
