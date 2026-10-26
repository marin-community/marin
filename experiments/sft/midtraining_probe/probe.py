# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT midtraining probe: fine-tune every Delphi 3e18 base checkpoint on four sets and measure validation loss.

One :func:`sft_bundle_step` per base checkpoint fine-tunes it on every (SFT set, learning rate, SFT seed) in its
bundle, sequentially on one TPU slice (each run in a fresh process, so JAX starts clean), and records per run the
final evaluation: the set's completion-masked validation loss (``eval/<set>/loss``) and the forgetting evals, the
seven paper Uncheatable components and Paloma C4 (``eval/uncheatable_eval/<x>/bpb``, ``eval/paloma/c4_en/bpb``),
computed in the same run from the swarm's validation caches. Results land in ``results.json`` under each run and
the bundle; a finished run is skipped on retry.

The SFT recipe follows Liu, Neubig and Xiong (2025): 4 epochs, 64 examples per step (one example per sequence, no
packing), 10% warmup then cosine to zero, AdamW, peak learning rate per set from a grid on the proportional-mixture
checkpoint (``--panel grid``) and then frozen for every arm. The training config is the ``experiments/sft/launcher``
one (``_levanter_train_config``) with the seed, evaluation cadence and export turned per run.

Usage (from a CPU coordinator job in us-east5)::

    python -m experiments.sft.midtraining_probe.probe --panel grid --accelerator v6e-8
    python -m experiments.sft.midtraining_probe.probe --panel july --lr gsm8k=1e-4 --lr sciq=1e-4 \\
        --lr pycode=3e-4 --lr norobots=1e-4 --accelerator v6e-8 --max-concurrent 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import fsspec
import pandas as pd
from fray.types import ResourceConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.distributed import DistributedConfig
from levanter.models.lm_model import LmConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, lower
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.training.training import TrainLmOnPodConfig
from rigging.filesystem import marin_prefix_for_region
from rigging.filesystem.storage_path import prefix_join

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as swarm
from experiments.llama import llama3_tokenizer
from experiments.sft.launcher import HFModel, SFTSpec, _levanter_train_config
from experiments.sft.midtraining_probe.datasets import (
    BASE_CHAT_TEMPLATE,
    CHAT_FORMAT,
    PROBE_DATASETS,
    ProbeDataset,
    chat_cache_step,
    prepare_dataset_step,
    read_manifest,
)

logger = logging.getLogger(__name__)

EXPERIMENT_ROOT = "sft_midtraining_probe_20261014"
SPEC_VERSION = "2026.10.14"
INVENTORY = Path(__file__).resolve().parents[3] / ".agents/projects/sft_midtraining_probe/checkpoint_inventory.csv"
ARMS_RUN_SPECS = (
    Path(__file__).resolve().parents[3]
    / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/sft_midtraining_probe_20261014"
    / "launch_dry_run/run_specs.json"
)
ARMS_EXPERIMENT = "pinlin_calvin_xu/data_mixture/delphi_sft_probe_arms_3e18_20261014"
PROPORTIONAL_RUN = "fit_000_baseline_proportional"
TOKENIZER = llama3_tokenizer
LLAMA3_EOS_TOKEN_ID = 128001
SEQ_LEN = 2048
BATCH_SIZE = 64
EPOCHS = 4
LR_GRID = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
GRID_SEED = 0
SFT_SEEDS = (1, 2)
WANDB_PROJECT = "sft-midtraining-probe"
UNCHEATABLE_COMPONENTS = (
    "ao3_english",
    "arxiv_computer_science",
    "arxiv_physics",
    "bbc_news",
    "github_cpp",
    "github_python",
    "wikipedia_english",
)
FORGETTING_SETS = ("paloma/c4_en", *(f"uncheatable_eval/{name}" for name in UNCHEATABLE_COMPONENTS))
TPU_ZONES = {"v6e-8": "us-east5-b", "v6e-4": "us-east5-b", "v5p-8": "us-east5-a"}
TPU_REGION = "us-east5"
PANELS = ("grid", "arms", "frontier", "july", "swarm")


@dataclass(frozen=True)
class BaseCheckpoint:
    """One base model to fine-tune: an HF export of a Delphi 3e18 run."""

    panel: str
    run_name: str
    checkpoint: str
    policy_class: str
    data_seed: int
    trainer_seed: int
    arm: str = ""
    design_id: str = ""


@dataclass(frozen=True)
class SftRun:
    """One fine-tune inside a bundle."""

    name: str
    dataset: str
    learning_rate: float
    seed: int
    num_train_steps: int
    steps_per_eval: int


@dataclass(frozen=True)
class SftBundleConfig:
    output_path: str
    base: BaseCheckpoint
    chat_caches: Mapping[str, str]
    forgetting_components: Mapping[str, DatasetComponent]
    runs: tuple[SftRun, ...]
    tpu_type: str
    tpu_zone: str
    export_hf: bool
    keep_checkpoints: bool
    wandb_group: str
    # Runtime choice (excluded from the step identity): which SFT seeds to run now. Runs at other seeds are
    # skipped, so a panel can drop to one seed without changing its steps.
    active_seeds: tuple[int, ...] = (0, 1, 2)


def _optimizer(learning_rate: float) -> AdamConfig:
    return AdamConfig(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=1.0,
        weight_decay=0.0,
        lr_schedule="cosine",
        warmup=0.1,
        min_lr_ratio=0.0,
    )


def num_train_steps(n_train: int) -> int:
    return math.ceil(EPOCHS * n_train / BATCH_SIZE)


def _run_id(bundle: SftBundleConfig, run: SftRun) -> str:
    digest = hashlib.sha256(bundle.output_path.encode()).hexdigest()[:8]
    return f"sftp-{bundle.base.run_name}-{digest}-{run.name}"[:120]


def build_pod_config(bundle: SftBundleConfig, run: SftRun, output_path: str) -> TrainLmOnPodConfig:
    """The launcher's Levanter config for one fine-tune, with the run's seed, cadence, export and forgetting evals."""
    cache_dir = bundle.chat_caches[run.dataset]
    sft_component = DatasetComponent(
        source=UrlDatasetSourceConfig(
            train_urls=[], validation_urls=[], cache_dir=cache_dir, format=CHAT_FORMAT, tags=[run.dataset]
        ),
        cache_dir=cache_dir,
        format=CHAT_FORMAT,
        split="train",
    )
    components = {run.dataset: sft_component, **dict(bundle.forgetting_components)}
    data = LmDataConfig(
        tokenizer=TOKENIZER,
        chat_template=BASE_CHAT_TEMPLATE,
        enforce_eos=True,
        auto_build_caches=False,
        components=components,
        train_weights={run.dataset: 1.0, **{name: 0.0 for name in bundle.forgetting_components}},
        shuffle=True,
    )
    spec = SFTSpec(
        name=f"{EXPERIMENT_ROOT}/{bundle.base.panel}/{bundle.base.run_name}/{run.name}",
        version=SPEC_VERSION,
        model=HFModel(model_ref=bundle.base.checkpoint, tokenizer_path=TOKENIZER, eos_token_ids=(LLAMA3_EOS_TOKEN_ID,)),
        chat_template=BASE_CHAT_TEMPLATE,
        datasets=[],
        optimizer=_optimizer(run.learning_rate),
        seq_len=SEQ_LEN,
        pack=False,
        batch_size=BATCH_SIZE,
        num_train_steps=run.num_train_steps,
        wandb_project=WANDB_PROJECT,
    )
    train_config = _levanter_train_config(
        spec,
        model_config=LmConfig.get_choice_class("qwen3")(),
        data_config=data,
        num_train_steps=run.num_train_steps,
        initialize_from_hf=bundle.base.checkpoint,
        initialize_model_from_checkpoint_path=None,
        use_hf_model_config=True,
        eos_token_ids=(LLAMA3_EOS_TOKEN_ID,),
        gpu_allocator=False,
    )
    trainer = replace(
        train_config.trainer,
        seed=run.seed,
        id=_run_id(bundle, run),
        steps_per_eval=run.steps_per_eval,
        # Single-host slice, one process: no jax.distributed, and no Iris job-completion hook at the child's exit.
        distributed=DistributedConfig(initialize_jax_distributed=False),
        allow_nondivisible_batch_size=True,
        tracker=WandbConfig(
            entity="marin-community",
            project=WANDB_PROJECT,
            group=bundle.wandb_group,
            tags=[
                "sft-midtraining-probe",
                f"panel={bundle.base.panel}",
                f"base={bundle.base.run_name}",
                f"dataset={run.dataset}",
                f"lr={run.learning_rate:g}",
                f"sft_seed={run.seed}",
                f"policy={bundle.base.policy_class}",
            ],
        ),
    )
    train_config = replace(
        train_config,
        trainer=trainer,
        data_seed=run.seed,
        hf_save_steps=run.num_train_steps if bundle.export_hf else None,
    )
    resources = ResourceConfig.with_tpu(bundle.tpu_type, regions=[TPU_REGION], zone=bundle.tpu_zone)
    return TrainLmOnPodConfig(
        train_config=train_config,
        resources=resources,
        output_path=output_path,
        env_vars={"MARIN_PREFIX": marin_prefix_for_region(TPU_REGION), swarm.SKIP_EVAL_HARNESS_ENV_VAR: "1"},
        auto_build_caches=False,
    )


REPO_ROOT = Path(__file__).resolve().parents[3]


def _train_one(pod_config: TrainLmOnPodConfig) -> int:
    """Run one fine-tune in a fresh interpreter (fresh JAX runtime); returns its exit code.

    The child is ``python -m experiments.sft.midtraining_probe.probe --train-one <pickled pod config>`` rather than a
    multiprocessing target: the task's cloudpickled entry point cannot look the target function up by name.
    """
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as handle:
        pickle.dump(pod_config, handle)
        path = handle.name
    try:
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT)
            + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
        }
        completed = subprocess.run(
            [sys.executable, "-m", "experiments.sft.midtraining_probe.probe", "--train-one", path],
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )
        return completed.returncode
    finally:
        os.unlink(path)


def _train_one_main(path: str) -> None:
    from marin.training.training import run_levanter_train_lm  # noqa: PLC0415  # the child process only

    with open(path, "rb") as handle:
        pod_config = pickle.load(handle)
    run_levanter_train_lm(pod_config)


def _eval_history_exists(output_path: str) -> bool:
    fs, root = fsspec.core.url_to_fs(prefix_join(output_path, "checkpoints/eval_metrics.jsonl"))
    return fs.exists(root)


def _read_eval_history(output_path: str) -> list[dict]:
    metrics_file = prefix_join(output_path, "checkpoints/eval_metrics.jsonl")
    with fsspec.open(metrics_file, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _delete_checkpoints(output_path: str) -> None:
    fs, root = fsspec.core.url_to_fs(prefix_join(output_path, "checkpoints"))
    for path in fs.glob(f"{root}/step-*"):
        fs.rm(path, recursive=True)


def run_sft_bundle(config: SftBundleConfig) -> Artifact:
    """Fine-tune the bundle's base checkpoint on every run, one fresh process at a time, and collect the evals."""
    results = {}
    for run in config.runs:
        if run.seed not in config.active_seeds:
            logger.info("Skipping %s: seed %d is not active in this submission", run.name, run.seed)
            continue
        run_output = prefix_join(config.output_path, run.name)
        result_path = prefix_join(run_output, "results.json")
        fs, _ = fsspec.core.url_to_fs(result_path)
        if fs.exists(result_path):
            with fsspec.open(result_path, "r") as handle:
                results[run.name] = json.load(handle)
            logger.info("Skipping %s: already finished", run.name)
            continue
        # Levanter numbers steps from 0, so the final forced evaluation reports step num_train_steps - 1.
        final_step = run.num_train_steps - 1
        history = _read_eval_history(run_output) if _eval_history_exists(run_output) else []
        if history and int(history[-1].get("step", -1)) >= final_step:
            logger.info("Reusing the finished evaluation of %s (no results.json yet)", run.name)
        else:
            logger.info(
                "Fine-tuning %s on %s (lr %g, seed %d, %d steps)",
                config.base.run_name,
                run.dataset,
                run.learning_rate,
                run.seed,
                run.num_train_steps,
            )
            pod_config = build_pod_config(config, run, run_output)
            exit_code = _train_one(pod_config)
            if exit_code != 0:
                raise RuntimeError(f"{run.name} on {config.base.run_name} exited with code {exit_code}")
            history = _read_eval_history(run_output)
        final = history[-1]
        if int(final.get("step", -1)) < final_step:
            raise RuntimeError(f"{run.name}: final evaluation at step {final.get('step')} < {final_step}")
        result = {"run": asdict(run), "base": asdict(config.base), "final": final, "history": history}
        with fsspec.open(result_path, "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        if not config.keep_checkpoints:
            _delete_checkpoints(run_output)
        results[run.name] = result
    with fsspec.open(prefix_join(config.output_path, "results.json"), "w") as handle:
        json.dump({"base": asdict(config.base), "runs": results}, handle, indent=2, sort_keys=True)
    return Artifact(path=config.output_path)


def sft_bundle_step(
    base: BaseCheckpoint,
    runs: tuple[SftRun, ...],
    *,
    chat_caches: Mapping[str, ArtifactStep],
    forgetting_components: Mapping[str, DatasetComponent],
    tpu_type: str,
    export_hf: bool,
    keep_checkpoints: bool,
    wandb_group: str,
    name_suffix: str = "",
    active_seeds: tuple[int, ...] = (0, 1, 2),
) -> ArtifactStep[Artifact]:
    resources = ResourceConfig.with_tpu(tpu_type, regions=[TPU_REGION], zone=TPU_ZONES[tpu_type])
    datasets_used = sorted({run.dataset for run in runs})
    deps = tuple(chat_caches[name] for name in datasets_used)

    def build_config(ctx: StepContext) -> SftBundleConfig:
        return SftBundleConfig(
            output_path=ctx.output_path,
            base=base,
            chat_caches={name: ctx.artifact_path(chat_caches[name]) for name in datasets_used},
            forgetting_components=dict(forgetting_components),
            runs=runs,
            tpu_type=ctx.runtime_arg("tpu_type"),
            tpu_zone=ctx.runtime_arg("tpu_zone"),
            export_hf=export_hf,
            keep_checkpoints=keep_checkpoints,
            wandb_group=wandb_group,
            active_seeds=tuple(ctx.runtime_arg("sft_seeds")),
        )

    return ArtifactStep(
        name=f"{EXPERIMENT_ROOT}/{base.panel}/{base.run_name}" + (f"/{name_suffix}" if name_suffix else ""),
        version=SPEC_VERSION,
        artifact_type=Artifact,
        run=remote(run_sft_bundle, resources=resources, max_retries_failure=2, max_task_failures=2),
        build_config=build_config,
        deps=deps,
        runtime_args={"tpu_type": tpu_type, "tpu_zone": TPU_ZONES[tpu_type], "sft_seeds": active_seeds},
    )


def load_inventory(path: Path = INVENTORY) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["exists"].astype(str).str.lower() == "true"]
    return frame


def _checkpoint(row: pd.Series, panel: str, **extra: str) -> BaseCheckpoint:
    return BaseCheckpoint(
        panel=panel,
        run_name=str(row["run_name"]),
        checkpoint=str(row["checkpoint_uri"]),
        policy_class=str(row["policy_class"]),
        data_seed=int(row["data_seed"]),
        trainer_seed=int(row["trainer_seed"]),
        **extra,
    )


def _arm_checkpoints(prefix: str) -> list[BaseCheckpoint]:
    """Arm A from the inventory plus the fifteen new arm runs, whose HF exports live under the arms experiment."""
    inventory = load_inventory()
    arm_a = inventory[inventory["anchor_or_design_id"] == "lwspu_t9_snc_cap08"]
    checkpoints = [_checkpoint(row, "arms", arm="A") for _, row in arm_a.iterrows()]
    specs = json.loads(ARMS_RUN_SPECS.read_text())
    fs = fsspec.filesystem("gcs")
    for spec in specs:
        pattern = (
            f"{prefix}/{ARMS_EXPERIMENT}/{spec['run_name']}-*"
            f"/hf/step-{spec['expected_checkpoint_step']}/model.safetensors"
        )
        hits = fs.glob(pattern.removeprefix("gs://"))
        if not hits:
            logger.warning("Arm run %s has no HF export yet; skipping", spec["run_name"])
            continue
        checkpoint = "gs://" + hits[0].rsplit("/", 1)[0]
        arm = spec["panel_source"].removeprefix("sft_probe_arm_")
        checkpoints.append(
            BaseCheckpoint(
                panel="arms",
                run_name=spec["run_name"],
                checkpoint=checkpoint,
                policy_class="two_phase" if arm != "C" else "one_phase",
                data_seed=int(spec["data_seed"]),
                trainer_seed=int(spec["trainer_seed"]),
                arm=arm,
            )
        )
    return checkpoints


def panel_checkpoints(panel: str, prefix: str, *, accelerator: str = "") -> list[BaseCheckpoint]:
    inventory = load_inventory()
    if panel == "grid":
        # The grid may be raced on several pools (distinct step names); the collector reads every grid_* panel.
        row = inventory[inventory["run_name"] == PROPORTIONAL_RUN].iloc[0]
        return [_checkpoint(row, f"grid_{accelerator.replace('-', '')}", design_id="baseline_proportional")]
    if panel == "arms":
        return _arm_checkpoints(prefix)
    if panel == "frontier":
        rows = inventory[inventory["panel"] == "p6_table9_two_phase_frontier"]
        return [_checkpoint(row, "frontier", design_id=str(row["anchor_or_design_id"])) for _, row in rows.iterrows()]
    if panel == "july":
        # Launch order = priority: the Table-9 anchor first (the primary objective), and within an anchor the tied
        # controls, the Dolmino-late continuum and the handcrafted recipes before the 96 balanced partitions.
        rows = inventory[inventory["panel"] == "p3_aggressive_phase_asymmetry"].copy()
        anchor_rank = {"table9_frontier": 0, "uncheatable_frontier": 1}
        family_rank = {
            "center_control": 0,
            "dolmino_late_continuum": 1,
            "handcrafted_late_quality": 2,
            "balanced_partition": 3,
        }
        rows["_order"] = rows["anchor_or_design_id"].map(anchor_rank) * 10 + rows["contrast_family"].map(family_rank)
        rows = rows.sort_values(["_order", "run_name"])
        return [_checkpoint(row, "july", design_id=str(row["anchor_or_design_id"])) for _, row in rows.iterrows()]
    if panel == "swarm":
        two_phase = inventory[(inventory["panel"] == "p1_two_phase_swarm") & (inventory["policy_class"] == "two_phase")]
        designs = set(two_phase["anchor_or_design_id"])
        one_phase = inventory[
            (inventory["panel"] == "p2_one_phase_swarm") & inventory["anchor_or_design_id"].isin(designs)
        ]
        if len(one_phase) != len(two_phase):
            raise ValueError(f"Swarm pairs are incomplete: {len(two_phase)} two-phase vs {len(one_phase)} single-phase")
        # Launch order = priority: designs with the largest phase shift first, each pair (two-phase then its
        # single-phase average) adjacent so paired deltas land together.
        rows = pd.concat([two_phase, one_phase]).copy()
        design_tv = two_phase.set_index("anchor_or_design_id")["phase_tv"]
        rows["_tv"] = rows["anchor_or_design_id"].map(design_tv)
        rows = rows.sort_values(["_tv", "anchor_or_design_id", "policy_class"], ascending=[False, True, False])
        return [_checkpoint(row, "swarm", design_id=str(row["anchor_or_design_id"])) for _, row in rows.iterrows()]
    raise ValueError(f"Unknown panel {panel!r}")


def _dataset_runs(
    dataset: ProbeDataset,
    n_train: int,
    *,
    learning_rates: Sequence[float],
    seeds: Sequence[int],
    steps_per_eval: int | None,
) -> list[SftRun]:
    steps = num_train_steps(n_train)
    return [
        SftRun(
            name=f"{dataset.slug}_lr{lr:g}_s{seed}",
            dataset=dataset.slug,
            learning_rate=lr,
            seed=seed,
            num_train_steps=steps,
            steps_per_eval=steps_per_eval or steps + 1,
        )
        for lr in learning_rates
        for seed in seeds
    ]


def forgetting_components() -> dict[str, DatasetComponent]:
    steps = swarm._default_validation_sets(tokenizer=TOKENIZER)
    return {name: step_to_lm_mixture_component(steps[name], include_raw_paths=False) for name in FORGETTING_SETS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=PANELS, required=True)
    parser.add_argument("--accelerator", default="v6e-8", choices=sorted(TPU_ZONES))
    parser.add_argument(
        "--lr", action="append", default=[], help="<dataset>=<peak lr>; required for every panel but grid"
    )
    parser.add_argument("--datasets", default=",".join(d.slug for d in PROBE_DATASETS))
    parser.add_argument("--only", default=None, help="Regex on base run names")
    parser.add_argument(
        "--sft-seeds", default=None, help="Comma-separated SFT seeds to run now (default: all of the panel's)"
    )
    parser.add_argument("--max-concurrent", type=int, default=8, help="Bundles in flight (one TPU job each)")
    parser.add_argument("--export-hf", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Build the datasets and chat caches, then stop")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) == 3 and sys.argv[1] == "--train-one":
        _train_one_main(sys.argv[2])
        return
    args = parse_args()
    prefix = marin_prefix_for_region(TPU_REGION)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match {prefix!r}")
    os.environ["MARIN_PREFIX"] = prefix

    selected = [d for d in PROBE_DATASETS if d.slug in set(args.datasets.split(","))]
    prepared = {d.slug: prepare_dataset_step(d) for d in selected}
    caches = {d.slug: chat_cache_step(d, prepared[d.slug]) for d in selected}
    if args.prepare_only or not args.dry_run:
        StepRunner().run([lower(step) for step in caches.values()], max_concurrent=len(caches))
        if args.prepare_only:
            return
    manifests = {} if args.dry_run else {slug: read_manifest(prepared[slug].path(prefix)) for slug in prepared}

    if args.panel == "grid":
        learning_rates = {d.slug: LR_GRID for d in selected}
        seeds: tuple[int, ...] = (GRID_SEED,)
        steps_per_eval: int | None = 100
    else:
        learning_rates = {}
        for item in args.lr:
            slug, value = item.split("=")
            learning_rates[slug] = (float(value),)
        missing = [d.slug for d in selected if d.slug not in learning_rates]
        if missing:
            raise ValueError(f"--lr is required for {missing}")
        seeds = SFT_SEEDS
        steps_per_eval = None

    bases = panel_checkpoints(args.panel, prefix, accelerator=args.accelerator)
    if args.only:
        bases = [base for base in bases if re.search(args.only, base.run_name)]
    forgetting = {} if args.dry_run else forgetting_components()
    steps = []
    total_runs = 0
    for base in bases:
        # The grid bundles one dataset per TPU job so its five learning rates run in parallel across sets; every
        # other panel bundles all of a checkpoint's fine-tunes into one job.
        groups = [[dataset] for dataset in selected] if args.panel == "grid" else [list(selected)]
        for group in groups:
            runs: list[SftRun] = []
            for dataset in group:
                n_train = manifests[dataset.slug]["n_train"] if manifests else 1
                runs.extend(
                    _dataset_runs(
                        dataset,
                        n_train,
                        learning_rates=learning_rates[dataset.slug],
                        seeds=seeds,
                        steps_per_eval=steps_per_eval,
                    )
                )
            total_runs += len(runs)
            steps.append(
                sft_bundle_step(
                    base,
                    tuple(runs),
                    chat_caches=caches,
                    forgetting_components=forgetting,
                    tpu_type=args.accelerator,
                    export_hf=args.export_hf,
                    keep_checkpoints=args.keep_checkpoints,
                    wandb_group=f"{EXPERIMENT_ROOT}/{args.panel}",
                    name_suffix=group[0].slug if len(groups) > 1 else "",
                    active_seeds=tuple(int(x) for x in args.sft_seeds.split(",")) if args.sft_seeds else tuple(seeds),
                )
            )
    logger.info(
        "Panel %s: %d base checkpoints, %d bundles, %d fine-tunes", args.panel, len(bases), len(steps), total_runs
    )
    if args.dry_run:
        for step in steps:
            config = step.build_config(StepContext.for_fingerprint(("tpu_type", "tpu_zone"), step.deps))
            print(
                step.name, config.base.checkpoint, [run.name for run in config.runs][:3], "...", len(config.runs), "runs"
            )
        return
    StepRunner().run([lower(step) for step in steps], max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
