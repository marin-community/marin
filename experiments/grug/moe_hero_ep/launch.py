# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GB200 launcher for the EP64 MoE hero configuration."""

import dataclasses
import os
from datetime import timedelta

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture, tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem.storage_path import prefix_join

from experiments.datasets.paloma import paloma_datasets
from experiments.grug.moe_hero_ep.heuristic import HERO_MODEL, build_hero_configs
from experiments.grug.moe_hero_ep.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    MasterParamMode,
    WatchMode,
    run_grug,
)
from experiments.llama import llama3_tokenizer

DEFAULT_HERO_STEPS = 25
DEFAULT_WANDB_PROJECT = "marin_moe"
HERO_EP_BATCH_SIZE = 1024
HERO_EP_NODES = 16
HERO_GPUS_PER_NODE = 4
HERO_EP_EXPERT_AXIS_SIZE = HERO_EP_NODES * HERO_GPUS_PER_NODE
HERO_PROCESSES_PER_TASK = 1
HERO_MIXED_PRECISION = "params=bfloat16,compute=bfloat16,output=bfloat16"
# Keep MuonH state on pinned host memory to leave room for the pooled all-to-all buffers.
HERO_OFFLOAD_OPT_STATE = True
HERO_WATCH_INTERVAL = 0
HERO_CHECKPOINT_INTERVAL = timedelta(minutes=15)

_SLIMPAJAMA_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="64g")
_SLIMPAJAMA_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel")

# Pinned JAX nightly for arms that need post-0.11.0 XLA (e.g. the device-initiated ragged
# all-to-all kernel from openxla#46116). This exact set ran cleanly against NCCL 2.30.7 on
# the cw-us-east-08a GB200 workers (marin#8108, MOK-JAX arms, 2026-08-10).
JAX_NIGHTLY_WHEELS_20260809: tuple[str, ...] = (
    "https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/jax/jax-0.11.1.dev20260809-py3-none-any.whl",
    "https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/jaxlib/jaxlib-0.11.1.dev20260809-cp312-cp312-manylinux_2_27_aarch64.whl",
    "https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/jax-cuda13-plugin/jax_cuda13_plugin-0.11.1.dev20260809-cp312-cp312-manylinux_2_27_aarch64.whl",
    "https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/jax-cuda13-pjrt/jax_cuda13_pjrt-0.11.1.dev20260809-py3-none-manylinux_2_27_aarch64.whl",
)


def pjrt_wheel_install_script(wheel_url: str) -> str:
    """Task setup script installing the pinned nightly with a substituted PJRT wheel.

    Installs the three stock dev20260809 wheels plus a self-built ``jax-cuda13-pjrt`` from
    object storage, so the runtime matches ``--jax-nightly`` except for the PJRT patch.
    """
    stock_wheels = " ".join(f'"{url}"' for url in JAX_NIGHTLY_WHEELS_20260809 if "pjrt" not in url)
    return f"""set -e
: "${{IRIS_WORKDIR:?}}"
: "${{IRIS_VENV:?}}"
wheel_dir="$IRIS_WORKDIR/.pjrt-wheel"
rm -rf "$wheel_dir"
mkdir -p "$wheel_dir"
echo 'downloading patched PJRT wheel'
"$IRIS_VENV/bin/python" - <<'PY'
import os
from pathlib import Path

import fsspec

wheel_url = {wheel_url!r}
wheel_dir = Path(os.environ["IRIS_WORKDIR"]) / ".pjrt-wheel"
filesystem, remote_path = fsspec.core.url_to_fs(wheel_url)
filesystem.get(remote_path, str(wheel_dir / remote_path.rsplit("/", 1)[1]))
PY
echo 'installing pinned nightly with patched PJRT'
uv pip install --python "$IRIS_VENV/bin/python" --no-deps --reinstall {stock_wheels} "$wheel_dir"/*.whl
"$IRIS_VENV/bin/python" - <<'PY'
from importlib.metadata import version

print("jax", version("jax"), "pjrt", version("jax-cuda13-pjrt"))
PY
# nvidia-nccl-cu12 (torch dep) and nvidia-nccl-cu13 (jax dep) both install
# nvidia/nccl/lib/libnccl.so.2; the last writer wins per node, so a fresh env can
# end up with different NCCL bootstrap wire formats across ranks (d08: rank 0 at
# 2.28.9 expected 172-byte messages, peers at 2.30.7 sent 176). Force cu13 last.
echo 'forcing nvidia-nccl-cu13 to own libnccl.so.2'
uv pip uninstall --python "$IRIS_VENV/bin/python" nvidia-nccl-cu12 || true
uv pip install --python "$IRIS_VENV/bin/python" --no-deps --reinstall nvidia-nccl-cu13==2.30.7
"$IRIS_VENV/bin/python" - <<'PY'
import ctypes
import hashlib
import importlib
from pathlib import Path

lib_path = Path(importlib.import_module("nvidia.nccl").__path__[0]) / "lib" / "libnccl.so.2"
lib = ctypes.CDLL(str(lib_path))
version = ctypes.c_int()
lib.ncclGetVersion(ctypes.byref(version))
digest = hashlib.sha256(lib_path.read_bytes()).hexdigest()[:16]
print(f"libnccl {{lib_path}} version_code={{version.value}} sha256={{digest}}")
PY
"""


# Held-out sets, added at weight 0 so they surface as tagged eval sets. The hero trains on
# llama3-tokenized SlimPajama, so these must carry the same tokenizer.
#
# Paloma only, deliberately. `paloma_dataset` and `uncheatable_dataset` both hardcode a `-llama3`
# suffix in the cache name while taking an arbitrary `tokenizer` argument, so callers asking for
# different tokenizers collide on one cache identity and whoever materializes first wins. The
# uncheatable caches under that name currently hold marin-tokenizer data, which fails the mixture's
# single-tokenizer check. Paloma is also the suite the scaling-law scoring uses, so dropping
# uncheatable costs nothing here.
def _validation_datasets() -> list[ArtifactStep[TokenizedCache]]:
    return list(paloma_datasets(tokenizer=llama3_tokenizer).values())


def _slimpajama_6b_dataset() -> ArtifactStep[TokenizedCache]:
    return tokenized(
        "slimpajama-6b",
        source="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        resources=_SLIMPAJAMA_TOKENIZE_RESOURCES,
        version="2026.06.28",
    )


class HeroThroughputResult(Artifact):
    """Result of the rack-scale throughput hero run.

    The run mirrors its tracker metrics to the output path. It writes no checkpoint by default.
    Checkpoint restore has a known memory-kind mismatch with the offloaded optimizer state.
    """


def build_hero_run(
    *,
    run_id: str,
    dp_racks: int,
    num_steps: int,
    schedule_steps: int | None = None,
    seed: int = 0,
    batch_size: int = HERO_EP_BATCH_SIZE,
    num_experts: int | None = None,
    num_experts_per_token: int | None = None,
    intermediate_dim: int | None = None,
    capacity_factor: float | None = None,
    latent_dim: int | None = None,
    eval_every: int = 0,
    save_checkpoints: bool = False,
    checkpoint_interval: timedelta = HERO_CHECKPOINT_INTERVAL,
    checkpoint_path: str | None = None,
    watch_interval: int = HERO_WATCH_INTERVAL,
    watch_mode: WatchMode = WatchMode.INLINE,
    profile_steps: int = 0,
    profile_start_step: int = 5,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build the EP64 hero throughput run.

    The overrides sweep expert count, expert width, routed top-k, and routing capacity from the
    hero spec. They keep the hidden dimension, so the compute-scaled optimizer values stay
    comparable across a sweep. ``None`` keeps the hero value.

    ``batch_size`` is the global batch across all data-parallel racks. It does not scale with
    ``dp_racks``. ``batch_size`` and ``schedule_steps`` change the token budget for the heuristic.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if dp_racks <= 0:
        raise ValueError(f"dp_racks must be positive, got {dp_racks}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if checkpoint_interval <= timedelta(0):
        raise ValueError(f"checkpoint_interval must be positive, got {checkpoint_interval}")
    if profile_steps < 0:
        raise ValueError(f"profile_steps must be non-negative, got {profile_steps}")
    if profile_start_step < 0:
        raise ValueError(f"profile_start_step must be non-negative, got {profile_start_step}")
    if profile_steps > 0 and profile_start_step >= num_steps:
        raise ValueError(f"profile_start_step must be less than num_steps={num_steps}, got {profile_start_step}")
    # `schedule_steps` sets the whole learning-rate schedule; `num_steps` sets how far the run goes.
    # Both matter, and they enter in different places. The optimizer heuristic scales learning rate,
    # adam_lr, and epsilon from a token budget (`num_train_steps * batch * seq`), which fixes the
    # peak. Warmup and decay are *fractions* of `TrainerConfig.num_train_steps`, so that field has to
    # carry the schedule length too -- passing `num_steps` there warms up in `0.01 * num_steps` and
    # decays to `min_lr_ratio` by the end of the short run, which is a whole miniature schedule
    # rather than the head of a long one. Default keeps the two equal, which is the previous behavior.
    if schedule_steps is not None and schedule_steps <= 0:
        raise ValueError(f"schedule_steps must be positive, got {schedule_steps}")
    if schedule_steps is not None and schedule_steps < num_steps:
        raise ValueError(f"schedule_steps={schedule_steps} must be at least num_steps={num_steps}")
    total_schedule_steps = schedule_steps if schedule_steps is not None else num_steps
    model, optimizer = build_hero_configs(
        num_train_steps=total_schedule_steps,
        batch_size=batch_size,
    )
    overrides = {
        name: value
        for name, value in (
            ("num_experts", num_experts),
            ("num_experts_per_token", num_experts_per_token),
            ("intermediate_dim", intermediate_dim),
            ("capacity_factor", capacity_factor),
            ("latent_dim", latent_dim),
        )
        if value is not None
    }
    if overrides:
        model = dataclasses.replace(model, **overrides)
    # A bank that is not divisible by the expert axis fails inside `moe_mlp`, which is after the rack
    # is already allocated and the workspace is built. Reject it here instead.
    if model.num_experts % HERO_EP_EXPERT_AXIS_SIZE != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by {HERO_EP_EXPERT_AXIS_SIZE}")
    local_experts = model.num_experts // HERO_EP_EXPERT_AXIS_SIZE
    if local_experts % model.num_expert_waves != 0:
        raise ValueError(
            f"local expert count={local_experts} must be divisible by num_expert_waves={model.num_expert_waves}"
        )
    if model.moe_implementation != "fixed_pooled_wave_all_to_all":
        raise AssertionError(f"unexpected hero MoE implementation: {model.moe_implementation}")
    if model.pooled_transport_capacity_factor is None:
        raise AssertionError("the pooled-wave hero requires a transport capacity factor")
    backend_tag = model.moe_implementation.replace("_", "-")
    capacity_tag = f"capacity-{model.capacity_factor:g}"
    transport_capacity_tag = f"transport-capacity-{model.pooled_transport_capacity_factor:g}"
    wave_tag = f"expert-waves-{model.num_expert_waves}"
    size_tag = f"e{model.num_experts}-i{model.intermediate_dim}"
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=HERO_OFFLOAD_OPT_STATE,
        master_param_mode=MasterParamMode.FP32_PINNED_HOST,
        watch_mode=watch_mode,
        # The default offloaded optimizer state has a known memory-kind mismatch during restore.
        save_checkpoints=save_checkpoints,
        expert_axis_size=HERO_EP_EXPERT_AXIS_SIZE,
        replica_axis_size=dp_racks,
        sharding_dump_path=None,
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=120,
        ram="850g",
        disk="1t",
        replicas=HERO_EP_NODES * dp_racks,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    slim = _slimpajama_6b_dataset()
    validation = _validation_datasets() if eval_every > 0 else []

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=seed,
            train_batch_size=batch_size,
            num_train_steps=total_schedule_steps,
            profiler=ProfilerConfig(
                enabled=profile_steps > 0,
                start_step=profile_start_step,
                num_steps=profile_steps,
                # One rank is enough for a step trace, and tracing all 64 multiplies the upload
                # without adding signal.
                process_index=0,
                profile_options=ProfileOptionsConfig(enable_hlo_proto=True),
            ),
            mp=jmp.get_policy(HERO_MIXED_PRECISION),
            tracker=WandbConfig(
                entity="marin-community",
                project=wandb_project,
                tags=[
                    "grug",
                    "moe",
                    "hero",
                    "ep",
                    backend_tag,
                    capacity_tag,
                    transport_capacity_tag,
                    wave_tag,
                    size_tag,
                    "gb200",
                    "MHEP",
                ],
                group="moe-hero-ep",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=watch_interval),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            # Levanter's default base path is pod-local, so a preempted run would have nothing to
            # resume from. `checkpoint_path` overrides this for runs targeting disposable storage.
            checkpointer=CheckpointerConfig(
                base_path=checkpoint_path or prefix_join(ctx.output_path, "checkpoints"),
                temporary_base_path=None,
                save_interval=checkpoint_interval,
                keep=None,
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=1,
            ),
        )
        return GrugRunConfig(
            model=model,
            data=mixture(ctx, {slim: 1.0}, validation=validation, shuffle=_SLIMPAJAMA_SHUFFLE),
            resources=ctx.runtime_arg("train_resources"),
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            # Off by default so a throughput run stays a throughput run. Turn it on to make a
            # run scoreable: comparing configs needs held-out loss, not train loss.
            eval=(
                GrugEvalConfig(steps_per_eval=eval_every, eval_ema=False, compute_bpb=True) if eval_every > 0 else None
            ),
            stop_after_steps=num_steps,
            processes_per_task=HERO_PROCESSES_PER_TASK,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=(slim, *validation),
        runtime_args={"train_resources": train_resources},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option(
    "--dp-racks",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Data-parallel NVL72 rack count. --batch-size stays global across all racks.",
)
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=DEFAULT_HERO_STEPS,
    show_default=True,
    help="Number of training steps.",
)
@click.option(
    "--schedule-steps",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Build the learning-rate schedule for this many steps instead of --num-steps. The optimizer "
        "heuristic scales its rates from the implied token budget, so this trains the head of a long "
        "run's schedule. Defaults to --num-steps."
    ),
)
@click.option(
    "--seed",
    type=int,
    default=0,
    help="Trainer seed. Vary it across otherwise identical runs to measure run-to-run variance.",
)
@click.option(
    "--num-experts",
    type=click.IntRange(min=1),
    default=HERO_MODEL.num_experts,
    show_default=True,
    help=(
        f"Override the routed expert count. The count must be divisible by {HERO_EP_EXPERT_AXIS_SIZE}, "
        f"and the local expert count must support {HERO_MODEL.num_expert_waves} waves."
    ),
)
@click.option(
    "--num-experts-per-token",
    type=click.IntRange(min=1),
    default=HERO_MODEL.num_experts_per_token,
    show_default=True,
    help="Override the routed top-k. Scales both active parameters and the EP dispatch buffers.",
)
@click.option(
    "--intermediate-dim",
    type=click.IntRange(min=1),
    default=HERO_MODEL.intermediate_dim,
    show_default=True,
    help="Override the routed expert width.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=HERO_EP_BATCH_SIZE,
    show_default=True,
    help="Global sequences per step. This value does not scale with --dp-racks.",
)
@click.option(
    "--latent-dim",
    type=click.IntRange(min=1),
    default=None,
    help="LatentMoE: run routed experts at this width. Divides all-to-all traffic by hidden/latent.",
)
@click.option(
    "--save-checkpoints/--no-save-checkpoints",
    default=False,
    show_default=True,
    help="Write checkpoints. Restore is not supported for the pinned-host optimizer state.",
)
@click.option(
    "--checkpoint-minutes",
    type=click.FloatRange(min=0, min_open=True),
    default=HERO_CHECKPOINT_INTERVAL.total_seconds() / 60,
    show_default=True,
    help="Wall-clock minutes between checkpoint writes.",
)
@click.option(
    "--checkpoint-path",
    default=None,
    help="Checkpoint output path, e.g. a marin_temp_bucket() path. Defaults to the step output path.",
)
@click.option(
    "--eval-every",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Run the paloma suite every N steps. 0 disables eval (throughput-only run).",
)
@click.option(
    "--watch-interval",
    type=click.IntRange(min=0),
    default=HERO_WATCH_INTERVAL,
    show_default=True,
    help="Steps between gradient and parameter norm logs. 0 disables norm logging.",
)
@click.option(
    "--watch-mode",
    type=click.Choice([mode.value for mode in WatchMode]),
    default=WatchMode.INLINE.value,
    show_default=True,
    help="Compute norms in the training step or in a separate forward and backward diagnostic step.",
)
@click.option(
    "--profile-steps",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Steps to trace with XProf on rank 0. 0 disables the profiler.",
)
@click.option(
    "--profile-start-step",
    type=click.IntRange(min=0),
    default=5,
    show_default=True,
    help="First traced step. Keep it past compile and warmup.",
)
@click.option(
    "--capacity-factor",
    type=click.FloatRange(min=0, min_open=True),
    default=HERO_MODEL.capacity_factor,
    show_default=True,
    help="Override the pooled receiver capacity factor.",
)
@build_options
def main(
    run_id: str,
    dp_racks: int,
    num_steps: int,
    schedule_steps: int | None,
    seed: int,
    batch_size: int,
    num_experts: int | None,
    num_experts_per_token: int | None,
    intermediate_dim: int | None,
    capacity_factor: float | None,
    latent_dim: int | None,
    save_checkpoints: bool,
    checkpoint_minutes: float,
    checkpoint_path: str | None,
    eval_every: int,
    watch_interval: int,
    watch_mode: str,
    profile_steps: int,
    profile_start_step: int,
) -> ArtifactStep[HeroThroughputResult]:
    return build_hero_run(
        run_id=run_id,
        dp_racks=dp_racks,
        num_steps=num_steps,
        schedule_steps=schedule_steps,
        seed=seed,
        batch_size=batch_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        intermediate_dim=intermediate_dim,
        capacity_factor=capacity_factor,
        latent_dim=latent_dim,
        save_checkpoints=save_checkpoints,
        checkpoint_interval=timedelta(minutes=checkpoint_minutes),
        checkpoint_path=checkpoint_path,
        eval_every=eval_every,
        watch_interval=watch_interval,
        watch_mode=WatchMode(watch_mode),
        profile_steps=profile_steps,
        profile_start_step=profile_start_step,
    )


if __name__ == "__main__":
    main()
