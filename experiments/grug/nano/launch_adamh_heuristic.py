# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nano (modded-nanogpt) trial run with `experiments/grug/moe/heuristic.py`'s AdamH setup.

A third nano variant alongside `launch.py` (Muon) and `launch_adamh.py` (AdamH-ref).
Architecture is the same nano transformer; the optimizer is now sized from
`MoeAdamHHeuristic` rather than the modded-nanogpt ref's hardcoded constants.

Differences vs. `launch_adamh.py` (the ref-aligned AdamH run):

- **lm head goes to AdamH** (matches moe's mask: any 2-D weight outside `embed`
  is AdamH-routed). The ref puts the lm head on AdamW with a tiny `1/320` LR.
- **Single AdamW LR** for embed / norms / biases (`adam_lr` = ~0.0036 here);
  the ref splits into three separate rates (0.3 / 1/320 / 0.01).
- β1 / β2 / ε / lr come from the heuristic formulas, evaluated at this run's
  (tokens, batch, dim). β2 and ε scale with batch and total tokens; the ref
  hardcodes them.
- Schedule is `lr_schedule="linear"`, warmup=10% of steps, full linear decay
  to 0 over the remaining 90% — applied to both groups by a single multiplier.
  The ref uses two separate schedules (h: 5% warmup → 95% decay; aux: 60%
  plateau → 40% decay).
- `init_scheme="default"` with `zero_init_proj=False`: every 2-D weight starts
  non-zero so AdamH (Frobenius-preserving) can move them. We do *not* use the
  ref's Kaiming-with-multipliers init since the moe heuristic was developed
  against truncated-normal weights with constant `initializer_std`.
- `z_loss_weight=1e-4` and `max_grad_norm=1.0` (heuristic defaults).

AdamH itself is `experiments/grug/nano/optimizer.py:scale_by_adamh_safe` so
the divide-by-zero on the 2-D path (still present in the levanter / moe
copies) is fixed.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jax
import jmp
import optax
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DatasetComponent, LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import leaf_key_paths
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.training.training import temporary_checkpoint_base_path

from experiments.grug.moe.heuristic import MoeAdamHHeuristic
from experiments.grug.nano.model import NanoModelConfig
from experiments.grug.nano.optimizer import scale_by_adamh_safe
from experiments.grug.nano.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

# Match the AdamH-ref run length so loss curves are directly comparable.
ADAMH_HEURISTIC_TRAIN_STEPS = 4875


# ---- Optimizer ----


@OptimizerConfig.register_subclass("nano_adamh_heuristic")
@dataclass(frozen=True)
class NanoHeuristicAdamHConfig(OptimizerConfig):
    """AdamH+AdamW config with hyperparameters supplied by `MoeAdamHHeuristic`.

    The heuristic-derived `learning_rate`, `adam_lr`, `beta1`, `beta2`, and
    `epsilon` are stored as fields so `optax.inject_hyperparams` can swap the
    schedule-multiplied LRs in cleanly. Schedule-related fields (`warmup`,
    `decay`, `lr_schedule`, `min_lr_ratio`) are inherited from `OptimizerConfig`
    and consumed by `self.lr_scheduler`.
    """

    # Adam (single AdamW LR for embed / norms / biases). Default approximates
    # the heuristic's value at the canonical 4875-step / 768d / batch=512 scale;
    # callers should normally pass the heuristic-computed value via the helper
    # below rather than relying on the default.
    adam_lr: float = 0.003616

    # Shared (β1 / β2 / ε / max_grad_norm).
    beta1: float = 0.9062
    beta2: float = 0.996006
    epsilon: float = 6.76e-16
    max_grad_norm: float | None = 1.0
    # Clamp floor for ||u|| and ||new_p|| inside `scale_by_adamh_safe`.
    h_norm_eps: float = 1e-10

    def build(self, num_train_steps: int):
        h_lr_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def adamh_group(learning_rate) -> optax.GradientTransformation:
            components: list[optax.GradientTransformation] = []
            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            components.append(
                scale_by_adamh_safe(
                    b1=self.beta1,
                    b2=self.beta2,
                    eps=self.epsilon,
                    learning_rate=learning_rate,
                    norm_eps=self.h_norm_eps,
                )
            )
            return optax.chain(*components)

        def adam_group(adam_lr) -> optax.GradientTransformation:
            components: list[optax.GradientTransformation] = []
            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
            components.append(optax.scale(-adam_lr))
            return optax.chain(*components)

        def optimizer(learning_rate, adam_lr):
            return optax.multi_transform(
                {"adamh": adamh_group(learning_rate), "adam": adam_group(adam_lr)},
                self._create_mask,
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=h_lr_schedule, adam_lr=adam_lr_schedule)

    def _create_mask(self, params):
        """Mirror moe's routing on nano paths.

        - Top-level `embed` -> AdamW.
        - Anything 2D+ that isn't the embedding (= block weights AND the lm
          head) -> AdamH. Matches `experiments/grug/moe/optimizer.py:create_mask`'s
          fall-through to `adamh` for non-special >=2D params.
        - Everything else (norms, biases, scalars) -> AdamW.

        Under MoE: the router weight (2-D ``(D, E)``) goes to AdamW per the
        user's spec. ``router_bias`` is 1-D and already lands in ``adam`` via
        the ndim rule. 3-D expert tensors (``w_gate`` / ``w_up`` / ``w_down``)
        fall through to AdamH so AdamH's per-matrix Frobenius-norm preservation
        applies independently to each expert (vmap over leading axis).
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()

            ndim = getattr(param, "ndim", None)
            # 1D params (norms, biases, etc.) -> AdamW.
            if ndim is None or ndim < 2:
                return "adam"
            # `attn_gate` -> AdamW (matches moe convention).
            if "attn_gate" in path_lower:
                return "adam"
            # `gated_norm` -> AdamH (per the user's spec; differs from moe's AdamW routing).
            # Checked BEFORE the embed rule so `embed_gated_norm` lands here, not `adam`.
            if "gated_norm" in path_lower:
                return "adamh"
            # MoE router weight -> AdamW (matches user's spec).
            if ".router" in path_lower or path_lower.endswith(".router"):
                return "adam"
            # Top-level embedding matrix -> AdamW.
            if "embed" in path_lower:
                return "adam"
            # Everything else 2D+ (block weights, MoE expert tensors, lm head) -> AdamH.
            return "adamh"

        return jax.tree.map(mask_fn, params, paths)


def build_heuristic_optimizer(
    *,
    batch_size: int,
    num_train_steps: int,
    seq_len: int,
    hidden_dim: int,
    heuristic: MoeAdamHHeuristic | None = None,
) -> NanoHeuristicAdamHConfig:
    """Compute β/ε/lrs from the moe heuristic at this run's scale and bundle them.

    `max_grad_norm` is forced to ``None`` rather than inheriting the heuristic's
    default of 1.0: under our explicit-mesh-axes setup, `optax.clip_by_global_norm`
    runs a tree-wide L2 reduction that doesn't get psum'd across data shards,
    which produces both a `train/loss = eval/loss / 4` reporting bug *and*
    per-shard divergent training that ends in NaN. See
    `experiments/grug/nano/launch_adamh_heuristic_test4.py` for the diagnostic
    that established this. The other working configs (`nano-muon`,
    `nano-adamh-ref`) also do not clip.
    """
    h = heuristic or MoeAdamHHeuristic()
    tokens_per_batch = batch_size * seq_len
    total_tokens = tokens_per_batch * num_train_steps
    return NanoHeuristicAdamHConfig(
        learning_rate=h._compute_learning_rate(tokens_per_batch, total_tokens, hidden_dim),
        adam_lr=h._compute_adam_lr(tokens_per_batch, total_tokens, hidden_dim),
        beta1=h.beta1,
        beta2=h._compute_beta2(tokens_per_batch),
        epsilon=h._compute_epsilon(tokens_per_batch, total_tokens),
        # See note in docstring: explicit-mesh-axes + clip_by_global_norm = bug.
        max_grad_norm=None,
        # AdamH preserves Frobenius norm; the heuristic does not weight-decay.
        weight_decay=0.0,
        # Schedule.
        warmup=h.warmup,
        decay=h.decay,
        lr_schedule=h.lr_schedule,
        min_lr_ratio=h.min_lr_ratio,
    )


# ---- Launch config ----


@dataclass(frozen=True)
class NanoAdamHHeuristicLaunchConfig:
    model: NanoModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


# Same architecture as the ref-aligned runs; init must be non-zero so AdamH
# can move the matrices, but we keep the ref's truncated-normal init style
# (zero_init_proj=False just removes the zero override on "proj" weights).
NANO_124M_HEURISTIC_MODEL = NanoModelConfig(
    vocab_size=50304,
    hidden_dim=768,
    intermediate_dim=3072,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    max_seq_len=1024,
    zero_init_proj=False,
    init_scheme="default",
)


def _fineweb_gpt2_data() -> LmDataConfig:
    """Same fineweb-10B / gpt2 / sequential read used by the other two nano launches."""
    from rigging.filesystem import marin_prefix

    base = os.path.join(marin_prefix(), "data", "fineweb10B-gpt2")
    return LmDataConfig(
        tokenizer="gpt2",
        block_cross_document_attention=False,
        auto_build_caches=False,
        shuffle=False,
        components={
            "fineweb_train": DatasetComponent(cache_dir=base, split="train"),
            "fineweb_val": DatasetComponent(cache_dir=base, split="validation"),
        },
        train_weights={"fineweb_train": 1.0},
    )


def _resolve_run_id(default_run_id: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_nano_adamh_heuristic_trial(config: NanoAdamHHeuristicLaunchConfig) -> None:
    # When the model is MoE, add an `expert` mesh axis (size 1, like moe's
    # default) so `MoEMLP`'s shard_map / expert reshards see the axis. Dense
    # runs keep their existing axis-less mesh to preserve cache hashes.
    extra_kwargs = {"mesh": MeshConfig(axes={"expert": 1})} if config.model.use_moe else {}
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            # Only the final force=True checkpoint is kept permanently — pick a
            # step interval larger than any walk's total steps so the pinning
            # rule never fires. Intermediate checkpoints rotate via save_interval.
            keep=[{"every": 1_000_000}],
        ),
        **extra_kwargs,
    )

    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-rsqrt_cap")

# Build optimizer config from the heuristic at this run's scale.
NANO_124M_HEURISTIC_OPTIMIZER = build_heuristic_optimizer(
    batch_size=512,
    num_train_steps=ADAMH_HEURISTIC_TRAIN_STEPS,
    seq_len=NANO_124M_HEURISTIC_MODEL.max_seq_len,
    hidden_dim=NANO_124M_HEURISTIC_MODEL.hidden_dim,
)


nano_adamh_heuristic_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(NANO_124M_HEURISTIC_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(ADAMH_HEURISTIC_TRAIN_STEPS),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic"],
            group="nano-trial",
            name=None,  # filled from run_id in _resolve_tracker
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NANO_124M_HEURISTIC_OPTIMIZER),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,  # heuristic default
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=125,
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_adamh_heuristic_trial],
        description="Nano (modded-nanogpt) 124M, AdamH from moe heuristic, 4875 steps on fineweb10B-gpt2.",
    )
