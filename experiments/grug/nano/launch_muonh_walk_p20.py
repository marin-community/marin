# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH walk p20: take p16's compute-optimal MoE recipe and swap the matrix
optimizer from Muon to MuonH (Newton-Schulz orthogonalised direction applied
via a Frobenius-norm-preserving hyperball projection).

Recipe lifted from `experiments/grug/muonh_tuned.py`:

  - **MuonH** for every block 2-D and 3-D weight (q/k/v/attn.proj/mlp.fc/mlp.proj
    plus the 3-D MoE expert tensors `w_gate`/`w_up`/`w_down`).
    `lr=0.018`, `mu=0.95`, Nesterov, **12 NS iterations**, `coefficient_type="simple"`,
    no weight decay (the hyperball projection already preserves ``||p||_F``).
    For 3-D ``(E, D, I)`` tensors we vmap NS over the leading expert axis and
    apply per-expert hyperball.
  - **AdamW** for embed (lr=0.3), lm head proj (lr=1/320), and every 1-D /
    auxiliary 2-D param (norm gains, biases, attn_gate, gated_norm, router).
    `betas=(0.8, 0.95)`, `eps=1e-10`, no WD.

Two schedules, both stable-then-decay, both with peak 1.0 at step 0:

  - MuonH groups: ``cooldown_frac=1.0`` (linear decay from step 0).
  - AdamW groups: ``cooldown_frac=0.4`` (constant for 60% of run, then decay).

`init_scheme="default"` with `zero_init_proj=False` and
`initializer_std=0.5/sqrt(hidden_dim)` so attn.proj / mlp.proj / lm-head start
non-zero. MuonH preserves ``||p||_F`` exactly, so any matrix that starts at
zero stays at zero forever — non-zero residual projections are mandatory.
The Kaiming-with-multipliers init from the torch reference doesn't have a
clean port to nano's MoE block (``_init_adamh_ref`` predates MoE / GatedNorm /
attn_gate), so we use the heuristic-adamh init recipe instead — every 2-D
weight starts at ``N(0, 0.018)`` truncated.

Final-logit z-loss enabled (``z_loss_weight=1e-4``), matching the muonz /
adamh side.
"""

import dataclasses
import math
from dataclasses import dataclass
from datetime import timedelta
from functools import partial

import jax
import jax.numpy as jnp
import jmp
import optax
from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.optim import OptimizerConfig
from levanter.optim.grugmuon import _grug_scale_with_muon, _match_update_sharding
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import leaf_key_paths
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.training.training import temporary_checkpoint_base_path

from experiments.grug.nano.launch import NanoLaunchConfig, _resolve_run_id, _resolve_tracker
from experiments.grug.nano.launch_muon_tuned_walk_p16 import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, P16_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

P20_TRAIN_STEPS = 10343
P20_BATCH_SIZE = 64

# Init: "default" + zero_init_proj=False so residual-side projections (and the
# lm head) start non-zero. The lm head still goes to AdamW (small LR=1/320),
# so it's free to learn a non-trivial output projection regardless of init.
P20_MODEL = dataclasses.replace(
    P16_MODEL,
    init_scheme="default",
    zero_init_proj=False,
    initializer_std=0.5 / math.sqrt(P16_MODEL.hidden_dim),
)


def _stable_then_decay_schedule(num_train_steps: int, cooldown_frac: float) -> optax.Schedule:
    """1.0 for the first (1 - cooldown_frac) fraction, then linear -> 0."""
    plateau_steps = int(num_train_steps * (1.0 - cooldown_frac))
    decay_steps = max(num_train_steps - plateau_steps, 1)
    return optax.join_schedules(
        [optax.constant_schedule(1.0), optax.linear_schedule(1.0, 0.0, decay_steps)],
        [plateau_steps],
    )


def _hyperball_project(learning_rate, eps: float = 1e-10) -> optax.GradientTransformation:
    """Frobenius-norm-preserving hyperball update step.

    Takes the NS-orthogonalised + fan-scaled update direction ``u`` and the
    parameter ``p``, and returns the *delta* such that
    ``optax.apply_updates(p, delta)`` lands at::

        new_p_intermediate = p - learning_rate * u * ||p|| / max(||u||, eps)
        new_p_renorm = new_p_intermediate / max(||new_p_intermediate||, eps) * ||p||

    For 3-D ``(E, D, I)`` tensors the norm is taken per-expert (axes 1..-1)
    so each of ``E`` matrices preserves its own Frobenius norm independently.
    Mirrors `levanter.optim.muonh.scale_with_muonh`'s ``scale_invariant_update``.
    """

    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state

        def project(p, u):
            if p is None or u is None:
                return None
            if p.ndim == 2:
                p_norm = jnp.linalg.norm(p)
                u_norm = jnp.maximum(jnp.linalg.norm(u), eps)
                new_p = p - learning_rate * u * p_norm / u_norm
                new_p_norm = jnp.maximum(jnp.linalg.norm(new_p), eps)
                return new_p / new_p_norm * p_norm - p
            if p.ndim > 2:
                axes = tuple(range(1, p.ndim))
                p_norm = jnp.sqrt(jnp.sum(jnp.square(p), axis=axes, keepdims=True))
                u_norm = jnp.maximum(jnp.sqrt(jnp.sum(jnp.square(u), axis=axes, keepdims=True)), eps)
                new_p = p - learning_rate * u * p_norm / u_norm
                new_p_norm = jnp.maximum(jnp.sqrt(jnp.sum(jnp.square(new_p), axis=axes, keepdims=True)), eps)
                return new_p / new_p_norm * p_norm - p
            # 0-D / 1-D should never reach this transform under the mask, but
            # fall back to a plain step so a bad mask doesn't NaN.
            return -learning_rate * u

        return jax.tree.map(project, params, updates, is_leaf=lambda x: x is None), state

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("nano_muonh_adamw")
@dataclass(frozen=True)
class NanoMuonHAdamWConfig(OptimizerConfig):
    """MuonH (Frobenius-preserving Muon) + AdamW. Mirrors muonh_tuned.py."""

    # AdamW.
    embed_lr: float = 0.3
    head_lr: float = 1.0 / 320.0
    norm_lr: float = 0.01
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95
    adam_eps: float = 1e-10
    adamw_cooldown_frac: float = 0.4
    # MuonH.
    muonh_lr: float = 0.018
    muonh_momentum: float = 0.95
    muonh_nesterov: bool = True
    muonh_ns_steps: int = 12
    muonh_ns_coeff_type: str = "simple"
    muonh_ns_eps: float = 1e-8
    muonh_norm_eps: float = 1e-10
    muonh_cooldown_frac: float = 1.0
    max_grad_norm: float | None = None

    def build(self, num_train_steps: int):
        adamw_eta = _stable_then_decay_schedule(num_train_steps, self.adamw_cooldown_frac)
        muonh_eta = _stable_then_decay_schedule(num_train_steps, self.muonh_cooldown_frac)
        muonh_lr_schedule = lambda step: self.muonh_lr * muonh_eta(step)  # noqa: E731

        def adamw_group(base_lr: float) -> optax.GradientTransformation:
            components: list[optax.GradientTransformation] = [
                optax.scale_by_adam(b1=self.adam_beta1, b2=self.adam_beta2, eps=self.adam_eps),
                optax.scale(-base_lr),
            ]
            return optax.chain(*components)

        def muonh_group(muonh_lr_now) -> optax.GradientTransformation:
            # NS direction (with momentum + Nesterov + post-NS fan scale +
            # 3-D vmap), then reshard updates back to param spec, then
            # hyperball projection (which folds in lr).
            return optax.chain(
                _grug_scale_with_muon(
                    self.muonh_momentum,
                    self.muonh_nesterov,
                    self.muonh_ns_steps,
                    self.muonh_ns_eps,
                    use_kimi_scaling=False,
                    coefficient_type=self.muonh_ns_coeff_type,
                ),
                _match_update_sharding(),
                _hyperball_project(learning_rate=muonh_lr_now, eps=self.muonh_norm_eps),
            )

        def optimizer(adamw_lr_scale, muonh_lr_now):
            transformations = {
                "adam_embed": optax.chain(adamw_group(self.embed_lr), optax.scale(adamw_lr_scale)),
                "adam_head": optax.chain(adamw_group(self.head_lr), optax.scale(adamw_lr_scale)),
                "adam_norm": optax.chain(adamw_group(self.norm_lr), optax.scale(adamw_lr_scale)),
                "muonh": muonh_group(muonh_lr_now),
            }
            grouped = optax.multi_transform(transformations, partial(self._create_mask))
            if self.max_grad_norm is None:
                return grouped
            return optax.chain(optax.clip_by_global_norm(self.max_grad_norm), grouped)

        return optax.inject_hyperparams(optimizer)(
            adamw_lr_scale=adamw_eta,
            muonh_lr_now=muonh_lr_schedule,
        )

    def _create_mask(self, params):
        """Route into {muonh, adam_embed, adam_head, adam_norm}.

        - 1-D params (RMSNorm gains, biases, ``router_bias``) -> ``adam_norm``.
        - ``attn_gate``, ``gated_norm`` (small auxiliary 2-D matrices) -> ``adam_norm``.
        - ``.router`` (the MoE router weight) -> ``adam_norm``.
        - Top-level ``embed`` -> ``adam_embed`` (lr=0.3).
        - Top-level ``proj.*`` (lm head) -> ``adam_head`` (lr=1/320).
        - Block 2-D and 3-D weights -> ``muonh``.
        - Defensive fall-through -> ``adam_norm``.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()

            ndim = getattr(param, "ndim", None)
            if ndim is not None and ndim < 2:
                return "adam_norm"
            if "attn_gate" in path_lower or "gated_norm" in path_lower:
                return "adam_norm"
            if ".router" in path_lower or path_lower.endswith(".router"):
                return "adam_norm"
            if "embed" in path_lower:
                return "adam_embed"
            if path_lower == "proj.weight" or path_lower.startswith("proj."):
                return "adam_head"
            if "blocks" in path_lower:
                return "muonh"
            return "adam_norm"

        return jax.tree.map(mask_fn, params, paths)


P20_OPTIMIZER = NanoMuonHAdamWConfig()


def run_nano_p20_trial(config: NanoLaunchConfig) -> None:
    """Same shape as `run_nano_trial`; expert mesh axis enabled because p20 is MoE."""
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
            base_path=f"{config.output_path}/checkpoints",
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1_000_000}],
        ),
        **extra_kwargs,
    )
    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    run_grug(
        GrugRunConfig(
            model=config.model,
            data=config.data,
            resources=config.resources,
            optimizer=config.optimizer,
            trainer=grug_trainer,
            eval=config.eval,
        )
    )


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muonh-tuned-p20")


nano_muonh_tuned_p20_trial = ExecutorStep(
    name="grug/nano-muonh-tuned-p20-trial",
    fn=run_nano_p20_trial,
    config=NanoLaunchConfig(
        model=versioned(P20_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P20_TRAIN_STEPS),
        batch_size=versioned(P20_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "muonh", "nemotron", "tuned", "p20", "moe", "fused-ce", "muonz"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P20_OPTIMIZER),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
                train_batch_pspec=P(("data", "expert")),
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=P20_BATCH_SIZE,
                steps_per_eval=250,
                max_eval_batches=40,
                eval_current=True,
                eval_ema=False,
                eval_batch_pspec=P(("data", "expert")),
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_muonh_tuned_p20_trial],
        description="muonh-tuned p20: p16 with MuonH (Frobenius-preserving) on block 2-D/3-D weights, dual cooldown.",
    )
