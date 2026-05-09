# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon walk p19: p16 + final-logit z-loss + move gated_norm and lm head to AdamH.

Three changes on top of p16-muon:

1. **Final-logit z-loss** (``z_loss_weight=1e-4``) — same as adamh-side.
2. **Mask change**: ``gated_norm`` and the top-level lm-head ``proj.*`` move
   from AdamW (``adam_norm`` / ``adam_head``) into a new ``adamh`` group that
   uses the AdamH heuristic LR / β / ε.
3. **Init scheme swap**: ``init_scheme="default"`` with
   ``zero_init_proj=False`` and ``initializer_std=0.5/√hidden_dim``, matching
   the AdamH-heuristic init recipe. The lm head can no longer start at zero
   (AdamH preserves Frobenius norm — a zero matrix would stay zero).

Block 2-D weights still go to Muon. Embeddings, 1-D params, attn_gate, and
the router weight stay on AdamW. The hybrid walks Muon for the bulk of the
network and AdamH for the two pieces that pure Muon doesn't reach well: the
scale-invariant gating layers (gated_norm) and the unembedding.
"""

import dataclasses
import math
from dataclasses import dataclass
from datetime import timedelta
from functools import partial

import jax
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

from experiments.grug.moe.heuristic import MoeAdamHHeuristic
from experiments.grug.nano.launch import (
    NanoLaunchConfig,
    _resolve_run_id,
    _resolve_tracker,
)
from experiments.grug.nano.launch_muon_tuned_walk_p16 import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, P16_MODEL
from experiments.grug.nano.optimizer import scale_by_adamh_safe
from experiments.grug.nano.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

P19_TRAIN_STEPS = 10343
P19_BATCH_SIZE = 64

# Use the AdamH-heuristic init recipe for parameters that AdamH will touch.
# `init_scheme="default"` with `zero_init_proj=False` keeps the lm head
# non-zero so AdamH has something to scale into. The block-weight init
# (truncated_normal x initializer_std) differs slightly from muon_tuned's
# N(0, sqrt(0.33/fan_in)) but Muon's NS erases that difference quickly.
P19_MODEL = dataclasses.replace(
    P16_MODEL,
    init_scheme="default",
    zero_init_proj=False,
    initializer_std=0.5 / math.sqrt(P16_MODEL.hidden_dim),
)

# Pull AdamH heuristic hyperparams at this scale.
_HEURISTIC = MoeAdamHHeuristic()
_TPB = P19_BATCH_SIZE * P19_MODEL.max_seq_len
_TOTAL_TOKENS = _TPB * P19_TRAIN_STEPS
P19_ADAMH_LR = _HEURISTIC._compute_learning_rate(_TPB, _TOTAL_TOKENS, P19_MODEL.hidden_dim)
P19_ADAM_LR = _HEURISTIC._compute_adam_lr(_TPB, _TOTAL_TOKENS, P19_MODEL.hidden_dim)
P19_ADAMH_BETA2 = _HEURISTIC._compute_beta2(_TPB)
P19_ADAMH_EPSILON = _HEURISTIC._compute_epsilon(_TPB, _TOTAL_TOKENS)


def _stable_then_decay_schedule(num_train_steps: int, cooldown_frac: float) -> optax.Schedule:
    """Modded-nanogpt LR shape: 1.0 for the first (1 - cooldown_frac) of steps, then linear → 0."""
    plateau_steps = int(num_train_steps * (1.0 - cooldown_frac))
    decay_steps = max(num_train_steps - plateau_steps, 1)
    return optax.join_schedules(
        [optax.constant_schedule(1.0), optax.linear_schedule(1.0, 0.0, decay_steps)],
        [plateau_steps],
    )


@OptimizerConfig.register_subclass("nano_muon_adamw_adamh")
@dataclass(frozen=True)
class NanoMuonAdamWAdamHConfig(OptimizerConfig):
    """Hybrid Muon + AdamW + AdamH for the p19 muon walk.

    Five parameter groups, all sharing one ``lr_scale`` schedule multiplier:

    - ``muon`` — block 2-D and 3-D weights (q/k/v/attn proj + MoE expert tensors).
    - ``adam_embed`` — token embedding (AdamW @ ``embed_lr``).
    - ``adam_norm`` — 1-D params (RMSNorm gains, biases, ``router_bias``,
      ``attn_gate``, ``.router`` weight) (AdamW @ ``norm_lr``).
    - ``adamh`` — ``gated_norm`` and the top-level ``proj.*`` (lm head). Uses
      AdamH (Frobenius-norm preservation) at the heuristic LR / β / ε.
    - ``adam_head`` is gone; the lm head moves into ``adamh``.
    """

    # AdamW.
    embed_lr: float = 0.3
    norm_lr: float = 0.01
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95
    adam_eps: float = 1e-10
    # Muon.
    muon_lr: float = 0.035
    muon_weight_decay: float = 0.025
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 12
    muon_ns_coeff_type: str = "simple"
    muon_eps: float = 1e-8
    muon_use_kimi_scaling: bool = False
    # AdamH (gated_norm + lm head). Defaults are the heuristic values for
    # b=64, seq=4096, steps=10343, hidden_dim=768. Override via kwargs if
    # rerunning at a different scale.
    adamh_lr: float = P19_ADAMH_LR
    adamh_beta1: float = 0.9062
    adamh_beta2: float = P19_ADAMH_BETA2
    adamh_eps: float = P19_ADAMH_EPSILON
    adamh_norm_eps: float = 1e-10
    # Schedule (modded-nanogpt cooldown).
    cooldown_frac: float = 0.7
    max_grad_norm: float | None = None

    def build(self, num_train_steps: int):
        schedule = _stable_then_decay_schedule(num_train_steps, self.cooldown_frac)

        def adamw_group(base_lr: float) -> optax.GradientTransformation:
            components: list[optax.GradientTransformation] = [
                optax.scale_by_adam(b1=self.adam_beta1, b2=self.adam_beta2, eps=self.adam_eps),
                optax.scale(-base_lr),
            ]
            return optax.chain(*components)

        def muon_group() -> optax.GradientTransformation:
            # NS first, then reshard updates back to the param's spec, then
            # decay/scale — same ordering fix as `launch.py:muon_group`.
            components: list[optax.GradientTransformation] = [
                _grug_scale_with_muon(
                    self.muon_momentum,
                    self.muon_nesterov,
                    self.muon_ns_steps,
                    self.muon_eps,
                    self.muon_use_kimi_scaling,
                    self.muon_ns_coeff_type,
                ),
                _match_update_sharding(),
            ]
            if self.muon_weight_decay > 0:
                components.append(optax.add_decayed_weights(self.muon_weight_decay))
            components.append(optax.scale(-self.muon_lr))
            return optax.chain(*components)

        def adamh_group() -> optax.GradientTransformation:
            # AdamH has its own betas/eps and its own (heuristic) LR. The lr
            # already lives inside `scale_by_adamh_safe`; multiplying by the
            # shared `lr_scale` schedule is the only outer LR knob.
            return scale_by_adamh_safe(
                b1=self.adamh_beta1,
                b2=self.adamh_beta2,
                eps=self.adamh_eps,
                learning_rate=self.adamh_lr,
                norm_eps=self.adamh_norm_eps,
            )

        def optimizer(lr_scale):
            transformations = {
                "adam_embed": optax.chain(adamw_group(self.embed_lr), optax.scale(lr_scale)),
                "adam_norm": optax.chain(adamw_group(self.norm_lr), optax.scale(lr_scale)),
                "muon": optax.chain(muon_group(), optax.scale(lr_scale)),
                "adamh": optax.chain(adamh_group(), optax.scale(lr_scale)),
            }
            grouped = optax.multi_transform(transformations, partial(self._create_mask))
            if self.max_grad_norm is None:
                return grouped
            return optax.chain(optax.clip_by_global_norm(self.max_grad_norm), grouped)

        return optax.inject_hyperparams(optimizer)(lr_scale=schedule)

    def _create_mask(self, params):
        """Route every leaf into one of {muon, adam_embed, adam_norm, adamh}.

        Order matters — the AdamH route must catch ``gated_norm`` *and* the
        top-level ``proj.*`` (lm head) before the generic 1-D / blocks /
        embed rules fire.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()

            # `gated_norm` (every block + post-embed + pre-lm-head) -> AdamH.
            # This catches `attn_gated_norm`, `mlp_gated_norm`, top-level
            # `embed_gated_norm`, and `final_gated_norm`.
            if "gated_norm" in path_lower:
                return "adamh"
            # Top-level lm head (`Transformer.proj`) -> AdamH.
            if path_lower == "proj.weight" or path_lower.startswith("proj."):
                return "adamh"

            ndim = getattr(param, "ndim", None)
            if ndim is not None and ndim < 2:
                return "adam_norm"
            if "attn_gate" in path_lower:
                return "adam_norm"
            if ".router" in path_lower or path_lower.endswith(".router"):
                return "adam_norm"
            if "embed" in path_lower:
                return "adam_embed"
            if "blocks" in path_lower:
                return "muon"
            # Defensive: anything 2D+ outside `blocks` and not embed/proj.
            return "adam_norm"

        return jax.tree.map(mask_fn, params, paths)


P19_OPTIMIZER = NanoMuonAdamWAdamHConfig()


def run_nano_p19_trial(config: NanoLaunchConfig) -> None:
    """Custom runner: same as `run_nano_trial` but the optimizer is hybrid.

    We can reuse `run_nano_trial` directly since it just calls
    `config.optimizer.build(...)`, but we duplicate the body here so any
    ``MeshConfig`` / ``checkpointer`` / ``profiler`` knobs that depend on
    `config.model.use_moe` keep working without edits to `launch.py`.
    """
    # Required to enable the expert axis for the MoE shard_map paths.
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
    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-p19")


nano_muon_tuned_p19_trial = ExecutorStep(
    name="grug/nano-muon-tuned-p19-trial",
    fn=run_nano_p19_trial,
    config=NanoLaunchConfig(
        model=versioned(P19_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P19_TRAIN_STEPS),
        batch_size=versioned(P19_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=[
                "grug",
                "nano",
                "muon",
                "nemotron",
                "tuned",
                "p19",
                "moe",
                "fused-ce",
                "muonz",
                "adamh-proj",
                "adamh-gated-norm",
            ],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P19_OPTIMIZER),
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
                eval_batch_size=P19_BATCH_SIZE,
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
        steps=[nano_muon_tuned_p19_trial],
        description="muon-tuned p19: + z_loss + move gated_norm and lm head to AdamH.",
    )
