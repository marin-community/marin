# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""grug value-embedding (VE) ablation, sized for a single H200.

A second token-indexed embedding table, shared across all layers, blended into each layer's
attention value path under a learnable per-layer gate. Queries and keys are untouched, so the
attention pattern is the base model's exactly; only the payload a token delivers when attended
to carries the extra signal. It buys ``vocab_size x kv_width`` parameters at the cost of one
embedding lookup -- no added matmul FLOPs.

The experiment is a paired A/B: the same model, data, seed, and token budget, run with the
value-embedding gate on and off. Both arms are the same code path -- ``value_emb_lambda_init``
is None in the control -- so the difference between them is the intervention and nothing else.

    # the pair, at 130M, on one H200
    python -m experiments.grug.ve.launch --arm ve
    python -m experiments.grug.ve.launch --arm base

    # the parameter-matched dumb control (wider MLP; see _wide_intermediate_dim)
    python -m experiments.grug.ve.launch --arm wide

    # the same pair at 1.3B once 130M says it is worth the tokens
    python -m experiments.grug.ve.launch --arm ve --size 1.3b

Parameter counts are not matched across arms, and deliberately so: the whole claim is
capacity-per-FLOP, and matching parameters would erase the effect being measured. The
comparison that settles it is loss at equal FLOPs (equivalently, equal tokens), which is what
these two arms hold fixed. ``throughput/mfu`` and ``ve/lambda/layer_<i>`` are the diagnostics
to watch: the first says the lookup stayed free, the second is the model's own report of where
in depth a token-identity side channel is worth having.

Know what the table costs before reading a win off it. Against a 128k Llama vocab it is not a
rounding error:

    130m:  148.6M -> 214.3M params  (+65.7M, +44%)
    1.3b: 1481.7M -> 1744.4M params (+262.7M, +18%)

Zero added matmul FLOPs, but 44% more parameters at the small rung -- the speedrun lineage this
comes from runs a ~50k vocab, and the cost here scales with vocab. Two consequences. First, the
optimizer state and HBM for that table are real even though the FLOPs are not, which is what the
batch sizes above leave room for. Second, if the VE arm wins at 130M, "value embeddings help" and
"44% more parameters help, wherever you put them" both explain it, and the run cannot tell them
apart; the ``wide`` arm separates those by spending the same budget somewhere dumb (a wider
MLP), and it is worth running before believing the mechanism.

The lever on that cost is ``num_kv_heads``. The table is sized to the KV width, so grouped-query
attention shrinks it directly -- at 130M, ``num_kv_heads=2`` takes it from 65.7M to 16.4M (+11%)
and makes the parameter-matching objection much smaller, at the price of changing the attention
config in both arms. Left at MHA here so the base rung matches the base template.
"""

import argparse
import dataclasses
import os
from dataclasses import dataclass
from datetime import timedelta

import jmp
from fray.types import ANY_REGION, ResourceConfig
from levanter.analysis.backward_flow import BackwardFlowConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint, resolve_checkpointer_output_path

from experiments.datasets.prebuilt_caches import fineweb_edu_10B_dataset
from experiments.grug.ve.model import GrugModelConfig
from experiments.grug.ve.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

# One H200. `regions=[ANY_REGION]` keeps the job schedulable on a GPU host that advertises no
# region, which is what a personal box looks like to the scheduler.
_TRAIN_RESOURCES = ResourceConfig.with_gpu("H200", count=1, cpu=16, disk="256G", ram="128G", regions=[ANY_REGION])

# The gate's init: 0.5 is "trust the computed value and the stored value equally", and lets the
# model move in either direction from the start. It is not a claim that half is right -- the
# per-layer gates are free to go anywhere, and where they land is the experiment's readout.
_LAMBDA_INIT = 0.5


def _wide_intermediate_dim(model: GrugModelConfig) -> int:
    """MLP width for the parameter-matched dumb control.

    The ``wide`` arm answers "is the VE win just 44% more parameters, wherever you put
    them?" by spending the VE table's exact parameter count on wider MLPs instead: the
    table costs ``vocab x kv_width`` and each unit of intermediate_dim costs
    ``2 x hidden x num_layers``, so the quotient is the width increase. Note this control
    is params-matched but NOT FLOPs-matched -- the wider MLP does real matmul work, so at
    equal tokens the wide arm gets strictly more compute than the ve arm. If ve still wins,
    the mechanism claim survives its strongest form.
    """
    table_params = model.vocab_size * model.value_emb_dim
    params_per_unit_width = 2 * model.hidden_dim * model.num_layers
    return model.intermediate_dim + round(table_params / params_per_unit_width)


# Sequence length is 2048 rather than the base template's 4096: at these model sizes attention is
# a small share of FLOPs, and the shorter context buys batch size on a single device.
_SEQ_LEN = 2048

# Adam for everything, including the value-embedding table. If this is ever ported onto Muon, the
# table belongs in the embedding partition, not with the matrices: a lookup table's per-row
# gradients are sparse, and Muon's orthogonalized update assumes the dense statistics of a matmul
# weight.
#
# `weight_decay_modules` is an explicit inclusion list because the default mask would decay the
# `lambda_ve` gates (it only exempts norms, biases, and `*Embedding*` class names, and grug's
# parameters are raw arrays). A decayed gate is pulled toward zero at every step regardless of
# gradient, which both suppresses the mechanism and confounds the lambda-vs-depth readout -- a
# layer at lambda~0 could mean "dialed away" or just "nothing opposed the decay". The list keeps
# decay on exactly the parameters the default decays today, minus the gates; norm weights stay
# undecayed as before.
_OPTIMIZER = AdamConfig(
    learning_rate=3e-3,
    weight_decay=0.1,
    weight_decay_modules=[
        "w_q",
        "w_k",
        "w_v",
        "w_o",
        "mlp_up",
        "mlp_down",
        "token_embed",
        "value_embed",
        "output_proj",
    ],
    lr_schedule="cosine",
    decay=0.2,
    min_lr_ratio=0.1,
    warmup=1000,
)

# Two rungs. 130M is the iteration rung -- it turns around fast enough to iterate on, and it is
# where a speedrun-scale result would show up if it transfers at all. 1.3B is the confirmation
# rung: the VE table costs a fixed vocab x kv_width, so it shrinks as a fraction of the model as
# the model grows, and the question worth the GPU-hours is whether the win shrinks with it.
_MODELS = {
    "130m": GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=512,
        intermediate_dim=1792,
        num_layers=6,
        num_heads=8,
        num_kv_heads=8,
        max_seq_len=_SEQ_LEN,
    ),
    "1.3b": GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=2048,
        intermediate_dim=5632,
        num_layers=24,
        num_heads=16,
        num_kv_heads=16,
        max_seq_len=_SEQ_LEN,
    ),
}

# Batch size and token budget per rung, sized for one 141GB H200 with the template's per-block
# activation checkpointing. On smaller devices override with --batch-size: batch 128 at the 130m
# rung OOMs on a 95GB H100 (the train step allocates ~78GiB). Halving the batch halves the token
# budget rather than doubling the steps -- both arms must share whatever value is used, and the
# A/B stays valid at any batch size.
_BATCH_SIZE = {"130m": 128, "1.3b": 32}
_STEPS = {"130m": 8_000, "1.3b": 12_000}


@dataclass(frozen=True)
class GrugVeLaunchConfig:
    """Last-mile run config for the value-embedding variant.

    Flat scalars rather than nested trainer/eval configs: those carry
    ``jax.sharding.PartitionSpec`` defaults the artifact fingerprint cannot serialize.
    ``build_grug_run_config`` reconstitutes them at run time.
    """

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    z_loss_weight: float = 1e-4
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    log_every: int = 10
    loss_implementation: str | tuple[str, ...] | None = None  # cross-entropy kernel; None uses the trainer default.
    eval_batch_size: int | None = 128  # None disables perplexity eval.
    steps_per_eval: int = 500
    max_eval_batches: int = 8
    eval_current: bool = True
    eval_ema: bool = False


def _resolve_run_id(default_run_id: str) -> str:
    """Resolve run id and append `FERRY_DATE` when launching from ferry workflows."""
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def build_grug_run_config(launch: GrugVeLaunchConfig) -> GrugRunConfig:
    """Map launch knobs onto the trainer's full ``GrugRunConfig``."""
    output_path = launch.output_path
    trainer = TrainerConfig(
        id=launch.run_id,
        seed=launch.seed,
        train_batch_size=launch.batch_size,
        num_train_steps=launch.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(launch.mp),
        tracker=_resolve_tracker(launch.tracker, launch.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=resolve_checkpointer_output_path(
            CheckpointerConfig(save_interval=timedelta(minutes=30), keep=None),
            output_path,
        ),
    )

    grug_trainer = GrugTrainerConfig(
        trainer=trainer,
        z_loss_weight=launch.z_loss_weight,
        ema_beta=launch.ema_beta,
        log_every=launch.log_every,
        loss_implementation=launch.loss_implementation,
        # Backward-flow tracing OFF: traced steps run without per-block activation
        # checkpointing (see model.py Block dispatch), which needs tens of GiB more than a
        # normal step -- it is what OOMed this config on a 95GB H100 at step 0. The
        # diagnostic is not part of this experiment; re-enable deliberately if needed.
        backward_flow=BackwardFlowConfig(interval=0),
    )

    eval_config = (
        GrugEvalConfig(
            eval_batch_size=launch.eval_batch_size,
            steps_per_eval=launch.steps_per_eval,
            max_eval_batches=launch.max_eval_batches,
            eval_current=launch.eval_current,
            eval_ema=launch.eval_ema,
        )
        if launch.eval_batch_size is not None
        else None
    )

    return GrugRunConfig(
        model=launch.model,
        data=launch.data,
        resources=launch.resources,
        output_path=output_path,
        optimizer=launch.optimizer,
        trainer=grug_trainer,
        eval=eval_config,
    )


def run_grug_ve_trial(config: GrugVeLaunchConfig) -> None:
    """Build the full grug run config and dispatch the training job to Fray."""
    run_grug(build_grug_run_config(config))


def grug_ve_trial(
    *,
    arm: str,
    size: str = "130m",
    version: str = "dev",
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
) -> ArtifactStep[LevanterCheckpoint]:
    """One arm of the value-embedding ablation as a lazy checkpoint.

    ``arm="ve"`` turns the value-embedding gate on; ``arm="base"`` is the control, identical in
    every other respect; ``arm="wide"`` is the parameter-matched dumb control -- no value
    embeddings, the same parameter budget spent on wider MLPs instead. All arms train on
    fineweb-edu at the same seed, batch size, and token budget, so loss gaps between them are
    the interventions. Keep --batch-size identical across every arm you intend to compare:
    it fixes the data order.
    """
    if arm not in ("ve", "base", "wide"):
        raise ValueError(f"arm must be 've', 'base', or 'wide', got {arm!r}")
    if size not in _MODELS:
        raise ValueError(f"size must be one of {tuple(_MODELS)}, got {size!r}")

    model = _MODELS[size]
    if arm == "ve":
        model = dataclasses.replace(model, value_emb_lambda_init=_LAMBDA_INIT)
    elif arm == "wide":
        model = dataclasses.replace(model, intermediate_dim=_wide_intermediate_dim(model))
    train_data = fineweb_edu_10B_dataset()
    run_id = _resolve_run_id(f"grug-ve-{size}-{arm}")
    resolved_batch_size = batch_size if batch_size is not None else _BATCH_SIZE[size]
    resolved_eval_batch_size = eval_batch_size if eval_batch_size is not None else resolved_batch_size

    def build_config(ctx: StepContext) -> GrugVeLaunchConfig:
        return GrugVeLaunchConfig(
            model=model,
            data=mixture(ctx, {train_data: 1.0}),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=_STEPS[size],
            batch_size=resolved_batch_size,
            # The same seed in both arms: control and treatment see the same data in the same
            # order and start from the same draw, so what separates them is not a seed.
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "value-embeddings", size, arm],
                group=f"grug-ve-{size}",
                name=None,  # filled from run_id in _resolve_tracker
                replicate_path=ctx.output_path,
            ),
            optimizer=_OPTIMIZER,
            z_loss_weight=1e-4,
            ema_beta=None,
            log_every=10,
            # Eval follows the train batch: eval is forward-only, so anything that fits a train
            # step fits eval at the same batch, and no separate memory budget is needed.
            eval_batch_size=resolved_eval_batch_size,
            steps_per_eval=500,
            max_eval_batches=8,
            eval_current=True,
            eval_ema=False,
        )

    return ArtifactStep(
        name=user_namespaced_name(f"grug/ve-{size}-{arm}", version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_ve_trial,
        build_config=build_config,
        deps=(train_data,),
        runtime_args={"train_resources": _TRAIN_RESOURCES},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("ve", "base", "wide"), default="ve")
    parser.add_argument("--size", choices=tuple(_MODELS), default="130m")
    parser.add_argument("--version", default="dev")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the per-size default batch size; both arms of a comparison must use the same value.",
    )
    args = parser.parse_args()
    StepRunner().run(
        [grug_ve_trial(arm=args.arm, size=args.size, version=args.version, batch_size=args.batch_size).lower()]
    )
