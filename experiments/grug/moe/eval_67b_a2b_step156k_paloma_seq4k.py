# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-score the 67B-A2B step-156k checkpoint on Paloma + UncheatableEval at seq=4096.

The training-time eval of the step-141k cooldown ran at seq=65,536, which gives long
documents up to 65k tokens of in-document context. The open-weight baselines in
issue #8824 were scored with lm-eval at a 4,096-token rolling window. This eval reruns
the same Levanter tagged evaluator at seq=4096 with full slice coverage
(``max_eval_batches=None``) so the context budget matches.

Only ``params`` is restored from the checkpoint (no optimizer state), so the job fits
on a v4-64 in us-central2, the checkpoint's region.

Submit::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.eval_67b_a2b_step156k_paloma_seq4k
"""

import dataclasses
import logging
import time
from dataclasses import dataclass

import jax
import jmp
import wandb
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import load_checkpoint
from levanter.data.text import LmDataConfig
from levanter.eval import eval_model
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.launch_datakit_moe_mix import _MIXTURE_BLOCK_SIZE, _datakit_components, _phase_weights
from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.grug.moe.train import GrugEvalConfig, build_tagged_evaluator
from experiments.marin_models import marin_tokenizer

logger = logging.getLogger(__name__)

_DIM: int = 2560
_SEQ: int = 4096
_EVAL_BATCH_SIZE: int = 256
_QK_MULT: float = 1.57  # value the checkpoint was trained with in the 65k cooldown
CHECKPOINT_PATH: str = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step141k-a30ef8/"
    "checkpoints/step-156000/"
)
CHECKPOINT_STEP: int = 156_000


@dataclass(frozen=True)
class CheckpointEvalConfig:
    run_id: str
    output_path: str
    resources: ResourceConfig
    model: GrugModelConfig
    checkpoint_path: str
    checkpoint_step: int
    seq_len: int
    eval_batch_size: int
    mp: str
    data: LmDataConfig


def _run_eval_local(config: CheckpointEvalConfig) -> None:
    """TPU-side entrypoint: restore params only and run the tagged evaluator once."""
    trainer = TrainerConfig(
        id=config.run_id,
        train_batch_size=config.eval_batch_size,
        num_train_steps=1,
        profiler=ProfilerConfig(),
        mp=jmp.get_policy(config.mp),
        tracker=WandbConfig(
            project="marin_moe",
            name=config.run_id,
            tags=["moe", "67b_a2b", "checkpoint_eval", f"seq{config.seq_len}", "issue8824_compare"],
            group="june-tpu-67b-a2b-evals",
        ),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        load_checkpoint=False,
    )
    trainer.initialize()

    with trainer.use_device_mesh():

        @jax.jit
        def _init(key):
            return trainer.mp.cast_to_param(Transformer.init(config.model, key=key))

        template = _init(jax.random.PRNGKey(0))
        t_load = time.perf_counter()
        model = load_checkpoint(
            template,
            config.checkpoint_path,
            subpath="params",
            axis_mapping=None,
            mesh=trainer.device_mesh,
            allow_partial=False,
        )
        logger.info("Restored params from %s in %.1fs", config.checkpoint_path, time.perf_counter() - t_load)

        evaluator = build_tagged_evaluator(
            data_config=config.data,
            max_seq_len=config.seq_len,
            mesh=trainer.device_mesh,
            eval_cfg=GrugEvalConfig(
                eval_batch_size=config.eval_batch_size,
                steps_per_eval=1,
                max_eval_batches=None,  # score every token of every slice
                eval_current=True,
                eval_ema=False,
            ),
        )
        assert evaluator is not None, "data config has no validation sets"

        t_eval = time.perf_counter()
        log = eval_model(evaluator, model, prefix="eval")
        logger.info(
            "paloma macro_loss=%.5f macro_bpb=%.5f; uncheatable macro_bpb=%.5f; elapsed=%.1fs",
            log.get("eval/paloma/macro_loss", float("nan")),
            log.get("eval/paloma/macro_bpb", float("nan")),
            log.get("eval/uncheatable_eval/macro_bpb", float("nan")),
            time.perf_counter() - t_eval,
        )
        wandb.log(log, step=config.checkpoint_step)

    wandb.finish()


def run_checkpoint_eval(config: CheckpointEvalConfig) -> None:
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_eval_local,
        resources=config.resources,
    )


# Same architecture as the step-141k cooldown launcher, at the eval seq_len.
MODEL_CONFIG = dataclasses.replace(
    MoeMuonHHeuristic(min_lr_ratio=0.05).build_model_config(_DIM, seq_len=_SEQ),
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
    qk_mult=_QK_MULT,
    hybrid_attention_flops_accounting=True,
)

_data = add_validation_sets_to_mixture(
    LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components=_datakit_components(),
        train_weights=[(0, _phase_weights(1))],
        auto_build_caches=False,
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
    ),
    default_validation_sets(tokenizer=marin_tokenizer),
)

_run_id = f"moe_67b_a2b_d{_DIM}_step156k_eval_seq{_SEQ}_full"
eval_step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_checkpoint_eval,
    config=CheckpointEvalConfig(
        run_id=_run_id,
        output_path=this_output_path(),
        resources=versioned(ResourceConfig.with_tpu("v4-64")),
        model=versioned(MODEL_CONFIG),
        checkpoint_path=versioned(CHECKPOINT_PATH),
        checkpoint_step=CHECKPOINT_STEP,
        seq_len=versioned(_SEQ),
        eval_batch_size=versioned(_EVAL_BATCH_SIZE),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        data=_data,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[eval_step],
        description=(
            f"67B-A2B step-{CHECKPOINT_STEP} params-only eval at seq={_SEQ}, full Paloma + "
            "UncheatableEval coverage, to match the 4K window of the #8824 open-weight baselines. v4-64 us-central2."
        ),
    )
