# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume d=1280 from step 13000 at EP=8 with 32k ctx, long-only YaRN, halfmix data.

All knobs at once:

- EP swap: trained EP=1 -> resume EP=8 (params reshard at load time, 32 experts/chip)
- Context: seq 4096 -> 32768; sliding_window stays at trained 2048 on short layers
- Long-only YaRN (NTK-by-parts on inv_freq, mscale_coef=0.1) on every-4th + last layers;
  short layers stay vanilla.
- Data: 50/50 nemotron + 3-bucket longmino mixture (matches halfmix arms)
- Batch: bs 256 -> 32 to preserve tokens_per_batch (= 256*4096 = 32*32768 = 1_048_576),
  so MuonH gives identical peak LR / beta / epsilon and the cosine continues exactly.

bs=32 % (data*expert=16) = 0 -- batch divisibility OK at EP=8. Per-chip MoE gathered
token count matches the d=1280 EP=8 seq=4k bs=256 case.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_d1280_ep8_32kctx_long_yarn_halfmix_resume
"""

import dataclasses
import math
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture, lm_mixture_data_config
from marin.processing.tokenize.data_configs import interpolate_mixture_weights
from marin.training.training import temporary_checkpoint_base_path

from experiments.defaults import default_tokenize, default_validation_sets
from experiments.grug.moe.heuristic_v2 import MoeHeuristicV2
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.long_context_datasets import longmino_bucket_token_counts, longmino_by_bucket
from experiments.marin_models import marin_tokenizer
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.pretraining_datasets.nemotron import NEMOTRON_WEIGHTS, tokenize_nemotron

_SOURCE_CKPT_PATH: str = "gs://marin-us-central2/grug/moe_may_compute_opt_d1280_ep1-b9a7ad/checkpoints/step-13000"
_DIM: int = 1280
_ORIG_BS: int = 256
_ORIG_SEQ: int = 4096
_NEW_BS: int = 32
_NEW_SEQ: int = 32_768
_YARN_OLD_SEQ_LEN: int = 4096
_YARN_ALPHA: int = 1
_YARN_BETA: int = 32
_YARN_MSCALE_COEF: float = 0.1
_TOTAL_STEPS: int = 14_325
_EXPERT_PARALLEL: int = 8

# Longmino side
_LONGMINO_BUCKETS = ["8k-16k", "16k-32k", "32k-64k"]
_longmino_tokenized = {
    bucket: default_tokenize(
        name=f"longmino_{bucket}_llama3", dataset=longmino_by_bucket[bucket], tokenizer=marin_tokenizer
    )
    for bucket in _LONGMINO_BUCKETS
}
_longmino_total = sum(longmino_bucket_token_counts[b] for b in _LONGMINO_BUCKETS)
_LONGMINO_WEIGHTS = {b: longmino_bucket_token_counts[b] / _longmino_total for b in _LONGMINO_BUCKETS}

# Nemotron side
_nemotron_components = {
    **tokenize_nemotron(),
    "starcoderdata": dclm_components_llama3["starcoderdata"],
    "proofpile_2": dclm_components_llama3["proofpile_2"],
}
_NEMOTRON_WEIGHTS_FULL = {**NEMOTRON_WEIGHTS, "starcoderdata": 0.25, "proofpile_2": 0.055}

_halfmix_weights = interpolate_mixture_weights([_NEMOTRON_WEIGHTS_FULL, _LONGMINO_WEIGHTS], [0.5, 0.5])
_halfmix_components = {**_nemotron_components, **_longmino_tokenized}
_halfmix = lm_mixture_data_config(_halfmix_components, _halfmix_weights)
HALFMIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    _halfmix,
    default_validation_sets(tokenizer=marin_tokenizer),
)


@dataclass(frozen=True)
class GrugMoeResumeLaunchConfig:
    model: GrugModelConfig
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
    load_checkpoint_path: str
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    expert_parallel: int = 1


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_moe_resume(config: GrugMoeResumeLaunchConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": config.expert_parallel}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=None,
        ),
        load_checkpoint=True,
        load_checkpoint_path=config.load_checkpoint_path,
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


_heuristic = MoeHeuristicV2()
# build_model_config gives sw=seq//2=16384 at seq=32k; the d=1280 baseline trained
# at sw=2048, which is what we keep on the short layers.
_base_model_cfg = dataclasses.replace(
    _heuristic.build_model_config(_DIM, seq_len=_NEW_SEQ),
    sliding_window=2048,
)
_long_mscale = _YARN_MSCALE_COEF * math.log(_NEW_SEQ / _ORIG_SEQ) + 1.0
_model_cfg = dataclasses.replace(
    _base_model_cfg,
    long_yarn_old_seq_len=_YARN_OLD_SEQ_LEN,
    long_qk_mult=_base_model_cfg.qk_mult * _long_mscale,
    yarn_alpha=_YARN_ALPHA,
    yarn_beta=_YARN_BETA,
)

_orig_tokens = float(_TOTAL_STEPS * _ORIG_BS * _ORIG_SEQ)
_optimizer = _heuristic.build_muonh_config(_NEW_BS, _orig_tokens, _DIM, seq_len=_NEW_SEQ)

_run_id = f"moe_may_compute_opt_d{_DIM}_ep{_EXPERT_PARALLEL}_32kctx_long_yarn_mscale01_halfmix_from13k"
resume_step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_grug_moe_resume,
    config=GrugMoeResumeLaunchConfig(
        model=versioned(_model_cfg),
        data=HALFMIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=_run_id,
        resources=versioned(ResourceConfig.with_tpu("v4-32")),
        steps=versioned(_TOTAL_STEPS),
        batch_size=versioned(_NEW_BS),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin_moe",
            tags=[
                "moe",
                "moe_may_compute_opt",
                f"d{_DIM}",
                f"ep{_EXPERT_PARALLEL}",
                "32kctx_long_yarn_mscale01_halfmix_resume",
            ],
            group="moe-may-compute-opt",
            name=None,
        ),
        optimizer=versioned(_optimizer),
        load_checkpoint_path=versioned(_SOURCE_CKPT_PATH),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=64,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
        expert_parallel=_EXPERT_PARALLEL,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[resume_step],
        description=(
            f"Resume d={_DIM} EP={_EXPERT_PARALLEL} (was EP=1) from step 13000 with "
            f"seq={_NEW_SEQ}, sliding_window=2048, bs={_NEW_BS}, 50/50 nemotron+longmino. "
            f"Long-only YaRN (old={_YARN_OLD_SEQ_LEN}, alpha={_YARN_ALPHA}, beta={_YARN_BETA}, "
            f"mscale_coef={_YARN_MSCALE_COEF}, long_qk_mult={_model_cfg.long_qk_mult:.4f}). "
            f"v4-32 us-central2."
        ),
    )
