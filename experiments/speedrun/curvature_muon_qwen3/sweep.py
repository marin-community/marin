# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-130m sweep for Curvature-corrected Muon (Nesterov + one-sided Shampoo curvature, hyperball).

    max_{X'X=I} <N_t, X> - (lambda/2) tr(X' P_t X) ,  P_t = EMA(G_t G_t')
    X_{k+1} = msign( N_t + (c_t I - lambda P_t) X_k ) ,  then MuonH hyperball reparam.

Built on the MuonH qwen3-130m speedrun submission (origin/qwen3-speedrun-muonh, c4_en/bpb = 1.1661):
all non-curvature hypers are held fixed at that submission; we sweep the curvature knobs lambda, K, alpha.
lambda = 0 reduces EXACTLY to MuonH (control). normalize_curvature renders lambda dimensionless (operator
lambda*||N||*(alpha*I - P/lmax)) so a fixed lambda transfers across layers/scales.

Compare on **eval/paloma/c4_en/bpb** (MuonH baseline 1.1661).

Submit (v6e-8 preemptible, us-east5):

    MARIN_PREFIX=gs://marin-us-east5 .venv/bin/iris --config lib/iris/config/marin.yaml job run \\
      --no-wait --preemptible --region us-east5 --extra tpu \\
      -e WANDB_API_KEY "$WANDB_API_KEY" -e MARIN_PREFIX gs://marin-us-east5 \\
      -e REGION us-east5 -e TPU_VARIANT v6e-8 \\
      -e LAMBDAS -e KS -e ALPHAS -e NORMALIZE -e RUN_TAG \\
      -- python -m experiments.speedrun.curvature_muon_qwen3.sweep
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.curvature_muon import CurvatureMuonConfig
from marin.execution.executor import ExecutorStep, executor_main

from experiments.defaults import default_train
from experiments.llama import llama_150m
from experiments.prebuilt_caches import fineweb_edu_subcache_10B
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)

REGION = os.environ.get("REGION", "us-east5")
SEQ_LEN = 4096
WANDB_ENTITY = "marin-community"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "speedrun")


def _floats(env: str, default: str) -> tuple[float, ...]:
    return tuple(float(x) for x in os.environ.get(env, default).split(","))


def _ints(env: str, default: str) -> tuple[int, ...]:
    return tuple(int(x) for x in os.environ.get(env, default).split(","))


# Curvature sweep axes. lambda=0 is the MuonH control (must reproduce 1.1661). alpha near 1 = aggressive
# curvature suppression of high-curvature directions; larger alpha (the spec's 1.2-2) is milder.
LAMBDAS: tuple[float, ...] = _floats("LAMBDAS", "0.0,0.1,1.0")
KS: tuple[int, ...] = _ints("KS", "1,2")
ALPHAS: tuple[float, ...] = _floats("ALPHAS", "1.0,1.5")
NORMALIZE: bool = os.environ.get("NORMALIZE", "true").lower() in ("1", "true", "yes")
# lambda tracks the LR schedule: lambda_t = (swept lambda = peak) * lr_t / peak_lr.
LAMBDA_TRACKS_LR: bool = os.environ.get("LAMBDA_TRACKS_LR", "false").lower() in ("1", "true", "yes")
# msign Newton-Schulz coefficients. polar_express (8-step schedule) is more precise than quintic (5);
# use BACKEND_STEPS=8 to run the full polar_express schedule (truncating to 5 drops the fine-polish steps).
COEFF_TYPE: str = os.environ.get("COEFF_TYPE", "polar_express")
BACKEND_STEPS: int = int(os.environ.get("BACKEND_STEPS", "8"))
RUN_TAG: str = os.environ.get("RUN_TAG", "")


@dataclass(frozen=True)
class SizeSpec:
    label: str
    llama_cfg: LlamaConfig
    learning_rate: float
    adam_lr: float
    train_batch_size: int
    num_train_steps: int
    tpu_variant: str
    warmup: int


_TPU_VARIANT = os.environ.get("TPU_VARIANT", "v6e-8")
# MuonH submission hypers: lr 0.02, adam_lr 4.615e-3, cosine + warmup 1000, 4959 steps, bs 128.
SIZE = SizeSpec("130m", llama_150m, 0.02, 0.004615384615384616, 128, 4959, _TPU_VARIANT, 1000)


def _to_qwen3_from_llama(llama_cfg: LlamaConfig) -> Qwen3Config:
    return Qwen3Config(
        max_seq_len=SEQ_LEN,
        hidden_dim=llama_cfg.hidden_dim,
        intermediate_dim=llama_cfg.intermediate_dim,
        num_layers=llama_cfg.num_layers,
        num_heads=llama_cfg.num_heads,
        num_kv_heads=llama_cfg.num_kv_heads,
        head_dim=getattr(llama_cfg, "head_dim", None),
        use_bias=getattr(llama_cfg, "use_bias", False),
        rope=llama_cfg.rope,
        activation_function=llama_cfg.activation_function,
        initializer_range=llama_cfg.initializer_range,
        layer_norm_epsilon=llama_cfg.layer_norm_epsilon,
        tie_word_embeddings=llama_cfg.tie_word_embeddings,
        upcast_attn=llama_cfg.upcast_attn,
        attn_backend=llama_cfg.attn_backend,
        flash_attention_block_size=llama_cfg.flash_attention_block_size,
        hybrid_norm=False,
        scan_layers=True,
        gradient_checkpointing=True,
    )


def _label(x: float) -> str:
    return f"{x:g}".replace(".", "_").replace("-", "m")


def _override_tracker(step: ExecutorStep) -> ExecutorStep:
    pod = step.config
    inner = pod.train_config
    trainer = dataclasses.replace(
        inner.trainer, tracker=dataclasses.replace(inner.trainer.tracker, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    )
    inner = dataclasses.replace(inner, trainer=trainer)
    return dataclasses.replace(step, config=dataclasses.replace(pod, train_config=inner))


def _build_step(lam: float, k: int, alpha: float) -> ExecutorStep:
    optimizer = CurvatureMuonConfig(
        adam_lr=SIZE.adam_lr,
        momentum=0.95,
        nesterov=True,
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-15,
        muon_epsilon=1e-5,
        max_grad_norm=1.0,
        backend_steps=BACKEND_STEPS,
        coefficient_type=COEFF_TYPE,
        curvature_beta=0.95,
        curvature_lambda=lam,
        curvature_alpha=alpha,
        inner_steps=k,
        normalize_curvature=NORMALIZE,
        lambda_tracks_lr=LAMBDA_TRACKS_LR,
        learning_rate=SIZE.learning_rate,
        lr_schedule="cosine",
        warmup=SIZE.warmup,
        min_lr_ratio=0.0,
    )
    train = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(SIZE.tpu_variant, regions=[REGION], preemptible=True),
        train_seq_len=SEQ_LEN,
        train_batch_size=SIZE.train_batch_size,
        num_train_steps=SIZE.num_train_steps,
        learning_rate=SIZE.learning_rate,
        optimizer_config=optimizer,
    )
    cf = {"polar_express": "pe", "quintic": "qx", "simple": "sm", "aol": "aol"}.get(COEFF_TYPE, COEFF_TYPE)
    tag = f"_{cf}{BACKEND_STEPS}{('_' + RUN_TAG) if RUN_TAG else ''}"
    if lam == 0.0:
        clabel = "_lam0"  # MuonH control (alpha/K/normalize irrelevant)
    else:
        nlabel = "n" if NORMALIZE else "r"
        tlabel = "_tl" if LAMBDA_TRACKS_LR else ""
        clabel = f"_lam{_label(lam)}_a{_label(alpha)}_K{k}_{nlabel}{tlabel}"
    run_id = f"qwen3_{SIZE.label}_curvmuon{clabel}{tag}_{SEQ_LEN}"
    step = default_train(
        name=run_id,
        tokenized=fineweb_edu_subcache_10B,
        model_config=_to_qwen3_from_llama(SIZE.llama_cfg),
        train_config=train,
        tags=["speedrun", "curvature_muon", f"qwen3_{SIZE.label}"],
        use_default_validation=True,
        eval_harness_tasks=(),
        wandb_name=run_id,
        wandb_group="curvature_muon_qwen3_130m",
    )
    return _override_tracker(step)


def main() -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return
    # lambda=0 control once; lambda>0 across alpha x K.
    combos: list[tuple[float, int, float]] = []
    seen_control = False
    for lam in LAMBDAS:
        if lam == 0.0:
            if not seen_control:
                combos.append((0.0, 1, ALPHAS[0]))
                seen_control = True
            continue
        for alpha in ALPHAS:
            for k in KS:
                combos.append((lam, k, alpha))
    steps = [_build_step(lam, k, alpha) for (lam, k, alpha) in combos]
    logger.info(
        "Curvature-Muon 130m sweep (normalize=%s): %d runs lambdas=%s alphas=%s Ks=%s",
        NORMALIZE,
        len(steps),
        LAMBDAS,
        ALPHAS,
        KS,
    )
    executor_main(steps=steps, description="Qwen3-130m curvature-corrected Muon sweep: lambda x alpha x K.")


if __name__ == "__main__":
    main()
