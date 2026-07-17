# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone evalchemy-on-TPU eval launcher for freshly-SFT'd models.

The eval BACKEND is the ``marin-community/evalchemy`` FORK (@ ``a84543c`` main HEAD; bundles MATH500/AIME24/
gsm8k + the boxed-answer graders + the ``eval/lm_eval_compat.py`` shim; lm-eval EleutherAI v0.4.12),
invoked as ``eval.eval`` INSIDE a pinned ``:evalchemy-tpu`` container on a TPU pod. Although it lives
under ``experiments/evals`` it does NOT import marin's existing eval harness there
(``evals.py``/``task_configs.py``) — the container is the portability boundary. The vLLM engine is
``vllm-tpu==0.20.0`` REUSED from the OT-Agent ``:tpu`` base image (the known-good version the
``eval-agentic-launch-iris`` TPU path already runs on) — no unproven TPU port.

Parameterization modeled on the OT-Agent ``python -m hpc.launch --job_type eval_...`` front door:
a ``suite`` -> tasks preset map, a per-model ``run_name``, a chat-template ``stage`` flag, a
``version`` bump = force-reeval, and a concurrency cap.

MODEL -> EXPORT -> EVAL FLOW:
  ``sft_step`` (``experiments.sft_launcher.marin_sft_launcher``) -> LevanterCheckpoint -> HF export at
  ``<out>/hf/step-<N>`` -> (optional ``hf upload`` to ``laion/<run>``) -> ``EvalSpec.model`` = that
  gs:// path OR the HF id -> ``evalchemy_tpu_step`` runs MATH500/AIME24 in the container on TPU ->
  ``results_*.json``.

RUN (CPU coordinator dispatches the eval sub-job into the :evalchemy-tpu container on a TPU slice):

    uv run iris --cluster=marin job run --job-name eval-<run> --region us-east5 \
      --cpu 1 --memory 2G --extra cpu --priority interactive --no-wait \
      -e MARIN_PREFIX gs://marin-us-east5 -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \
      -- python -m experiments.evals.evalchemy.marin_evalchemy_tpu
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass

import fsspec
from fray.types import ResourceConfig
from marin.execution.lazy import Artifact, ArtifactStep, StepContext, lower
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner

from experiments.evals.evalchemy.image import EVALCHEMY_IMAGE, EVALCHEMY_PYTHON

# The OT-Agent "--preset" analog: suite name -> lm-eval task names (evalchemy fork registry).
SUITE_TO_TASKS: dict[str, list[str]] = {
    "delphi_math": ["MATH500", "AIME24"],  # the Delphi #6279 core (AIME24 = 10-seed at run time)
    "math": ["MATH500", "AIME24", "AMC23", "OLYMPIADBENCH"],
    "gsm8k": ["gsm8k"],
}


@dataclass(frozen=True)
class EvalSpec:
    """One evalchemy-on-TPU eval of one model."""

    run_name: str  # OT-Agent RUN_NAME analog -> output paths + wandb
    model: str  # gs://.../hf/step-N  OR  laion/<run> (HF id) — the SFT export
    suite: str = "delphi_math"  # -> SUITE_TO_TASKS[suite]
    stage: str = "sft"  # sft|rl -> --apply_chat_template ON; base -> OFF (OT-Agent STAGE)
    max_model_len: int = 4096  # Delphi 4k context (a 4k model hard-rejects a 32k request)
    max_gen_toks: int = 3584
    seeds: Sequence[int] = (42,)  # AIME24 uses 42..51 (10-seed) at run time
    version: str = "2026.07.15"  # bump / "-dev" = force-reeval (OT-Agent force-reeval analog)
    tpu_type: str = "v6e-4"  # eval default matches eval-agentic-launch-iris (--tpu v6e-4)
    image: str = EVALCHEMY_IMAGE
    # Pin the TPU slice to a region (e.g. "us-east5"); None lets the scheduler place it. Set it when
    # submitting off-cluster (e.g. CI) to colocate the slice with its artifacts and avoid cross-region
    # I/O -- there is no coordinator region to inherit off-cluster.
    region: str | None = None


def _tasks(spec: EvalSpec) -> list[str]:
    if spec.suite not in SUITE_TO_TASKS:
        raise ValueError(f"unknown suite {spec.suite!r}; choices={sorted(SUITE_TO_TASKS)}")
    return SUITE_TO_TASKS[spec.suite]


def _run_evalchemy(spec: EvalSpec, out_path: str) -> None:
    """Runs INSIDE the :evalchemy-tpu container on a TPU pod. Drives the fork CLI per seed.

    Loops the requested seeds (AIME24 wants 42..51). evalchemy/lm-eval write ``results_*.json`` to the
    LOCAL filesystem, so each seed runs against a local temp dir which is then copied to
    ``out_path/seed<N>/`` (a Marin ``gs://``/``s3://`` artifact path via fsspec) for downstream harvest.
    """
    tasks = ",".join(_tasks(spec))
    apply_ct = spec.stage in {"sft", "rl"}
    out_fs, _ = fsspec.core.url_to_fs(out_path)
    for seed in spec.seeds:
        dest = os.path.join(out_path, f"seed{seed}")
        with tempfile.TemporaryDirectory() as local_out:
            cmd = [
                EVALCHEMY_PYTHON,
                "-m",
                "eval.eval",
                "--model",
                "vllm",
                "--model_args",
                f"pretrained={spec.model},tensor_parallel_size=1,max_model_len={spec.max_model_len}",
                "--tasks",
                tasks,
                "--gen_kwargs",
                f"max_gen_toks={spec.max_gen_toks}",
                "--seed",
                str(seed),
                "--output_path",
                local_out,
            ]
            if apply_ct:
                cmd.append("--apply_chat_template")  # picks up the model repo's baked delphi_v0 template
            subprocess.run(cmd, check=True)
            # Upload the local results tree to the Marin artifact path (no-op-equivalent for a local dest).
            out_fs.put(local_out, dest, recursive=True)


def evalchemy_tpu_step(spec: EvalSpec) -> ArtifactStep[Artifact]:
    """The eval as a lazy ArtifactStep. Identity = model + suite + stage + len + seeds + version.

    The container image + TPU slice ride on ``ResourceConfig`` (``.image`` is the per-task container
    override, ``fray.types``); resources are a runtime arg -> excluded from the fingerprint, so
    re-pinning the image/slice never forks identity.
    """
    resources = ResourceConfig.with_tpu(spec.tpu_type, image=spec.image, regions=[spec.region] if spec.region else None)

    def build_config(ctx: StepContext) -> dict:
        return {
            "model": spec.model,
            "tasks": _tasks(spec),
            "stage": spec.stage,
            "max_model_len": spec.max_model_len,
            "seeds": list(spec.seeds),
            "out": ctx.output_path,
        }

    def run(cfg: dict) -> None:
        remote(_run_evalchemy, resources=resources)(spec, cfg["out"])

    return ArtifactStep(
        name=f"evals/{spec.run_name}/{spec.suite}",
        version=spec.version,
        artifact_type=Artifact,
        run=run,
        build_config=build_config,
        deps=(),
        runtime_args={"eval_resources": resources},
    )


# A default worked example: the parity model on the Delphi math suite.
SPEC = EvalSpec(
    run_name="delphi-1e22-magpie-levanter-parity",
    model="laion/delphi-1e22-magpie-levanter-parity",
    suite="delphi_math",
    stage="sft",
    seeds=tuple(range(42, 52)),  # AIME24 10-seed (42..51); MATH500 uses seed 42 only at harvest
)


if __name__ == "__main__":
    StepRunner().run([lower(evalchemy_tpu_step(SPEC))])
