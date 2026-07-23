# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified evalchemy launcher: serve a model, eval it, and record the result as a composable
``ArtifactStep`` whose identity (model + suite + stage + seeds + version) gates caching and
force-reeval.

Backend-agnostic: the caller picks TPU or GPU via ``ServeSpec``. The actual serve + eval + teardown
pipeline is delegated to :func:`~experiments.evals.evalchemy.serve_and_eval.serve_and_eval`, which
submits a serve child (vLLM or Levanter) and an eval child (evalchemy as an OpenAI client) as Iris
jobs. Per-model vLLM flags (GDN triton backend, multimodal limits, reasoning parser, native context
cap) are auto-derived from the model's ``config.json`` inside ``serve_and_eval``.

Replaces ``marin_evalchemy_tpu.py`` (inline TPU eval, pre-``serve_and_eval``) with the decoupled
serve+eval path for both TPU and GPU.

MODEL → EVAL FLOW::

    EvalSpec(model, suite, serve=ServeSpec(...)) → evalchemy_step(spec) → ArtifactStep
        run() builds EvalchemyEvalConfig → serve_and_eval(config)
            ├── serve child  (vLLM/Levanter on TPU/GPU)  ──▶ OpenAI endpoint
            └── eval child   (:evalchemy-tpu, CPU)        ──local-completions──▶ endpoint

Run (the pipeline coordinator is an Iris job — ``serve_and_eval`` submits children from inside it)::

    uv run iris --cluster=marin job run --job-name eval-<run> --region us-east5 \\
      --cpu 1 --memory 2G --priority interactive --no-wait \\
      -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.evals.evalchemy.marin_evalchemy \\
         --model laion/delphi-1e22-magpie-levanter-parity --suite delphi_math --stage sft
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace

from marin.evaluation.eval_result import EvalchemyResult
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower
from marin.execution.step_runner import StepRunner

from experiments.evals.evalchemy.serve_and_eval import (
    DEFAULT_NUM_CONCURRENT,
    EvalchemyEvalConfig,
    ServeBackend,
    ServeSpec,
    serve_and_eval,
)

# The suite → task-name preset map. These are the current runnable evalchemy tasks; the full
# registry (including benchmarks that need image dependencies not yet present) lives in
# ``experiments.evaluation.evals``.
SUITE_TO_TASKS: dict[str, tuple[str, ...]] = {
    "delphi_math": ("MATH500", "AIME24"),
    "math": ("MATH500", "AIME24", "AMC23", "OlympiadBench"),
    "gsm8k": ("gsm8k",),
    "tier1": (
        "gsm8k",
        "mmlu",
        "hellaswag",
        "arc_challenge",
        "arc_easy",
        "piqa",
        "winogrande",
        "openbookqa",
        "boolq",
        "truthfulqa_mc2",
        "lambada_openai",
        "triviaqa",
        "nq_open",
        "drop",
    ),
    "tier2": ("MATH500", "AIME24", "OlympiadBench"),
}

logger = logging.getLogger(__name__)


def _tasks(spec: EvalSpec) -> tuple[EvalTaskConfig, ...]:
    """Build the task list from the suite + spec, expanding multi-seed AIME24 when requested."""
    if spec.suite not in SUITE_TO_TASKS:
        raise ValueError(f"unknown suite {spec.suite!r}; choices={sorted(SUITE_TO_TASKS)}")

    tasks: list[EvalTaskConfig] = []
    for name in SUITE_TO_TASKS[spec.suite]:
        if name == "AIME24":
            for seed in spec.seeds:
                tasks.append(
                    EvalTaskConfig(
                        name="AIME24",
                        num_fewshot=0,
                        task_alias=f"AIME24_seed{seed}",
                        task_kwargs={"seed": seed},
                    )
                )
            continue
        tasks.append(
            EvalTaskConfig(
                name,
                _SHOT_MAP.get(name, 0),
                generation=name in _GENERATIVE_TASKS,
            )
        )
    return tuple(tasks)


# Per-task few-shot defaults (POLICY §3; tier-1 lm-harness tasks).
_SHOT_MAP: dict[str, int] = {
    "mmlu": 5,
    "hellaswag": 10,
    "arc_challenge": 25,
    "winogrande": 5,
    "triviaqa": 5,
    "nq_open": 5,
    "drop": 3,
}

_GENERATIVE_TASKS = frozenset({"gsm8k", "triviaqa", "nq_open", "drop", "MATH500", "OlympiadBench"})


@dataclass(frozen=True)
class EvalSpec:
    """One evalchemy eval of one model, backend-agnostic.

    Parameters
    ----------
    run_name:
        OT-Agent RUN_NAME analog → output paths + wandb. Used in the ArtifactStep identity.
    model:
        HF repo id or object-store (``gs://``) path of the model to serve and eval.
    suite:
        Suite name from :data:`SUITE_TO_TASKS`.
    stage:
        ``sft`` or ``rl`` → ``--apply_chat_template`` ON; ``base`` → OFF.
    serve:
        Backend + slice config (:class:`ServeSpec`). Defaults to TPU v6e-8 vLLM. Pass
        ``ServeSpec(gpu_type="H100", gpu_count=8, tpu_type=None)`` for GPU.
    seeds:
        AIME24 runs one process per seed. Default ``(42,)``; use ``range(42, 52)`` for 10-seed.
    max_gen_toks:
        Generation budget. Must be < ``serve.max_model_len``.
    version:
        Bump to force a re-eval (identity change). The OT-Agent force-reeval analog.
    """

    run_name: str
    model: str
    suite: str = "delphi_math"
    stage: str = "sft"
    serve: ServeSpec | None = None
    seeds: tuple[int, ...] = (42,)
    max_gen_toks: int = 3584
    max_model_len: int | None = None
    version: str = "2026.07.15"
    tokenizer: str | None = None
    num_concurrent: int = DEFAULT_NUM_CONCURRENT
    max_eval_instances: int | None = None
    extra_gen_kwargs: dict[str, str] | None = None
    region: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in {"sft", "rl", "base"}:
            raise ValueError(f"unknown stage {self.stage!r}")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if self.max_gen_toks <= 0:
            raise ValueError("max_gen_toks must be positive")


def _serve_spec(spec: EvalSpec) -> ServeSpec:
    """Resolve this launcher's TPU default without overriding an explicit serve configuration."""
    if spec.serve is None:
        return ServeSpec(
            backend=ServeBackend.VLLM,
            tpu_type="v6e-8",
            gpu_type=None,
            gpu_count=None,
            max_model_len=spec.max_model_len,
            region=spec.region,
        )
    if spec.max_model_len is not None and spec.serve.max_model_len is None:
        return replace(spec.serve, max_model_len=spec.max_model_len)
    return spec.serve


def _build_config(spec: EvalSpec) -> EvalchemyEvalConfig:
    """Build the EvalchemyEvalConfig from the EvalSpec."""
    tasks = _tasks(spec)
    apply_chat_template = spec.stage in {"sft", "rl"}
    return EvalchemyEvalConfig(
        model=spec.model,
        tokenizer=spec.tokenizer,
        tasks=tasks,
        serve=_serve_spec(spec),
        apply_chat_template=apply_chat_template,
        max_gen_toks=spec.max_gen_toks,
        max_eval_instances=spec.max_eval_instances,
        num_concurrent=spec.num_concurrent,
        extra_gen_kwargs=spec.extra_gen_kwargs or {},
    )


def evalchemy_step(spec: EvalSpec) -> ArtifactStep[EvalchemyResult]:
    """The eval as a lazy ``ArtifactStep``.

    Identity = model + suite + stage + seeds + version. Re-running the pipeline skips the eval
    when nothing changed; bumping ``version`` forces a re-eval.

    The container image + TPU/GPU slice ride on ``ResourceConfig`` via ``serve_and_eval``'s own
    Iris job submission — this step's ``run()`` calls ``serve_and_eval`` directly (it is already an
    Iris orchestrator that submits serve + eval children), so no outer ``remote()`` wrapper is needed.
    """

    def build_config(ctx: StepContext) -> dict:
        serve = _serve_spec(spec)
        return {
            "model": spec.model,
            "suite": spec.suite,
            "stage": spec.stage,
            "seeds": list(spec.seeds),
            "tokenizer": spec.tokenizer,
            "max_gen_toks": spec.max_gen_toks,
            "max_model_len": serve.max_model_len,
            "num_concurrent": spec.num_concurrent,
            "max_eval_instances": spec.max_eval_instances,
            "extra_gen_kwargs": spec.extra_gen_kwargs or {},
            "serve": {
                "backend": serve.backend.value,
                "dtype": serve.dtype,
                "tensor_parallel_size": serve.tensor_parallel_size,
                "vllm_extra_args": list(serve.vllm_extra_args),
                "chat_template_content": serve.chat_template_content,
                "auto_overrides": serve.auto_overrides,
            },
            "out": ctx.output_path,
        }

    def run(cfg: dict) -> EvalchemyResult:
        config = _build_config(spec)
        config = replace(config, out_path=cfg["out"])
        result = serve_and_eval(config)
        logger.info("Evalchemy artifacts: %s; child jobs: %s", result.out_path, result.jobs)
        return EvalchemyResult(path=result.out_path)

    return ArtifactStep(
        name=f"evals/{spec.run_name}/{spec.suite}",
        version=spec.version,
        artifact_type=EvalchemyResult,
        run=run,
        build_config=build_config,
        deps=(),
    )


# A default worked example: the parity model on the Delphi math suite.
SPEC = EvalSpec(
    run_name="delphi-1e22-magpie-levanter-parity",
    model="laion/delphi-1e22-magpie-levanter-parity",
    suite="delphi_math",
    stage="sft",
    seeds=tuple(range(42, 52)),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="laion/delphi-1e22-magpie-levanter-parity")
    parser.add_argument("--run_name", default="eval-run")
    parser.add_argument("--suite", default="delphi_math", choices=sorted(SUITE_TO_TASKS))
    parser.add_argument("--stage", default="sft", choices=["sft", "rl", "base"])
    parser.add_argument("--version", default="2026.07.15")
    parser.add_argument("--tpu_type", default="v6e-8")
    parser.add_argument("--gpu_type", default=None)
    parser.add_argument("--gpu_count", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_gen_toks", type=int, default=3584)
    parser.add_argument("--region", default=None)
    parser.add_argument("--tp", type=int, default=None, help="tensor_parallel_size")
    args = parser.parse_args()

    tp_size = args.tp or (8 if args.gpu_count else None)
    serve = ServeSpec(
        backend=ServeBackend.VLLM,
        tpu_type=args.tpu_type if args.gpu_type is None else None,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
        max_model_len=args.max_model_len,
        tensor_parallel_size=tp_size,
        region=args.region,
    )
    spec = EvalSpec(
        run_name=args.run_name,
        model=args.model,
        suite=args.suite,
        stage=args.stage,
        serve=serve,
        max_gen_toks=args.max_gen_toks,
        max_model_len=args.max_model_len,
        version=args.version,
        region=args.region,
    )
    StepRunner().run([lower(evalchemy_step(spec))])


if __name__ == "__main__":
    main()
