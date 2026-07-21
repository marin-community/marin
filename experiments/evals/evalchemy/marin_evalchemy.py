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
from collections.abc import Sequence

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.lazy import Artifact, ArtifactStep, StepContext, lower
from marin.execution.step_runner import StepRunner

from experiments.evals.evalchemy.serve_and_eval import (
    DEFAULT_NUM_CONCURRENT,
    EvalchemyEvalConfig,
    ServeBackend,
    ServeSpec,
    serve_and_eval,
)

# The suite → task-name preset map (evalchemy fork registry).
SUITE_TO_TASKS: dict[str, list[str]] = {
    "delphi_math": ["MATH500", "AIME24"],
    "math": ["MATH500", "AIME24", "AMC23", "OLYMPIADBENCH"],
    "gsm8k": ["gsm8k"],
    "tier1": [
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
    ],
    "tier2": ["MATH500", "HumanEvalPlus", "MBPPPlus", "GPQADiamond", "IFEval"],
}


def _tasks(spec: EvalSpec) -> tuple[EvalTaskConfig, ...]:
    """Build the task list from the suite + spec, expanding multi-seed AIME24 when requested."""
    if spec.suite not in SUITE_TO_TASKS:
        raise ValueError(f"unknown suite {spec.suite!r}; choices={sorted(SUITE_TO_TASKS)}")

    tasks: list[EvalTaskConfig] = []
    for name in SUITE_TO_TASKS[spec.suite]:
        if name == "AIME24" and len(spec.seeds) > 1:
            for seed in spec.seeds:
                tasks.append(
                    EvalTaskConfig(
                        name="AIME24",
                        num_fewshot=0,
                        task_alias=f"AIME24_seed{seed}",
                        task_kwargs={"seed": seed},
                    )
                )
        else:
            shots = _SHOT_MAP.get(name, 0)
            tasks.append(EvalTaskConfig(name, shots))
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


class EvalSpec:
    """One evalchemy eval of one model, backend-agnostic.

    Parameters
    ----------
    run_name:
        OT-Agent RUN_NAME analog → output paths + wandb. Used in the ArtifactStep identity.
    model:
        HF repo id or object-store (``gs://``) path of the model to serve and eval.
    suite:
        Suite name from :data:`SUITE_TO_TASKS`. Ignored if ``tasks`` is provided directly.
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

    def __init__(
        self,
        run_name: str,
        model: str,
        suite: str = "delphi_math",
        stage: str = "sft",
        serve: ServeSpec | None = None,
        seeds: Sequence[int] = (42,),
        max_gen_toks: int = 3584,
        max_model_len: int | None = None,
        version: str = "2026.07.15",
        tokenizer: str | None = None,
        num_concurrent: int = DEFAULT_NUM_CONCURRENT,
        max_eval_instances: int | None = None,
        extra_gen_kwargs: dict[str, str] | None = None,
        region: str | None = None,
    ):
        self.run_name = run_name
        self.model = model
        self.suite = suite
        self.stage = stage
        self.seeds = tuple(seeds)
        self.max_gen_toks = max_gen_toks
        self.version = version
        self.tokenizer = tokenizer
        self.num_concurrent = num_concurrent
        self.max_eval_instances = max_eval_instances
        self.extra_gen_kwargs = extra_gen_kwargs or {}

        # Default serve spec: TPU v6e-8, vLLM backend.
        if serve is not None:
            self.serve = serve
        else:
            self.serve = ServeSpec(
                backend=ServeBackend.VLLM,
                tpu_type="v6e-8",
                gpu_type=None,
                gpu_count=None,
                max_model_len=max_model_len,
                region=region,
            )
        if max_model_len is not None and self.serve.max_model_len is None:
            self.serve = ServeSpec(**{**self.serve.__dict__, "max_model_len": max_model_len})


def _build_config(spec: EvalSpec) -> EvalchemyEvalConfig:
    """Build the EvalchemyEvalConfig from the EvalSpec."""
    tasks = _tasks(spec)
    apply_chat_template = spec.stage in {"sft", "rl"}
    return EvalchemyEvalConfig(
        model=spec.model,
        tokenizer=spec.tokenizer,
        tasks=tasks,
        serve=spec.serve,
        apply_chat_template=apply_chat_template,
        max_gen_toks=spec.max_gen_toks,
        max_eval_instances=spec.max_eval_instances,
        num_concurrent=spec.num_concurrent,
        extra_gen_kwargs=spec.extra_gen_kwargs,
    )


def evalchemy_step(spec: EvalSpec) -> ArtifactStep[Artifact]:
    """The eval as a lazy ``ArtifactStep``.

    Identity = model + suite + stage + seeds + version. Re-running the pipeline skips the eval
    when nothing changed; bumping ``version`` forces a re-eval.

    The container image + TPU/GPU slice ride on ``ResourceConfig`` via ``serve_and_eval``'s own
    Iris job submission — this step's ``run()`` calls ``serve_and_eval`` directly (it is already an
    Iris orchestrator that submits serve + eval children), so no outer ``remote()`` wrapper is needed.
    """

    def build_config(ctx: StepContext) -> dict:
        return {
            "model": spec.model,
            "suite": spec.suite,
            "stage": spec.stage,
            "seeds": list(spec.seeds),
            "out": ctx.output_path,
        }

    def run(cfg: dict) -> None:
        config = _build_config(spec)
        # If the caller gave an explicit out_path, use it; otherwise the step's output_path
        # (a gs:// artifact path) becomes the eval's out_path.
        if not config.out_path:
            config = EvalchemyEvalConfig(**{**config.__dict__, "out_path": cfg["out"]})
        result = serve_and_eval(config)
        print(f"evalchemy artifacts: {result.out_path}  jobs: {result.jobs}")

    return ArtifactStep(
        name=f"evals/{spec.run_name}/{spec.suite}",
        version=spec.version,
        artifact_type=Artifact,
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
