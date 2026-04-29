# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from marin.inference.served_model import RunningModel

LmEvalModelArgValue = str | int | float | bool


@dataclass(frozen=True)
class LmEvalRun:
    """A single lm-eval run against an already served model."""

    tasks: Sequence[str]
    output_path: str
    apply_chat_template: bool = False
    limit: int | None = None
    num_fewshot: int | None = None
    batch_size: int | str | None = None
    extra_model_args: Mapping[str, LmEvalModelArgValue] = field(default_factory=dict)


def run_lm_eval(model: RunningModel, run: LmEvalRun) -> None:
    """Run lm-eval against a launcher-neutral served model."""
    if not run.tasks:
        raise ValueError("LmEvalRun.tasks must contain at least one task.")

    from lm_eval.evaluator import simple_evaluate
    from lm_eval.loggers import EvaluationTracker

    evaluation_tracker = EvaluationTracker(output_path=run.output_path)
    results = simple_evaluate(
        model=_lm_eval_model_name(run),
        tasks=list(run.tasks),
        num_fewshot=run.num_fewshot,
        model_args=build_lm_eval_model_args(model, run),
        apply_chat_template=run.apply_chat_template,
        batch_size=run.batch_size,
        confirm_run_unsafe_code=True,
        limit=run.limit,
        evaluation_tracker=evaluation_tracker,
        log_samples=True,
    )
    if results is None:
        raise RuntimeError("lm-eval returned no results.")

    samples = results.pop("samples")
    evaluation_tracker.save_results_aggregated(results=results, samples=samples)
    for task_name in results["configs"].keys():
        evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])


def build_lm_eval_model_args(model: RunningModel, run: LmEvalRun) -> str:
    """Build the comma-delimited model_args string consumed by lm-eval API models."""
    model_args: dict[str, object] = {
        "model": model.endpoint.model,
        "base_url": _lm_eval_base_url(model, run),
        "tokenizer_backend": "huggingface",
        "tokenized_requests": False,
    }
    if model.endpoint.api_key is not None:
        model_args["api_key"] = model.endpoint.api_key
    if model.tokenizer is not None:
        model_args["tokenizer"] = model.tokenizer
    model_args.update(run.extra_model_args)
    return ",".join(
        f"{_format_model_arg_key(key)}={_format_model_arg_value(value)}" for key, value in model_args.items()
    )


def _lm_eval_model_name(run: LmEvalRun) -> str:
    if run.apply_chat_template:
        return "local-chat-completions"
    return "local-completions"


def _lm_eval_base_url(model: RunningModel, run: LmEvalRun) -> str:
    endpoint = "chat/completions" if run.apply_chat_template else "completions"
    return f"{model.endpoint.base_url.rstrip('/')}/{endpoint}"


def _format_model_arg_key(key: str) -> str:
    if not key:
        raise ValueError("lm-eval model_args keys must be non-empty.")
    if "," in key or "=" in key:
        raise ValueError(f"lm-eval model_args key cannot contain ',' or '=': {key!r}")
    return key


def _format_model_arg_value(value: object) -> str:
    text = str(value)
    if "," in text:
        raise ValueError(f"lm-eval model_args value cannot contain ',': {text!r}")
    return text
