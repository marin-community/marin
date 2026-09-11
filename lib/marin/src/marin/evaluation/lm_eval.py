# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from marin.execution.artifact import Artifact
from marin.inference.types import RunningModel

LmEvalModelArgValue = str | int | float | bool
LM_EVAL_UV_PACKAGES = (
    "lm-eval[api]@git+https://github.com/EleutherAI/lm-evaluation-harness@f4d4b3de3ee6741a7151a9fe74945ee515262f4c",
    "transformers<5",
)


class LmEvalResults(Artifact):
    """A lazy reference to lm-eval metrics and samples."""

    results_path: str


class LmEvalAdapter(StrEnum):
    """lm-eval OpenAI-compatible adapter names supported by the served runner."""

    LOCAL_COMPLETIONS = "local-completions"
    LOCAL_CHAT_COMPLETIONS = "local-chat-completions"

    @property
    def endpoint_path(self) -> str:
        match self:
            case LmEvalAdapter.LOCAL_COMPLETIONS:
                return "completions"
            case LmEvalAdapter.LOCAL_CHAT_COMPLETIONS:
                return "chat/completions"


@dataclass(frozen=True)
class LmEvalRun:
    """A single lm-eval run against an already served model."""

    tasks: Sequence[str]
    adapter: LmEvalAdapter = LmEvalAdapter.LOCAL_COMPLETIONS
    apply_chat_template: bool = False
    limit: int | None = None
    num_fewshot: int | None = None
    batch_size: int | str | None = None
    confirm_run_unsafe_code: bool = False
    extra_model_args: Mapping[str, LmEvalModelArgValue] = field(default_factory=dict)


def run_lm_eval(model: RunningModel, run: LmEvalRun, output_path: str) -> None:
    """Evaluate tasks against a served model and persist metrics and samples."""
    if not run.tasks:
        raise ValueError("LmEvalRun.tasks must contain at least one task.")

    command = ["uv", "run", "--isolated", "--no-project"]
    for package in LM_EVAL_UV_PACKAGES:
        command.extend(["--with", package])
    command.extend(
        [
            "lm_eval",
            "--model",
            run.adapter.value,
            "--model_args",
            build_lm_eval_model_args(model, run),
            "--tasks",
            ",".join(run.tasks),
            "--output_path",
            output_path,
            "--log_samples",
        ]
    )
    if run.apply_chat_template:
        command.append("--apply_chat_template")
    if run.confirm_run_unsafe_code:
        command.append("--confirm_run_unsafe_code")
    if run.limit is not None:
        command.extend(["--limit", str(run.limit)])
    if run.num_fewshot is not None:
        command.extend(["--num_fewshot", str(run.num_fewshot)])
    if run.batch_size is not None:
        command.extend(["--batch_size", str(run.batch_size)])
    subprocess.run(command, check=True)


def build_lm_eval_model_args(model: RunningModel, run: LmEvalRun) -> str:
    """Build the comma-delimited model_args string consumed by lm-eval API models."""
    model_args: dict[str, object] = {
        "model": model.endpoint.model,
        "base_url": model.endpoint.url(run.adapter.endpoint_path),
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
