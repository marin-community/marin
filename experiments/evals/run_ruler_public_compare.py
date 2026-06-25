# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a small RULER comparison against a public Hugging Face baseline."""

import argparse
import sys
from collections.abc import Sequence

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

from experiments.evals.evals import default_ruler_eval, ruler_effective_context_length
from experiments.evals.task_configs import RULER_MAX_GENERATION_TOKENS

PUBLIC_BASELINE_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_CONTEXT_LENGTHS = (4096, 8192)
DEFAULT_TASK_NAMES = ("niah_single_1",)


def _csv_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list."""
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated integers, got {value!r}.") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return parsed


def _csv_strings(value: str) -> tuple[str, ...]:
    """Parse a comma-separated string list."""
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one task name.")
    return parsed


def _effective_context(max_length: int, sliding_window: int | None) -> int:
    """Resolve the RULER target context length for one model."""
    return ruler_effective_context_length(max_length, sliding_window)


def _ruler_step(
    *,
    model: str,
    max_length: int,
    sliding_window: int | None,
    resource: str,
    context_lengths: Sequence[int],
    task_names: Sequence[str],
    max_eval_instances: int | None,
    tokenizer: str | None,
    max_gen_toks: int,
    apply_chat_template: bool,
    wandb_tags: list[str],
):
    """Build one vLLM-backed RULER eval step."""
    return default_ruler_eval(
        model,
        model_max_length=_effective_context(max_length, sliding_window),
        resource_config=ResourceConfig.with_tpu(resource),
        context_lengths=context_lengths,
        task_names=task_names,
        max_eval_instances=max_eval_instances,
        tokenizer=tokenizer,
        max_gen_toks=max_gen_toks,
        apply_chat_template=apply_chat_template,
        discover_latest_checkpoint=False,
        wandb_tags=wandb_tags,
    )


def _parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse script args and preserve executor args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-model",
        default=None,
        help="Optional HF/vLLM-compatible checkpoint path or Hugging Face model id to compare to the public baseline.",
    )
    parser.add_argument(
        "--candidate-tokenizer",
        default=None,
        help="Optional tokenizer id/path for the candidate model, e.g. stanford-crfm/marin-tokenizer.",
    )
    parser.add_argument(
        "--candidate-max-length",
        type=int,
        default=None,
        help="Candidate context window. Defaults to --public-max-length.",
    )
    parser.add_argument(
        "--candidate-sliding-window",
        type=int,
        default=None,
        help="Candidate sliding attention window; use this for sliding-window models.",
    )
    parser.add_argument("--public-model", default=PUBLIC_BASELINE_MODEL, help="Public Hugging Face baseline model.")
    parser.add_argument("--public-tokenizer", default=None, help="Tokenizer for the public baseline.")
    parser.add_argument("--public-max-length", type=int, default=8192, help="Public baseline context window to test.")
    parser.add_argument(
        "--public-sliding-window",
        type=int,
        default=None,
        help="Public baseline sliding attention window, if any.",
    )
    parser.add_argument("--resource", default="v6e-8", help="TPU type for each RULER eval job.")
    parser.add_argument("--context-lengths", type=_csv_ints, default=DEFAULT_CONTEXT_LENGTHS)
    parser.add_argument("--task-names", type=_csv_strings, default=DEFAULT_TASK_NAMES)
    parser.add_argument("--max-eval-instances", type=int, default=20)
    parser.add_argument("--max-gen-toks", type=int, default=RULER_MAX_GENERATION_TOKENS)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--max-parallel-jobs", type=int, default=1)
    parser.add_argument("--wandb-tag", action="append", default=[])
    return parser.parse_known_args(argv)


def main() -> None:
    """Run the public baseline and optional candidate RULER evals."""
    args, remaining = _parse_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    # Public baseline
    steps = [
        _ruler_step(
            model=args.public_model,
            max_length=args.public_max_length,
            sliding_window=args.public_sliding_window,
            resource=args.resource,
            context_lengths=args.context_lengths,
            task_names=args.task_names,
            max_eval_instances=args.max_eval_instances,
            tokenizer=args.public_tokenizer or args.public_model,
            max_gen_toks=args.max_gen_toks,
            apply_chat_template=args.apply_chat_template,
            wandb_tags=["ruler", "public-baseline", *args.wandb_tag],
        )
    ]

    # Candidate model
    if args.candidate_model is not None:
        steps.append(
            _ruler_step(
                model=args.candidate_model,
                max_length=args.candidate_max_length or args.public_max_length,
                sliding_window=args.candidate_sliding_window,
                resource=args.resource,
                context_lengths=args.context_lengths,
                task_names=args.task_names,
                max_eval_instances=args.max_eval_instances,
                tokenizer=args.candidate_tokenizer,
                max_gen_toks=args.max_gen_toks,
                apply_chat_template=args.apply_chat_template,
                wandb_tags=["ruler", "candidate", *args.wandb_tag],
            )
        )

    executor_main(
        steps=steps,
        max_concurrent=args.max_parallel_jobs,
        description="Run RULER on a public baseline and optional candidate model.",
    )


if __name__ == "__main__":
    main()
