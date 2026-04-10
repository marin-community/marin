# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
from contextlib import ExitStack

from rigging.log_setup import configure_logging

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment
from marin.test_time_scaling import (
    DEFAULT_REASONING_SELECTORS,
    CandidateGenerationConfig,
    OpenAIChatCompletionProvider,
    SelectorName,
    TestTimeScalingConfig,
    build_run_summary,
    generate_candidates,
    load_prompt_manifest,
    replay_selectors,
    write_candidate_records,
    write_prompt_manifest,
    write_run_summary,
    write_selection_records,
)

logger = logging.getLogger(__name__)


def _parse_selector_names(raw_selectors: list[str] | None) -> tuple[SelectorName, ...]:
    if not raw_selectors:
        return DEFAULT_REASONING_SELECTORS
    return tuple(SelectorName(raw_selector) for raw_selector in raw_selectors)


def _parse_engine_kwargs(raw_engine_kwargs: str) -> dict:
    try:
        engine_kwargs = json.loads(raw_engine_kwargs)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --engine-kwargs-json: {exc}") from exc
    if not isinstance(engine_kwargs, dict):
        raise ValueError("--engine-kwargs-json must decode to a JSON object")
    return engine_kwargs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sample-only reasoning TTS against a prompt manifest.")
    parser.add_argument("--manifest", required=True, help="Prompt manifest directory or prompts.jsonl path")
    parser.add_argument("--output-dir", required=True, help="Directory where TTS artifacts should be written")
    parser.add_argument("--model", required=True, help="Model name or request model id")
    parser.add_argument("--model-path", help="Optional checkpoint path when launching a local vLLM server")
    parser.add_argument("--server-url", help="Existing OpenAI-compatible /v1 server URL")
    parser.add_argument("--vllm-mode", choices=["native", "docker"], help="Mode to use when launching vLLM")
    parser.add_argument(
        "--engine-kwargs-json",
        default="{}",
        help="JSON object of engine kwargs forwarded when launching vLLM",
    )
    parser.add_argument(
        "--selector",
        dest="selectors",
        action="append",
        choices=[selector.value for selector in SelectorName],
        help="Selector to evaluate. Can be repeated. Defaults to the built-in set.",
    )
    parser.add_argument("--num-candidates", type=int, default=4, help="Number of candidates to sample per prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling threshold")
    parser.add_argument("--max-gen-toks", type=int, default=2048, help="Maximum generated tokens per candidate")
    parser.add_argument("--seed", type=int, help="Base request seed")
    parser.add_argument("--request-timeout", type=float, default=600.0, help="HTTP timeout for each request")
    parser.add_argument("--startup-timeout", type=int, default=3600, help="Timeout when launching a local vLLM server")
    parser.add_argument("--api-key", default="marin-tts", help="API key for OpenAI-compatible servers")
    parser.add_argument(
        "--extra-vllm-arg",
        action="append",
        default=[],
        help="Extra CLI argument passed through to `vllm serve`. Can be repeated.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    configure_logging(level=logging.INFO)

    manifest = load_prompt_manifest(args.manifest)
    run_config = TestTimeScalingConfig(
        generation=CandidateGenerationConfig(
            num_candidates=args.num_candidates,
            temperature=args.temperature,
            top_p=args.top_p,
            max_gen_toks=args.max_gen_toks,
            seed=args.seed,
        ),
        selectors=_parse_selector_names(args.selectors),
    )

    write_prompt_manifest(args.output_dir, manifest)

    with ExitStack() as stack:
        if args.server_url:
            server_url = args.server_url
            model_id = args.model
        else:
            model = ModelConfig(
                name=args.model,
                path=args.model_path,
                engine_kwargs=_parse_engine_kwargs(args.engine_kwargs_json),
            )
            environment = stack.enter_context(
                VllmEnvironment(
                    model,
                    mode=args.vllm_mode,
                    timeout_seconds=args.startup_timeout,
                    extra_args=args.extra_vllm_arg,
                )
            )
            server_url = environment.server_url
            model_id = environment.model_id or args.model

        provider = OpenAIChatCompletionProvider(
            server_url=server_url,
            model=model_id,
            api_key=args.api_key,
            timeout=args.request_timeout,
        )
        candidates = generate_candidates(manifest, provider, run_config.generation)

    write_candidate_records(args.output_dir, candidates)
    selections = replay_selectors(candidates, run_config.selectors)
    write_selection_records(args.output_dir, selections)
    summary = build_run_summary(manifest, run_config, candidates, selections)
    write_run_summary(args.output_dir, summary)

    logger.info(
        "Completed reasoning TTS run for %s with %d prompts and %d candidates",
        manifest.task_name,
        len(manifest.records),
        len(candidates),
    )
    for selector_summary in summary.selector_summaries:
        logger.info(
            "selector=%s accuracy=%s oracle_gap_rate=%s",
            selector_summary.selector_name,
            selector_summary.accuracy,
            selector_summary.oracle_gap_rate,
        )


if __name__ == "__main__":
    main()
