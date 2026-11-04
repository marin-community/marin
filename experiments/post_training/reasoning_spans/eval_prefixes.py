# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score one served model on the reasoning-span prefix tasks with a single in-process vLLM server.

Serves the model from its catalog config, samples replies to every hazard and control prefix the way pi
requested them (pi's tools advertised, thinking enabled, special tokens kept), and grades each reply with
verifyit's ``reasoning_markup`` grader. The grader counts a span left open by the token limit as
``unclosed``; ``length_exempt_violation`` records the verdict with that truncation exempt. Writes
``samples.jsonl`` and ``summary.json`` under ``--output-prefix``. Run it as one H100x8 Iris task:

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \\
      --enable-extra-resources --target-cluster cw-rno2a --priority interactive \\
      --gpu H100x8 --cpu 64 --memory 512GB --disk 600GB --timeout 14400 \\
      --job-name reasoning-span-eval-<model> \\
      -- python -m experiments.post_training.reasoning_spans.eval_prefixes \\
      --model-config experiments/evaluation/serve/models/open-athena/Marin-Snowball-67B-A2B-v0.1.yaml \\
      --output-prefix s3://marin-us-east-02a/marin/rl-data/evals/reasoning-span-prefixes/<model>/<version>
"""

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import click
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from iris.rpc.proto_display import priority_band_value
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.config import RemoteInferenceConfig, ServedModelConfig
from marin.inference.model_preparation import resolve_model_path
from marin.inference.serve import local_inference
from openai import OpenAI
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.pi_trajectory import CONTROL_SPLIT, HAZARD_SPLIT
from verifyit.modes.grade_reasoning_markup import grade_reasoning_markup_candidate
from verifyit.reasoning_markup import reasoning_span_violation
from verifyit.spec import ReasoningMarkupSpec

from experiments.post_training.task_curation.datasets.reasoning_spans import (
    CONTROL_REPOSITORY,
    CONTROL_REVISION,
    HAZARD_REPOSITORY,
    HAZARD_REVISION,
    ROWS_FILE,
)

logger = logging.getLogger(__name__)

SPLITS = (
    (HAZARD_SPLIT, HAZARD_REPOSITORY, HAZARD_REVISION),
    (CONTROL_SPLIT, CONTROL_REPOSITORY, CONTROL_REVISION),
)
# The quick RL loop's per-turn generation budget.
MAX_NEW_TOKENS = 8_192
SAMPLES_PER_PROMPT = 4
GPU_COUNT = 8
# Enough requests in flight to fill every engine's sequence slots.
REQUEST_THREADS = 256
REQUEST_TIMEOUT = 3600
# pi's request settings for a chat-template thinking model.
REQUEST_EXTRA_BODY = {"skip_special_tokens": False, "chat_template_kwargs": {"enable_thinking": True}}
SAMPLES_FILENAME = "samples.jsonl"
SUMMARY_FILENAME = "summary.json"


@dataclass(frozen=True)
class GradedSample:
    digest: str
    split: str
    benchmark: str
    source_model: str
    source_violation: str | None
    sample: int
    finish_reason: str
    content: str
    tool_calls: int
    score: float
    violation: str | None
    length_exempt_violation: str | None


def prefix_rows(repository: str, revision: str) -> list[dict[str, Any]]:
    path = hf_hub_download(repository, ROWS_FILE, repo_type="dataset", revision=revision)
    return pq.read_table(path).to_pylist()


def chat_request(row: dict[str, Any], model: str, samples: int, max_tokens: int) -> dict[str, Any]:
    """The chat completion request pi would send for the row's prompt."""
    return {
        "model": model,
        "messages": json.loads(row["messages"]),
        "tools": [{"type": "function", "function": tool} for tool in json.loads(row["tools"])],
        "n": samples,
        "max_tokens": max_tokens,
        "extra_body": REQUEST_EXTRA_BODY,
    }


def graded_samples(row: dict[str, Any], choices: Sequence[dict[str, Any]]) -> list[GradedSample]:
    """Grade each sampled reply's text, which holds its reasoning markers."""
    graded = []
    for index, choice in enumerate(choices):
        message = choice["message"]
        content = message.get("content") or ""
        verdict = grade_reasoning_markup_candidate(ReasoningMarkupSpec(), content)
        length_exempt = reasoning_span_violation(content, choice["finish_reason"])
        graded.append(
            GradedSample(
                digest=row["digest"],
                split=row["split"],
                benchmark=row["benchmark"],
                source_model=row["model"],
                source_violation=row["violation"],
                sample=index,
                finish_reason=choice["finish_reason"],
                content=content,
                tool_calls=len(message.get("tool_calls") or ()),
                score=verdict.reward,
                violation=verdict.detail["violation"],
                length_exempt_violation=None if length_exempt is None else length_exempt.value,
            )
        )
    return graded


def _group_summary(samples: Sequence[GradedSample]) -> dict[str, Any]:
    return {
        "samples": len(samples),
        "prompts": len({sample.digest for sample in samples}),
        "avg_score": sum(sample.score for sample in samples) / len(samples),
        "length_exempt_avg_score": sum(sample.length_exempt_violation is None for sample in samples) / len(samples),
        "violations": dict(Counter(sample.violation or "well_formed" for sample in samples)),
        "finish_reasons": dict(Counter(sample.finish_reason for sample in samples)),
    }


def _grouped(samples: Iterable[GradedSample], key: Callable[[GradedSample], str]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[GradedSample]] = defaultdict(list)
    for sample in samples:
        groups[key(sample)].append(sample)
    return {name: _group_summary(group) for name, group in sorted(groups.items())}


def evaluation_summary(samples: Sequence[GradedSample]) -> dict[str, Any]:
    """Scores by split, and within each split by the source reply's violation, source model and benchmark."""
    return {
        "all": _group_summary(samples),
        "split": _grouped(samples, lambda sample: sample.split),
        "split_source_violation": _grouped(
            samples, lambda sample: f"{sample.split}/{sample.source_violation or 'well_formed'}"
        ),
        "split_source_model": _grouped(samples, lambda sample: f"{sample.split}/{sample.source_model}"),
        "split_benchmark": _grouped(samples, lambda sample: f"{sample.split}/{sample.benchmark}"),
    }


def regional_weights(inference: RemoteInferenceConfig) -> ServedModelConfig:
    """Serve from the region-local weight mirror, as Iris inference workers do.

    vLLM streams object-store weights with the runai loader, which the catalog's
    ``--model-loader-extra-config`` targets; the tokenizer stays on the pinned repository.
    """
    model = inference.model
    weights = resolve_model_path(model.weights, inference.iris.cache_ttl_days, model.revision)
    return replace(
        model,
        weights=weights,
        api_model=model.model_id,
        tokenizer=model.tokenizer or model.weights,
        revision=model.revision if weights == model.weights else None,
        tokenizer_revision=model.effective_tokenizer_revision,
    )


def sample_rows(
    client: OpenAI, model: str, rows: Sequence[dict[str, Any]], samples: int, max_tokens: int
) -> list[GradedSample]:
    def one(row: dict[str, Any]) -> list[GradedSample]:
        response = client.chat.completions.create(**chat_request(row, model, samples, max_tokens))
        return graded_samples(row, [choice.model_dump() for choice in response.choices])

    graded: list[GradedSample] = []
    with ThreadPoolExecutor(REQUEST_THREADS) as pool:
        for done, row_samples in enumerate(pool.map(one, rows), start=1):
            graded.extend(row_samples)
            if done % 50 == 0:
                logger.info("graded %d/%d prompts", done, len(rows))
    return graded


@click.command(help=__doc__)
@click.option("--model-config", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-prefix", required=True, help="Object-store prefix for samples.jsonl and summary.json.")
@click.option("--samples", default=SAMPLES_PER_PROMPT, show_default=True, help="Replies sampled per prefix.")
@click.option("--max-tokens", default=MAX_NEW_TOKENS, show_default=True)
@click.option("--limit", type=int, help="Score only the first N prefixes of each split, for a smoke run.")
def main(model_config: Path, output_prefix: str, samples: int, max_tokens: int, limit: int | None) -> None:
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    output = StoragePath(output_prefix)
    if (output / SUMMARY_FILENAME).exists():
        raise click.UsageError(f"{output / SUMMARY_FILENAME} exists; choose a new output prefix")
    rows = [row for _, repository, revision in SPLITS for row in prefix_rows(repository, revision)[:limit]]
    logger.info("scoring %d prefixes, %d samples each", len(rows), samples)

    model = load_model_config(model_config)
    inference = inference_config_for_model(
        model,
        AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=GPU_COUNT),
        env_vars={},
        priority=priority_band_value("interactive"),
        api_model=model.location,
    )
    with local_inference(regional_weights(inference), inference.engine, num_chips=GPU_COUNT) as session:
        endpoint = session.model.endpoint
        client = OpenAI(base_url=endpoint.base_url, api_key="unused", timeout=REQUEST_TIMEOUT)
        graded = sample_rows(client, endpoint.model, rows, samples, max_tokens)

    summary = {
        "model_config": model_config.read_text(),
        "datasets": {split: f"{repository}@{revision}" for split, repository, revision in SPLITS},
        "samples_per_prompt": samples,
        "max_tokens": max_tokens,
        "request_extra_body": REQUEST_EXTRA_BODY,
        "scores": evaluation_summary(graded),
    }
    (output / SAMPLES_FILENAME).write_text("".join(json.dumps(asdict(sample)) + "\n" for sample in graded))
    (output / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s: %s", output, json.dumps(summary["scores"]["split"]))


if __name__ == "__main__":
    main()
