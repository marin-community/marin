# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute a trusted harness corpus scorer under the script mode boundary.

This is a retained-runtime integration, not a native correctness projection.
The script returns every sample observation and corpus point metric; its zero
reward deliberately does not reinterpret unbounded or differently scaled metrics.
"""

import hashlib
import json
import math
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verifyit.grade import InvalidTask, Reward, Status, invalid_task, run
from verifyit.spec import ScriptSpec, render_spec


@dataclass(frozen=True)
class BatchResult:
    observations: tuple[Mapping[str, Any], ...]
    aggregates: Mapping[str, Any]
    verdict: Reward


def _function_source(value: object, symbol: str) -> Path | None:
    module, name = symbol.split(".")
    if isinstance(value, Mapping):
        if (
            set(value) != {"tag", "value", "source_dir"}
            or value.get("tag") != "function"
            or value.get("value") != symbol
        ):
            return None
        directory = value.get("source_dir")
        return Path(directory) / f"{module}.py" if isinstance(directory, str) else None
    if callable(value) and getattr(value, "__name__", None) == name:
        code = getattr(value, "__code__", None)
        return Path(code.co_filename) if code is not None else None
    return None


def _code_text_function(value: object, symbol: str, digest: str) -> bool:
    path = _function_source(value, symbol)
    return (
        path is not None
        and path.as_posix().endswith(f"/lm_eval/tasks/code_x_glue/code-text/{symbol.split('.')[0]}.py")
        and path.is_file()
        and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    )


def _xlsum_rouge_profile(config: Mapping[str, Any]) -> bool:
    definitions = config.get("metric_list")
    if (
        config.get("output_type") != "generate_until"
        or config.get("doc_to_target") != "{{summary}}"
        or config.get("doc_to_choice") is not None
        or config.get("filter_list") is not None
        or not isinstance(definitions, list)
        or len(definitions) != 1
    ):
        return False
    definition = definitions[0]
    if not isinstance(definition, Mapping) or set(definition) != {"metric", "aggregation", "higher_is_better"}:
        return False
    if definition.get("higher_is_better") is not True:
        return False
    paths = [
        _function_source(definition.get(key), symbol)
        for key, symbol in (("metric", "utils.rougeL"), ("aggregation", "utils.rougeL_agg"))
    ]
    return all(
        path is not None
        and any(
            path.as_posix().endswith(f"/lm_eval/tasks/afrobench/xlsum/prompt_{prompt}/utils.py")
            for prompt in range(1, 4)
        )
        and path.is_file()
        and hashlib.sha256(path.read_bytes()).hexdigest()
        == "160a95e68f4d37927ccc46e1b5277162422e1b50b1ea7b93af0c44bc25269714"
        for path in paths
    )


def validate_aggregation_seed(seed: object) -> int:
    """Validate evaluate's task-owned uint32 aggregation seed."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise InvalidTask("aggregation seed must be a uint32 integer")
    return seed


def _code_text_profile(config: Mapping[str, Any]) -> bool:
    definitions = config.get("metric_list")
    if config.get("output_type") != "generate_until" or not isinstance(definitions, list) or len(definitions) != 1:
        return False
    metric = definitions[0]
    if not isinstance(metric, Mapping) or set(metric) != {"metric", "aggregation", "higher_is_better"}:
        return False
    if metric.get("aggregation") != "mean" or metric.get("higher_is_better") is not True:
        return False
    if config.get("doc_to_choice") is not None:
        return False
    utility_hash = "06dd12019f0eaee28b622302654a35a53551b28fbbf7277b2e0e66370ceaba4a"
    return (
        _code_text_function(
            metric.get("metric"),
            "bleu.smoothed_bleu_4",
            "6c60882bf795ccdf764a75ef157644b7d629027b6fa547027a000d5c50fcbea6",
        )
        and _code_text_function(config.get("doc_to_text"), "utils.doc_to_text", utility_hash)
        and _code_text_function(config.get("doc_to_target"), "utils.doc_to_target", utility_hash)
    )


def corpus_config_profile(config: Mapping[str, Any]) -> str | None:
    """Recognize the narrow retained-runtime profile without importing harness.

    This proves configuration eligibility only. Runtime module identity, metric
    callables, sample validity and the complete batch remain independently checked.
    """
    if config.get("class") is not None or config.get("process_results") is not None:
        return None
    if _xlsum_rouge_profile(config):
        return "xlsum_rouge_corpus"
    if _code_text_profile(config):
        return "code_text_smoothed_bleu"
    output = config.get("output_type")
    allowed = (
        {"bleu", "chrf", "ter"}
        if output == "generate_until"
        else (
            {"word_perplexity", "byte_perplexity", "bits_per_byte"}
            if output == "loglikelihood_rolling"
            else {"perplexity", "acc"} if output == "loglikelihood" else set()
        )
    )
    definitions = config.get("metric_list")
    if not allowed or not isinstance(definitions, list) or not definitions:
        return None
    names = []
    for definition in definitions:
        if not isinstance(definition, Mapping) or set(definition) - {"metric", "aggregation", "higher_is_better"}:
            return None
        name = definition.get("metric")
        if not isinstance(name, str) or name not in allowed:
            return None
        direction = definition.get("higher_is_better")
        if direction is not None and not isinstance(direction, bool):
            return None
        expected = (
            "weighted_perplexity"
            if name in {"word_perplexity", "byte_perplexity"}
            else "mean" if name == "acc" else name
        )
        if definition.get("aggregation") not in (None, expected):
            return None
        names.append(name)
    if len(set(names)) != len(names):
        return None
    return {
        "generate_until": "translation_corpus",
        "loglikelihood_rolling": "rolling_likelihood_corpus",
        "loglikelihood": "likelihood_corpus",
    }[str(output)]


def score_corpus(
    source_root: Path,
    config_path: Path,
    samples: Sequence[Mapping[str, Any]],
    *,
    timeout: float = 600.0,
    stage: str = "complete",
    aggregation_seed: int | None = None,
) -> BatchResult:
    """Run the source checkout's trusted corpus producer with JSON-only samples.

    ``source_root`` and ``config_path`` are task-owned installation metadata,
    never candidate-selected paths. Samples contain only ``doc`` and already
    filtered ``responses``. The source runner checks its narrow metric contract.
    """
    source_root = source_root.resolve()
    config_path = config_path.resolve()
    tasks_root = source_root / "lm_eval" / "tasks"
    runner = source_root / "lm_eval" / "verifyit_runtime.py"
    if not config_path.is_relative_to(tasks_root) or not config_path.is_file() or not runner.is_file():
        return BatchResult((), {}, invalid_task("trusted harness config or corpus runner is missing"))
    if not isinstance(stage, str) or stage not in {"complete", "observations"}:
        return BatchResult((), {}, invalid_task("unknown corpus execution stage"))
    try:
        seed = validate_aggregation_seed(aggregation_seed) if aggregation_seed is not None else None
    except InvalidTask as error:
        return BatchResult((), {}, invalid_task(str(error)))
    try:
        serialized = json.dumps(
            {"config": str(config_path), "samples": list(samples), "stage": stage, "aggregation_seed": seed},
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        return BatchResult((), {}, invalid_task(f"corpus samples must be finite JSON data: {error}"))
    with tempfile.TemporaryDirectory(prefix="verifyit-harness-corpus-") as directory:
        tests = Path(directory)
        payload = tests / "samples.json"
        payload.write_text(serialized, encoding="utf-8")
        (tests / "producer.sh").write_text('exec "$@"\n', encoding="utf-8")
        spec = ScriptSpec(
            "producer.sh",
            args=(sys.executable, str(runner), str(payload)),
            timeout=timeout,
            verdict_file="corpus-verdict.json",
        )
        spec_path = tests / "verifier.toml"
        spec_path.write_text(render_spec(spec), encoding="utf-8")
        verdict = run(spec_path, tests)
    if verdict.status != Status.SCORED:
        return BatchResult((), {}, verdict)
    observations = verdict.detail.get("observations")
    aggregates = verdict.detail.get("aggregates")
    if not isinstance(observations, list) or not all(isinstance(item, dict) for item in observations):
        raise RuntimeError("trusted corpus producer returned malformed observations")
    if len(observations) != len(samples) or not isinstance(aggregates, dict):
        raise RuntimeError("trusted corpus producer omitted samples or aggregate metrics")
    if verdict.reward != 0:
        raise RuntimeError("retained corpus runtime cannot imply a correctness reward")
    if stage == "observations":
        if aggregates or not observations or any(set(item) != set(observations[0]) for item in observations):
            raise RuntimeError("observation stage returned inconsistent metrics or premature aggregates")
        return BatchResult(tuple(observations), {}, verdict)
    if any(set(observation) != set(aggregates) for observation in observations):
        raise RuntimeError("trusted corpus producer returned inconsistent metric coverage")
    if any(
        isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0
        for value in aggregates.values()
    ):
        raise RuntimeError("trusted corpus producer returned invalid corpus point metrics")
    return BatchResult(tuple(observations), aggregates, verdict)
