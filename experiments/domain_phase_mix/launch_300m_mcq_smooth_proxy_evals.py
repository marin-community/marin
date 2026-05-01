# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Launch MCQ smooth-proxy scoring for hard-only 300M English-lite tasks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
)
from marin.execution.remote import remote
import pandas as pd

from experiments.domain_phase_mix.launch_300m_gsm8k_humaneval_evals import (
    DEFAULT_EXPECTED_300M_STEP,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_ZONE,
    _bool_value,
    _candidate_records,
    _exact_hf_checkpoint,
    _executor_prefix,
    _slug,
    _string_value,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
METRIC_REGISTRY_DIR = TWO_PHASE_MANY_DIR / "metric_registry"
OUTPUT_DIR = METRIC_REGISTRY_DIR / "300m_mcq_smooth_proxy_completion"
STATE_CSV = OUTPUT_DIR / "300m_mcq_smooth_proxy_eval_state.csv"
LAUNCH_MANIFEST_CSV = OUTPUT_DIR / "300m_mcq_smooth_proxy_eval_launch_manifest.csv"
RESULTS_CSV_LOCAL = OUTPUT_DIR / "300m_mcq_smooth_proxy_eval_results.csv"

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_300m_mcq_smooth_proxy_evals_20260430"
DEFAULT_REQUEST_CACHE_URI = "gs://marin-us-east5/raw/eval-datasets/300m-mcq-smooth-proxy-v1/requests.jsonl"
RESULTS_JSON = "results.json"
RESULTS_CSV = "300m_mcq_smooth_proxy_eval_results.csv"
STATE_OUTPUT_CSV = "300m_mcq_smooth_proxy_eval_state.csv"
REQUEST_CACHE_MANIFEST = ".mcq_smooth_proxy_requests_manifest.json"
FEWSHOT_SEED = 1234

TASK_SPECS = (
    ("medmcqa_5shot", "medmcqa", 5, False),
    ("sciq_5shot", "sciq", 5, False),
    ("swag_0shot", "swag", 0, False),
    ("truthfulqa_mc1_0shot", "truthfulqa_mc1", 0, False),
    ("truthfulqa_mc2_0shot", "truthfulqa_mc2", 0, True),
)
TASK_ALIASES = tuple(alias for alias, _task_name, _fewshot, _multi_gold in TASK_SPECS)
TASK_NAME_BY_ALIAS = {alias: task_name for alias, task_name, _fewshot, _multi_gold in TASK_SPECS}
FEWSHOT_BY_ALIAS = {alias: fewshot for alias, _task_name, fewshot, _multi_gold in TASK_SPECS}
MULTI_GOLD_BY_ALIAS = {alias: multi_gold for alias, _task_name, _fewshot, multi_gold in TASK_SPECS}

SINGLE_GOLD_METRICS = (
    "choice_prob_norm",
    "choice_logprob_norm",
    "choice_logprob",
    "choice_prob",
    "logprob",
    "bpb",
    "nll",
    "example_count",
)
MULTI_GOLD_METRICS = (
    "choice_prob_norm",
    "choice_logprob_norm",
    "choice_logprob",
    "choice_prob",
    "example_count",
)
MCQ_SMOOTH_METRICS = tuple(
    f"mcq_smooth/{alias}/{metric}"
    for alias, _task_name, _fewshot, multi_gold in TASK_SPECS
    for metric in (MULTI_GOLD_METRICS if multi_gold else SINGLE_GOLD_METRICS)
)

BOOL_STATE_FIELDS = {"has_exact_hf_checkpoint", "is_region_local", "eligible", "has_existing_results"}
INT_STATE_FIELDS = {
    "expected_checkpoint_step",
    "hf_checkpoint_count",
    "hf_checkpoint_latest_step",
}


@dataclass(frozen=True)
class McqSmoothProxyEvalSpec:
    """One MCQ smooth-proxy state row and potential launch unit."""

    eval_key: str
    panel: str
    run_name: str
    registry_key: str
    source_experiment: str
    cohort: str
    checkpoint_root: str
    expected_checkpoint_step: int
    hf_checkpoint_count: int
    hf_checkpoint_latest: str
    hf_checkpoint_latest_step: int
    has_exact_hf_checkpoint: bool
    checkpoint_region: str
    is_region_local: bool
    has_existing_results: bool
    task_aliases: str
    launch_tpu_type: str
    launch_tpu_region: str
    launch_tpu_zone: str
    eligible: bool
    launch_decision: str
    step_name: str
    result_path: str


@dataclass(frozen=True)
class WriteMcqSmoothProxyRequestCacheConfig:
    """Config for writing shared MCQ request rows."""

    output_path: str
    request_cache_uri: str
    max_eval_instances: int | None = None


@dataclass(frozen=True)
class McqSmoothProxyScoreConfig:
    """Config for one checkpoint's MCQ smooth-proxy scorer."""

    eval_key: str
    checkpoint_root: str
    output_path: str
    request_cache_uri: str
    request_cache_dependency: InputName | str | None = None
    max_eval_instances: int | None = None
    max_length: int = 4096


@dataclass(frozen=True)
class CollectMcqSmoothProxyResultsConfig:
    """Config for collecting MCQ smooth-proxy outputs."""

    output_path: str
    state_rows_json: str
    results_by_eval_key: dict[str, InputName]


def _read_csv(path_or_uri: str | Path) -> pd.DataFrame:
    path_string = str(path_or_uri)
    if path_string.startswith("gs://"):
        with fsspec.open(path_string, "rt") as handle:
            return pd.read_csv(handle, low_memory=False)
    return pd.read_csv(path_or_uri, low_memory=False)


def _checkpoint_region(checkpoint_root: str) -> str:
    if not checkpoint_root.startswith("gs://"):
        return ""
    bucket = checkpoint_root.removeprefix("gs://").split("/", maxsplit=1)[0]
    prefix = "marin-"
    if bucket.startswith(prefix):
        return bucket.removeprefix(prefix)
    return ""


def _region_local_checkpoint_root(checkpoint_root: str, region: str) -> str:
    if not checkpoint_root.startswith("gs://marin-"):
        return checkpoint_root
    bucket_and_path = checkpoint_root.removeprefix("gs://")
    _bucket, _, path = bucket_and_path.partition("/")
    return f"gs://marin-{region}/{path}"


def _region_local_hf_checkpoint(checkpoint_root: str, expected_step: int, region: str) -> str:
    local_root = _region_local_checkpoint_root(checkpoint_root, region)
    return _exact_hf_checkpoint(local_root, expected_step)


def _existing_result_roots(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, low_memory=False)
    if "checkpoint_root" not in frame.columns:
        return set()
    if not all(metric in frame.columns for metric in MCQ_SMOOTH_METRICS):
        return set()
    return {
        _string_value(row.get("checkpoint_root")).rstrip("/")
        for _, row in frame.iterrows()
        if all(pd.notna(row.get(metric)) for metric in MCQ_SMOOTH_METRICS)
    }


def _launch_decision(
    *,
    checkpoint_root: str,
    has_exact_hf_checkpoint: bool,
    is_region_local: bool,
    has_existing_results: bool,
) -> tuple[bool, str]:
    if not checkpoint_root:
        return False, "defer_missing_checkpoint"
    if not has_exact_hf_checkpoint:
        return False, "defer_missing_exact_hf_checkpoint"
    if not is_region_local:
        return False, "defer_checkpoint_region_mismatch"
    if has_existing_results:
        return True, "skip_existing"
    return True, "launch"


def build_state_rows(
    *,
    default_tpu_type: str,
    default_tpu_region: str,
    default_tpu_zone: str,
    eval_key_suffix: str,
) -> list[McqSmoothProxyEvalSpec]:
    """Build state rows for 300M MCQ smooth-proxy evals."""
    existing_roots = _existing_result_roots(RESULTS_CSV_LOCAL)
    rows: list[McqSmoothProxyEvalSpec] = []
    for idx, candidate in enumerate(_candidate_records()):
        exact_hf_checkpoint = _exact_hf_checkpoint(candidate.checkpoint_root, candidate.expected_checkpoint_step)
        checkpoint_region = _checkpoint_region(candidate.checkpoint_root)
        if checkpoint_region not in {"", default_tpu_region}:
            region_local_checkpoint = _region_local_hf_checkpoint(
                candidate.checkpoint_root,
                candidate.expected_checkpoint_step,
                default_tpu_region,
            )
            if region_local_checkpoint:
                exact_hf_checkpoint = region_local_checkpoint
                checkpoint_region = default_tpu_region
        has_exact_hf_checkpoint = bool(exact_hf_checkpoint)
        is_region_local = checkpoint_region in {"", default_tpu_region}
        has_existing_results = candidate.checkpoint_root in existing_roots
        eligible, launch_decision = _launch_decision(
            checkpoint_root=candidate.checkpoint_root,
            has_exact_hf_checkpoint=has_exact_hf_checkpoint,
            is_region_local=is_region_local,
            has_existing_results=has_existing_results,
        )
        suffix = f"_{_slug(eval_key_suffix)}" if eval_key_suffix else ""
        eval_key = f"mcqsmooth300m_{idx:03d}_{candidate.panel}_{_slug(candidate.run_name)}{suffix}"
        rows.append(
            McqSmoothProxyEvalSpec(
                eval_key=eval_key,
                panel=candidate.panel,
                run_name=candidate.run_name,
                registry_key=candidate.registry_key,
                source_experiment=candidate.source_experiment,
                cohort=candidate.cohort,
                checkpoint_root=candidate.checkpoint_root,
                expected_checkpoint_step=DEFAULT_EXPECTED_300M_STEP,
                hf_checkpoint_count=1 if has_exact_hf_checkpoint else 0,
                hf_checkpoint_latest=exact_hf_checkpoint,
                hf_checkpoint_latest_step=DEFAULT_EXPECTED_300M_STEP if has_exact_hf_checkpoint else -1,
                has_exact_hf_checkpoint=has_exact_hf_checkpoint,
                checkpoint_region=checkpoint_region,
                is_region_local=is_region_local,
                has_existing_results=has_existing_results,
                task_aliases=";".join(TASK_ALIASES),
                launch_tpu_type=default_tpu_type,
                launch_tpu_region=default_tpu_region,
                launch_tpu_zone=default_tpu_zone,
                eligible=eligible,
                launch_decision=launch_decision,
                step_name=f"mcq_smooth_proxy/{eval_key}",
                result_path="",
            )
        )
    return rows


def _load_state_rows(path: str | Path) -> list[McqSmoothProxyEvalSpec]:
    frame = _read_csv(path)
    missing = {field.name for field in fields(McqSmoothProxyEvalSpec)} - set(frame.columns)
    if missing:
        raise ValueError(f"State CSV {path} is missing columns: {sorted(missing)}")
    rows: list[McqSmoothProxyEvalSpec] = []
    for _, row in frame.iterrows():
        kwargs: dict[str, Any] = {}
        for field in fields(McqSmoothProxyEvalSpec):
            value = row[field.name]
            if field.name in BOOL_STATE_FIELDS:
                kwargs[field.name] = _bool_value(value)
            elif field.name in INT_STATE_FIELDS:
                kwargs[field.name] = int(value)
            else:
                kwargs[field.name] = _string_value(value)
        rows.append(McqSmoothProxyEvalSpec(**kwargs))
    return rows


def _write_local_outputs(rows: list[McqSmoothProxyEvalSpec]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records([asdict(row) for row in rows])
    frame.to_csv(STATE_CSV, index=False)
    frame[frame["launch_decision"].eq("launch")].to_csv(LAUNCH_MANIFEST_CSV, index=False)


def _doc_stream(task: Any) -> list[dict[str, Any]]:
    docs = task.validation_docs()
    if docs is None:
        docs = task.test_docs()
    if docs is None:
        raise ValueError(f"Task {task.config.task} has neither validation nor test docs")
    return list(docs)


def _gold_indices_from_doc(task_alias: str, task: Any, doc: dict[str, Any], choices: list[str]) -> list[int]:
    if task_alias == "truthfulqa_mc2_0shot":
        labels = doc.get("mc2_targets", {}).get("labels")
        if labels is None:
            raise ValueError(f"TruthfulQA MC2 doc is missing mc2_targets.labels: {doc}")
        gold_indices = [idx for idx, label in enumerate(labels) if int(label) == 1]
        if not gold_indices:
            raise ValueError(f"TruthfulQA MC2 doc has no true choices: {doc}")
        return gold_indices

    target = task.doc_to_target(doc)
    if isinstance(target, list):
        raw_targets = target
    else:
        raw_targets = [target]
    gold_indices: list[int] = []
    for raw_target in raw_targets:
        if isinstance(raw_target, int):
            gold_indices.append(raw_target)
        elif isinstance(raw_target, str):
            stripped = raw_target.strip()
            if raw_target in choices:
                gold_indices.append(choices.index(raw_target))
            elif stripped in choices:
                gold_indices.append(choices.index(stripped))
            else:
                raise ValueError(f"Could not map target {raw_target!r} to choices {choices!r}")
        else:
            raise TypeError(f"Unsupported target type {type(raw_target)} for {task_alias}")
    return gold_indices


def _request_rows_from_lm_eval(max_eval_instances: int | None) -> list[dict[str, object]]:
    """Build serializable per-choice request rows using lm-eval task formatting."""
    from lm_eval.tasks import TaskManager, get_task_dict

    task_manager = TaskManager()
    rows: list[dict[str, object]] = []
    for task_alias, task_name, num_fewshot, multi_gold in TASK_SPECS:
        task = get_task_dict([task_name], task_manager=task_manager)[task_name]
        task.set_config("num_fewshot", num_fewshot)
        docs = _doc_stream(task)
        if max_eval_instances is not None:
            docs = docs[:max_eval_instances]
        for doc_id, doc in enumerate(docs):
            if getattr(task, "fewshot_rnd", None) is not None:
                task.fewshot_rnd.seed(FEWSHOT_SEED)
            context = task.fewshot_context(doc, num_fewshot=num_fewshot)
            if not isinstance(context, str):
                raise TypeError(f"Expected string context for {task_alias}, got {type(context)}")
            choices = [str(choice) for choice in task.doc_to_choice(doc)]
            gold_indices = _gold_indices_from_doc(task_alias, task, doc, choices)
            requests = task.construct_requests(doc, context)
            if not isinstance(requests, list):
                requests = [requests]
            if len(requests) < len(choices):
                raise ValueError(f"Task {task_alias} emitted {len(requests)} requests for {len(choices)} choices")
            for choice_idx, (choice, instance) in enumerate(zip(choices, requests[: len(choices)], strict=True)):
                request_context, target = instance.arguments
                rows.append(
                    {
                        "task_alias": task_alias,
                        "task_name": task_name,
                        "doc_id": doc_id,
                        "choice_idx": choice_idx,
                        "choice": choice,
                        "context": request_context,
                        "target": target,
                        "choice_bytes": max(1, len(choice.encode("utf-8"))),
                        "target_bytes": max(1, len(str(target).encode("utf-8"))),
                        "is_gold": choice_idx in gold_indices,
                        "gold_indices": ";".join(str(index) for index in gold_indices),
                        "multi_gold": multi_gold,
                    }
                )
    if not rows:
        raise ValueError("No MCQ smooth-proxy request rows were built")
    return rows


def _write_jsonl(path: str, rows: list[dict[str, object]]) -> None:
    fs, _, _ = fsspec.get_fs_token_paths(path)
    fs.makedirs(path.rsplit("/", maxsplit=1)[0], exist_ok=True)
    with fsspec.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_request_cache(output_uri: str, max_eval_instances: int | None) -> None:
    """Write MCQ smooth-proxy request rows to GCS or a local path."""
    rows = _request_rows_from_lm_eval(max_eval_instances)
    _write_jsonl(output_uri, rows)
    logger.info("Wrote %d MCQ smooth-proxy request rows to %s", len(rows), output_uri)


def write_request_cache_step(config: WriteMcqSmoothProxyRequestCacheConfig) -> None:
    """Write shared request cache and a small executor-output manifest."""
    write_request_cache(config.request_cache_uri, config.max_eval_instances)
    output_path = config.output_path.rstrip("/")
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)
    with fsspec.open(os.path.join(output_path, REQUEST_CACHE_MANIFEST), "wt") as handle:
        json.dump(
            {
                "request_cache_uri": config.request_cache_uri,
                "max_eval_instances": config.max_eval_instances,
                "task_aliases": TASK_ALIASES,
            },
            handle,
            indent=2,
            sort_keys=True,
        )


def _request_rows_from_cache(request_cache_uri: str, max_eval_instances: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    doc_counts: dict[str, set[int]] = {}
    with fsspec.open(request_cache_uri, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if max_eval_instances is not None:
                docs = doc_counts.setdefault(row["task_alias"], set())
                if int(row["doc_id"]) not in docs and len(docs) >= max_eval_instances:
                    continue
                docs.add(int(row["doc_id"]))
            rows.append(row)
    if not rows:
        raise ValueError(f"No MCQ request rows found in {request_cache_uri}")
    return rows


def _mcq_requests_from_rows(cached_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from lm_eval.api.instance import Instance

    request_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(cached_rows):
        request_rows.append(
            {
                **row,
                "instance": Instance("loglikelihood", {}, (row["context"], row["target"]), idx),
            }
        )
    return request_rows


def _logsumexp(values: list[float]) -> float:
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def _summarize_mcq_groups(rows: list[dict[str, Any]], loglikelihoods: list[tuple[float, bool]]) -> dict[str, float]:
    """Summarize per-choice loglikelihoods into MCQ smooth metrics."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row, (loglikelihood, _greedy) in zip(rows, loglikelihoods, strict=True):
        item = dict(row)
        item["loglikelihood"] = float(loglikelihood)
        grouped.setdefault((str(row["task_alias"]), int(row["doc_id"])), []).append(item)

    sums: dict[str, dict[str, float]] = {}
    for (task_alias, _doc_id), items in grouped.items():
        items = sorted(items, key=lambda item: int(item["choice_idx"]))
        log_probs = [float(item["loglikelihood"]) for item in items]
        choice_bytes = [float(item["choice_bytes"]) for item in items]
        gold_indices = [idx for idx, item in enumerate(items) if _bool_value(item["is_gold"])]
        if not gold_indices:
            raise ValueError(f"No gold choices found for {task_alias}")
        raw_denominator = _logsumexp(log_probs)
        normalized_scores = [
            log_prob / byte_count / math.log(2.0) for log_prob, byte_count in zip(log_probs, choice_bytes, strict=True)
        ]
        normalized_denominator = _logsumexp(normalized_scores)

        raw_gold_logprob = _logsumexp([log_probs[index] for index in gold_indices]) - raw_denominator
        normalized_gold_logprob = (
            _logsumexp([normalized_scores[index] for index in gold_indices]) - normalized_denominator
        )
        entry = sums.setdefault(task_alias, {"example_count": 0.0})
        entry["choice_logprob"] = entry.get("choice_logprob", 0.0) + raw_gold_logprob
        entry["choice_prob"] = entry.get("choice_prob", 0.0) + math.exp(raw_gold_logprob)
        entry["choice_logprob_norm"] = entry.get("choice_logprob_norm", 0.0) + normalized_gold_logprob
        entry["choice_prob_norm"] = entry.get("choice_prob_norm", 0.0) + math.exp(normalized_gold_logprob)
        entry["example_count"] += 1.0

        if not MULTI_GOLD_BY_ALIAS[task_alias]:
            gold_index = gold_indices[0]
            nll = -log_probs[gold_index]
            entry["nll"] = entry.get("nll", 0.0) + nll
            entry["bpb"] = entry.get("bpb", 0.0) + nll / choice_bytes[gold_index] / math.log(2.0)
            entry["logprob"] = entry.get("logprob", 0.0) + log_probs[gold_index]

    metrics: dict[str, float] = {}
    for task_alias, entry in sums.items():
        example_count = max(1.0, entry["example_count"])
        prefix = f"mcq_smooth/{task_alias}"
        for metric, value in entry.items():
            if metric == "example_count":
                metrics[f"{prefix}/example_count"] = value
            else:
                metrics[f"{prefix}/{metric}"] = value / example_count
    return metrics


def score_mcq_smooth_proxies(config: McqSmoothProxyScoreConfig) -> None:
    """Score one checkpoint on MCQ smooth-proxy requests."""
    import typing

    import jax
    import jmp
    from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
    from levanter.eval_harness import _LmEvalHarnessWorker
    from levanter.models.lm_model import LmHeadModel
    import levanter.tracker
    from levanter.tracker import NoopConfig
    from levanter.trainer import TrainerConfig
    from levanter.utils.tree_utils import inference_mode

    trainer_config = TrainerConfig(
        tracker=NoopConfig(),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        per_device_eval_parallelism=1,
    )
    trainer_config.initialize()
    tokenizer = load_tokenizer(config.checkpoint_root)
    model_config = HFCheckpointConverter.from_hf(config.checkpoint_root).LevConfigClass()

    compute_axis_mapping = trainer_config.compute_axis_mapping
    parameter_axis_mapping = trainer_config.parameter_axis_mapping
    with trainer_config.use_device_mesh():
        converter = model_config.hf_checkpoint_converter()
        converter = converter.replaced(reference_checkpoint=config.checkpoint_root, tokenizer=tokenizer)
        model = converter.load_pretrained(
            model_config.model_type,
            ref=config.checkpoint_root,
            dtype=trainer_config.mp.compute_dtype,
            axis_mapping=parameter_axis_mapping,
        )
        model = typing.cast(LmHeadModel, inference_mode(model, True))

        eval_pos = model.Pos.resize(config.max_length)
        worker = _LmEvalHarnessWorker(
            trainer_config.EvalBatch,
            eval_pos,
            model,
            compute_axis_mapping,
            tokenizer,
            trainer_config.mp,
            max_packed_segments=64,
        )

        if jax.process_index() == 0:
            harness = worker.make_harness_lm()
            cached_rows = _request_rows_from_cache(config.request_cache_uri, config.max_eval_instances)
            request_rows = _mcq_requests_from_rows(cached_rows)
            loglikelihoods = harness.loglikelihood([row["instance"] for row in request_rows])
            worker.stop()
            metrics = _summarize_mcq_groups(request_rows, loglikelihoods)
            metrics.update(
                {
                    "eval_key": config.eval_key,
                    "scored_hf_checkpoint": config.checkpoint_root,
                    "max_eval_instances": config.max_eval_instances if config.max_eval_instances is not None else "",
                }
            )
            output_path = config.output_path.rstrip("/")
            fs, _, _ = fsspec.get_fs_token_paths(output_path)
            fs.makedirs(output_path, exist_ok=True)
            with fsspec.open(os.path.join(output_path, RESULTS_JSON), "wt") as handle:
                json.dump(metrics, handle, indent=2, sort_keys=True)
            levanter.tracker.current_tracker().finish()
        else:
            worker.worker_message_loop()


def _read_mcq_smooth_metrics(path: InputName | str) -> tuple[dict[str, Any], str]:
    result_path = os.path.join(str(path).rstrip("/"), RESULTS_JSON)
    try:
        with fsspec.open(result_path, "rt") as handle:
            data = json.load(handle)
    except OSError as exc:
        return {}, str(exc)
    return data, ""


def collect_mcq_smooth_proxy_results(config: CollectMcqSmoothProxyResultsConfig) -> None:
    """Collect MCQ smooth-proxy outputs into one normalized CSV."""
    state_rows = [McqSmoothProxyEvalSpec(**row) for row in json.loads(config.state_rows_json)]
    output_path = config.output_path.rstrip("/")
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in state_rows:
        record = asdict(row)
        result_path = config.results_by_eval_key.get(row.eval_key)
        if result_path is None:
            record["collection_status"] = "not_launched"
            records.append(record)
            continue
        metrics, error = _read_mcq_smooth_metrics(result_path)
        record.update(metrics)
        record["collection_status"] = "collected" if metrics else "missing_metrics"
        record["collection_error"] = error
        record["result_path"] = str(result_path)
        records.append(record)

    with fsspec.open(os.path.join(output_path, RESULTS_CSV), "wt") as handle:
        pd.DataFrame.from_records(records).to_csv(handle, index=False)
    with fsspec.open(os.path.join(output_path, STATE_OUTPUT_CSV), "wt") as handle:
        pd.DataFrame.from_records([asdict(row) for row in state_rows]).to_csv(handle, index=False)


def build_request_cache_step(
    *,
    name_prefix: str,
    request_cache_uri: str,
    max_eval_instances: int | None,
) -> ExecutorStep:
    """Build the shared request-cache step."""
    return ExecutorStep(
        name=f"{name_prefix}/build_mcq_smooth_proxy_request_cache",
        description="Build 300M MCQ smooth-proxy request cache",
        fn=remote(
            write_request_cache_step,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="64g", regions=[DEFAULT_TPU_REGION]),
            pip_dependency_groups=["eval", "tpu"],
        ),
        config=WriteMcqSmoothProxyRequestCacheConfig(
            output_path=this_output_path(),
            request_cache_uri=request_cache_uri,
            max_eval_instances=max_eval_instances,
        ),
    )


def build_eval_steps(
    *,
    name_prefix: str,
    state_rows: list[McqSmoothProxyEvalSpec],
    max_eval_instances: int | None,
    request_cache_uri: str,
    request_cache_dependency: InputName | str,
) -> tuple[list[ExecutorStep], dict[str, InputName]]:
    """Build MCQ smooth-proxy eval steps for rows requiring launch."""
    eval_steps: list[ExecutorStep] = []
    results_by_eval_key: dict[str, InputName] = {}
    for row in state_rows:
        if row.launch_decision != "launch":
            continue
        resource_config = ResourceConfig.with_tpu(
            row.launch_tpu_type,
            regions=[row.launch_tpu_region],
            zone=row.launch_tpu_zone,
        )
        eval_step = ExecutorStep(
            name=f"{name_prefix}/mcq_smooth_proxy/{row.eval_key}",
            description=f"Score MCQ smooth proxies for {row.eval_key}",
            fn=remote(
                score_mcq_smooth_proxies,
                resources=resource_config,
                pip_dependency_groups=["eval", "tpu"],
            ),
            config=McqSmoothProxyScoreConfig(
                eval_key=row.eval_key,
                checkpoint_root=row.hf_checkpoint_latest,
                output_path=this_output_path(),
                request_cache_uri=request_cache_uri,
                request_cache_dependency=request_cache_dependency,
                max_eval_instances=max_eval_instances,
            ),
        )
        eval_steps.append(eval_step)
        results_by_eval_key[row.eval_key] = output_path_of(eval_step)
    return eval_steps, results_by_eval_key


def build_collect_step(
    *,
    name_prefix: str,
    state_rows: list[McqSmoothProxyEvalSpec],
    results_by_eval_key: dict[str, InputName],
) -> ExecutorStep:
    """Build the final MCQ smooth-proxy collection step."""
    return ExecutorStep(
        name=f"{name_prefix}/collect_results",
        description=f"Collect MCQ smooth-proxy results for {len(results_by_eval_key)} eval steps",
        fn=collect_mcq_smooth_proxy_results,
        config=CollectMcqSmoothProxyResultsConfig(
            output_path=this_output_path(),
            state_rows_json=json.dumps([asdict(row) for row in state_rows], sort_keys=True),
            results_by_eval_key=results_by_eval_key,
        ),
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--max-eval-instances", type=int)
    parser.add_argument("--executor-prefix")
    parser.add_argument("--eval-key-suffix", default="")
    parser.add_argument("--state-csv")
    parser.add_argument("--request-cache-uri", default=DEFAULT_REQUEST_CACHE_URI)
    parser.add_argument("--write-request-cache", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.write_request_cache:
        write_request_cache(args.request_cache_uri, args.max_eval_instances)
        return
    if args.state_csv is None:
        state_rows = build_state_rows(
            default_tpu_type=args.tpu_type,
            default_tpu_region=args.tpu_region,
            default_tpu_zone=args.tpu_zone,
            eval_key_suffix=args.eval_key_suffix,
        )
    else:
        state_rows = _load_state_rows(args.state_csv)

    _write_local_outputs(state_rows)
    launch_count = sum(row.launch_decision == "launch" for row in state_rows)
    logger.info("Wrote state to %s", STATE_CSV)
    logger.info("Wrote launch manifest to %s", LAUNCH_MANIFEST_CSV)
    logger.info(
        "Prepared %d MCQ smooth-proxy eval steps over %d candidate checkpoints and %d tasks",
        launch_count,
        len(state_rows),
        len(TASK_ALIASES),
    )
    if args.dry_run or os.getenv("CI") is not None:
        return

    cache_step = build_request_cache_step(
        name_prefix=args.name_prefix,
        request_cache_uri=args.request_cache_uri,
        max_eval_instances=args.max_eval_instances,
    )
    eval_steps, results_by_eval_key = build_eval_steps(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        max_eval_instances=args.max_eval_instances,
        request_cache_uri=args.request_cache_uri,
        request_cache_dependency=output_path_of(cache_step, REQUEST_CACHE_MANIFEST),
    )
    collect_step = build_collect_step(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        results_by_eval_key=results_by_eval_key,
    )
    executor_prefix = _executor_prefix(args.executor_prefix, args.tpu_region)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=[cache_step, *eval_steps, collect_step],
        description=f"{args.name_prefix}: 300M MCQ smooth proxies for hard-only English-lite tasks",
    )


if __name__ == "__main__":
    main()
