# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Launch teacher-forced smooth proxies for 300M GSM8K/HumanEval rows.

The hard GSM8K/HumanEval evals are generation-based and noisy at 300M/6B. This
launcher scores deterministic gold continuations with Levanter loglikelihood so
we can estimate smooth loss-style proxies for the same signal/noise population.
"""

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
OUTPUT_DIR = METRIC_REGISTRY_DIR / "300m_generative_smooth_proxy_completion"
STATE_CSV = OUTPUT_DIR / "300m_generative_smooth_proxy_eval_state.csv"
LAUNCH_MANIFEST_CSV = OUTPUT_DIR / "300m_generative_smooth_proxy_eval_launch_manifest.csv"
RESULTS_CSV_LOCAL = OUTPUT_DIR / "300m_generative_smooth_proxy_eval_results.csv"

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_300m_generative_smooth_proxy_evals_20260429"
DEFAULT_REQUEST_CACHE_URI = "gs://marin-us-east5/raw/eval-datasets/300m-generative-smooth-proxy-v1/requests.jsonl"
RESULTS_JSON = "results.json"
RESULTS_CSV = "300m_generative_smooth_proxy_eval_results.csv"
STATE_OUTPUT_CSV = "300m_generative_smooth_proxy_eval_state.csv"

BOOL_STATE_FIELDS = {"has_exact_hf_checkpoint", "is_region_local", "eligible", "has_existing_results"}
INT_STATE_FIELDS = {
    "expected_checkpoint_step",
    "hf_checkpoint_count",
    "hf_checkpoint_latest_step",
}

SMOOTH_PROXY_METRICS = (
    "teacher_forced/gsm8k_5shot_gold_solution/bpb",
    "teacher_forced/gsm8k_5shot_gold_solution/nll",
    "teacher_forced/gsm8k_5shot_answer_hash/bpb",
    "teacher_forced/gsm8k_5shot_answer_hash/nll",
    "teacher_forced/humaneval_10shot_canonical_solution/bpb",
    "teacher_forced/humaneval_10shot_canonical_solution/nll",
)


@dataclass(frozen=True)
class SmoothProxyEvalSpec:
    """One teacher-forced smooth-proxy state row and potential launch unit."""

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
    launch_tpu_type: str
    launch_tpu_region: str
    launch_tpu_zone: str
    eligible: bool
    launch_decision: str
    step_name: str
    result_path: str


@dataclass(frozen=True)
class SmoothProxyScoreConfig:
    """Config for one checkpoint's teacher-forced smooth-proxy scorer."""

    eval_key: str
    checkpoint_root: str
    output_path: str
    request_cache_uri: str
    max_eval_instances: int | None = None
    max_length: int = 4096


@dataclass(frozen=True)
class CollectSmoothProxyResultsConfig:
    """Config for collecting teacher-forced smooth-proxy outputs."""

    output_path: str
    state_rows_json: str
    results_by_eval_key: dict[str, InputName]


def _read_csv(path_or_uri: str | Path) -> pd.DataFrame:
    path_string = str(path_or_uri)
    if path_string.startswith("gs://"):
        with fsspec.open(path_string, "rt") as handle:
            return pd.read_csv(handle, low_memory=False)
    return pd.read_csv(path_or_uri, low_memory=False)


def _existing_result_roots(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, low_memory=False)
    if "checkpoint_root" not in frame.columns:
        return set()
    required = [metric for metric in SMOOTH_PROXY_METRICS if metric in frame.columns]
    if len(required) != len(SMOOTH_PROXY_METRICS):
        return set()
    return {
        _string_value(row.get("checkpoint_root")).rstrip("/")
        for _, row in frame.iterrows()
        if all(pd.notna(row.get(metric)) for metric in SMOOTH_PROXY_METRICS)
    }


def _checkpoint_region(checkpoint_root: str) -> str:
    if not checkpoint_root.startswith("gs://"):
        return ""
    bucket = checkpoint_root.removeprefix("gs://").split("/", maxsplit=1)[0]
    prefix = "marin-"
    if bucket.startswith(prefix):
        return bucket.removeprefix(prefix)
    return ""


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
) -> list[SmoothProxyEvalSpec]:
    """Build state rows for 300M teacher-forced smooth-proxy evals."""
    existing_roots = _existing_result_roots(RESULTS_CSV_LOCAL)
    rows: list[SmoothProxyEvalSpec] = []
    for idx, candidate in enumerate(_candidate_records()):
        exact_hf_checkpoint = _exact_hf_checkpoint(candidate.checkpoint_root, candidate.expected_checkpoint_step)
        has_exact_hf_checkpoint = bool(exact_hf_checkpoint)
        checkpoint_region = _checkpoint_region(candidate.checkpoint_root)
        is_region_local = checkpoint_region in {"", default_tpu_region}
        has_existing_results = candidate.checkpoint_root in existing_roots
        eligible, launch_decision = _launch_decision(
            checkpoint_root=candidate.checkpoint_root,
            has_exact_hf_checkpoint=has_exact_hf_checkpoint,
            is_region_local=is_region_local,
            has_existing_results=has_existing_results,
        )
        suffix = f"_{_slug(eval_key_suffix)}" if eval_key_suffix else ""
        eval_key = f"gensmooth300m_{idx:03d}_{candidate.panel}_{_slug(candidate.run_name)}{suffix}"
        rows.append(
            SmoothProxyEvalSpec(
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
                launch_tpu_type=default_tpu_type,
                launch_tpu_region=default_tpu_region,
                launch_tpu_zone=default_tpu_zone,
                eligible=eligible,
                launch_decision=launch_decision,
                step_name=f"teacher_forced_smooth_proxy/{eval_key}",
                result_path="",
            )
        )
    return rows


def _load_state_rows(path: str | Path) -> list[SmoothProxyEvalSpec]:
    frame = _read_csv(path)
    rows: list[SmoothProxyEvalSpec] = []
    expected_fields = {field.name for field in fields(SmoothProxyEvalSpec)}
    missing = expected_fields - set(frame.columns)
    if missing:
        raise ValueError(f"State CSV {path} is missing columns: {sorted(missing)}")
    for _, row in frame.iterrows():
        kwargs: dict[str, Any] = {}
        for field in fields(SmoothProxyEvalSpec):
            value = row[field.name]
            if field.name in BOOL_STATE_FIELDS:
                kwargs[field.name] = _bool_value(value)
            elif field.name in INT_STATE_FIELDS:
                kwargs[field.name] = int(value)
            else:
                kwargs[field.name] = _string_value(value)
        rows.append(SmoothProxyEvalSpec(**kwargs))
    return rows


def _write_local_outputs(rows: list[SmoothProxyEvalSpec]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records([asdict(row) for row in rows])
    frame.to_csv(STATE_CSV, index=False)
    frame[frame["launch_decision"].eq("launch")].to_csv(LAUNCH_MANIFEST_CSV, index=False)


def _limit_dataset(dataset, max_eval_instances: int | None):
    if max_eval_instances is None:
        return dataset
    return dataset.select(range(min(max_eval_instances, len(dataset))))


def _gsm8k_final_answer(answer: str) -> str:
    marker = "####"
    if marker not in answer:
        raise ValueError(f"GSM8K answer is missing {marker!r}: {answer[:200]}")
    return answer.rsplit(marker, maxsplit=1)[1].strip()


def _gsm8k_fewshot_prefix(train_dataset) -> str:
    examples: list[str] = []
    for example in train_dataset.select(range(5)):
        examples.append(f"Question: {example['question']}\nAnswer: {example['answer']}\n")
    return "\n".join(examples) + "\n"


def _request_rows_from_cache(request_cache_uri: str, max_eval_instances: int | None) -> list[dict[str, str]]:
    metric_counts: dict[str, int] = {}
    rows: list[dict[str, str]] = []
    with fsspec.open(request_cache_uri, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            prefix = row["metric_prefix"]
            count = metric_counts.get(prefix, 0)
            if max_eval_instances is not None and count >= max_eval_instances:
                continue
            metric_counts[prefix] = count + 1
            rows.append(row)
    if not rows:
        raise ValueError(f"No teacher-forced request rows found in {request_cache_uri}")
    return rows


def _smooth_proxy_requests_from_rows(cached_rows: list[dict[str, str]]):
    """Build loglikelihood requests and metric labels from cached request rows."""
    from lm_eval.api.instance import Instance

    request_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(cached_rows):
        target = row["target"]
        request_rows.append(
            {
                "metric_prefix": row["metric_prefix"],
                "instance": Instance("loglikelihood", {}, (row["context"], target), idx),
                "target": target,
            }
        )
    return request_rows


def _smooth_proxy_request_rows_from_hf(max_eval_instances: int | None) -> list[dict[str, str]]:
    """Build serializable request rows for smooth proxy scoring."""
    from datasets import load_dataset

    request_rows: list[dict[str, str]] = []

    gsm8k_train = load_dataset("gsm8k", "main", split="train")
    gsm8k_test = _limit_dataset(load_dataset("gsm8k", "main", split="test"), max_eval_instances)
    gsm8k_prefix = _gsm8k_fewshot_prefix(gsm8k_train)
    for _idx, example in enumerate(gsm8k_test):
        context = f"{gsm8k_prefix}Question: {example['question']}\nAnswer:"
        answer = example["answer"].strip()
        final_answer = _gsm8k_final_answer(answer)
        request_rows.append(
            {
                "metric_prefix": "teacher_forced/gsm8k_5shot_gold_solution",
                "context": context,
                "target": " " + answer,
            }
        )
        request_rows.append(
            {
                "metric_prefix": "teacher_forced/gsm8k_5shot_answer_hash",
                "context": context,
                "target": f" #### {final_answer}",
            }
        )

    try:
        humaneval = load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        humaneval = load_dataset("openai_humaneval", split="test")
    humaneval = _limit_dataset(humaneval, max_eval_instances)
    for _idx, example in enumerate(humaneval):
        target = example["canonical_solution"]
        request_rows.append(
            {
                "metric_prefix": "teacher_forced/humaneval_10shot_canonical_solution",
                "context": example["prompt"],
                "target": target,
            }
        )

    return request_rows


def write_request_cache(output_uri: str, max_eval_instances: int | None) -> None:
    """Write teacher-forced request rows once so child evals do not hit Hugging Face."""
    request_rows = _smooth_proxy_request_rows_from_hf(max_eval_instances)
    fs, _, _ = fsspec.get_fs_token_paths(output_uri)
    fs.makedirs(output_uri.rsplit("/", maxsplit=1)[0], exist_ok=True)
    with fsspec.open(output_uri, "wt") as handle:
        for row in request_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    logger.info("Wrote %d teacher-forced request rows to %s", len(request_rows), output_uri)


def _summarize_loglikelihoods(
    request_rows: list[dict[str, Any]], loglikelihoods: list[tuple[float, bool]]
) -> dict[str, float]:
    by_metric: dict[str, dict[str, float]] = {}
    for row, (loglikelihood, _greedy) in zip(request_rows, loglikelihoods, strict=True):
        prefix = row["metric_prefix"]
        entry = by_metric.setdefault(prefix, {"nll_sum": 0.0, "byte_sum": 0.0, "example_count": 0.0})
        target = row["target"]
        entry["nll_sum"] += -float(loglikelihood)
        entry["byte_sum"] += max(1, len(target.encode("utf-8")))
        entry["example_count"] += 1

    metrics: dict[str, float] = {}
    for prefix, entry in by_metric.items():
        nll_sum = entry["nll_sum"]
        byte_sum = entry["byte_sum"]
        example_count = entry["example_count"]
        metrics[f"{prefix}/nll"] = nll_sum / max(1.0, example_count)
        metrics[f"{prefix}/bpb"] = nll_sum / max(1.0, byte_sum) / math.log(2.0)
        metrics[f"{prefix}/example_count"] = example_count
        metrics[f"{prefix}/target_bytes"] = byte_sum
    return metrics


def score_teacher_forced_smooth_proxies(config: SmoothProxyScoreConfig) -> None:
    """Score one checkpoint on teacher-forced GSM8K/HumanEval smooth proxies."""
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
            request_rows = _smooth_proxy_requests_from_rows(cached_rows)
            loglikelihoods = harness.loglikelihood([row["instance"] for row in request_rows])
            worker.stop()
            metrics = _summarize_loglikelihoods(request_rows, loglikelihoods)
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


def _read_smooth_proxy_metrics(path: InputName | str) -> tuple[dict[str, Any], str]:
    result_path = os.path.join(str(path).rstrip("/"), RESULTS_JSON)
    try:
        with fsspec.open(result_path, "rt") as handle:
            data = json.load(handle)
    except OSError as exc:
        return {}, str(exc)
    return data, ""


def collect_smooth_proxy_results(config: CollectSmoothProxyResultsConfig) -> None:
    """Collect teacher-forced smooth-proxy outputs into one normalized CSV."""
    state_rows = [SmoothProxyEvalSpec(**row) for row in json.loads(config.state_rows_json)]
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
        metrics, error = _read_smooth_proxy_metrics(result_path)
        record.update(metrics)
        record["collection_status"] = "collected" if metrics else "missing_metrics"
        record["collection_error"] = error
        record["result_path"] = str(result_path)
        records.append(record)

    with fsspec.open(os.path.join(output_path, RESULTS_CSV), "wt") as handle:
        pd.DataFrame.from_records(records).to_csv(handle, index=False)
    with fsspec.open(os.path.join(output_path, STATE_OUTPUT_CSV), "wt") as handle:
        pd.DataFrame.from_records([asdict(row) for row in state_rows]).to_csv(handle, index=False)


def build_eval_steps(
    *,
    name_prefix: str,
    state_rows: list[SmoothProxyEvalSpec],
    max_eval_instances: int | None,
    request_cache_uri: str,
) -> tuple[list[ExecutorStep], dict[str, InputName]]:
    """Build teacher-forced smooth-proxy eval steps for rows requiring launch."""
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
            name=f"{name_prefix}/teacher_forced_smooth_proxy/{row.eval_key}",
            description=f"Score teacher-forced GSM8K/HumanEval smooth proxies for {row.eval_key}",
            fn=remote(
                score_teacher_forced_smooth_proxies,
                resources=resource_config,
                pip_dependency_groups=["eval", "tpu"],
            ),
            config=SmoothProxyScoreConfig(
                eval_key=row.eval_key,
                checkpoint_root=row.hf_checkpoint_latest,
                output_path=this_output_path(),
                request_cache_uri=request_cache_uri,
                max_eval_instances=max_eval_instances,
            ),
        )
        eval_steps.append(eval_step)
        results_by_eval_key[row.eval_key] = output_path_of(eval_step)
    return eval_steps, results_by_eval_key


def build_collect_step(
    *,
    name_prefix: str,
    state_rows: list[SmoothProxyEvalSpec],
    results_by_eval_key: dict[str, InputName],
) -> ExecutorStep:
    """Build the final teacher-forced smooth-proxy collection step."""
    return ExecutorStep(
        name=f"{name_prefix}/collect_results",
        description=f"Collect teacher-forced smooth-proxy results for {len(results_by_eval_key)} eval steps",
        fn=collect_smooth_proxy_results,
        config=CollectSmoothProxyResultsConfig(
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
        "Prepared %d teacher-forced smooth-proxy eval steps over %d candidate checkpoints",
        launch_count,
        len(state_rows),
    )
    if args.dry_run or os.getenv("CI") is not None:
        return

    eval_steps, results_by_eval_key = build_eval_steps(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        max_eval_instances=args.max_eval_instances,
        request_cache_uri=args.request_cache_uri,
    )
    collect_step = build_collect_step(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        results_by_eval_key=results_by_eval_key,
    )
    executor_prefix = _executor_prefix(args.executor_prefix, args.tpu_region)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=[*eval_steps, collect_step],
        description=f"{args.name_prefix}: 300M teacher-forced GSM8K/HumanEval smooth proxies",
    )


if __name__ == "__main__":
    main()
