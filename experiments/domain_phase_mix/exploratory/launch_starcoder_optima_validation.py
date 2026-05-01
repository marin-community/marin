# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas"]
# ///
"""Launch StarCoder validation runs for predicted optima from the selector benchmark."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from rigging.filesystem import marin_prefix

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "harbor" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "fray" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "haliax" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "marin" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "levanter" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "iris" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "zephyr" / "src"))

from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.utils import create_cache_tokenizer_step

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.three_phase_starcoder_experiment import (
    EVAL_DATASETS_CACHE_PATH as THREE_PHASE_EVAL_DATASETS_CACHE_PATH,
    EVAL_TASKS as THREE_PHASE_EVAL_TASKS,
    TOKENIZER_CACHE_BASE as THREE_PHASE_TOKENIZER_CACHE_BASE,
    TOKENIZER_NAME as THREE_PHASE_TOKENIZER_NAME,
    create_three_phase_experiment,
)
from experiments.domain_phase_mix.two_phase_starcoder_experiment import (
    EVAL_DATASETS_CACHE_PATH as TWO_PHASE_EVAL_DATASETS_CACHE_PATH,
    EVAL_TASKS as TWO_PHASE_EVAL_TASKS,
    TOKENIZER_CACHE_BASE as TWO_PHASE_TOKENIZER_CACHE_BASE,
    TOKENIZER_NAME as TWO_PHASE_TOKENIZER_NAME,
    create_two_phase_experiment,
)

logger = logging.getLogger(__name__)

STATIC_POLICIES = (
    "feature_maximin_observed",
    "feature_dpp_observed",
    "feature_bayes_linear_observed",
)
DEFAULT_RUN_ID_BASE = {
    "two_phase_starcoder": 95000,
    "three_phase_starcoder": 96000,
}


@dataclass(frozen=True)
class DatasetLaunchConfig:
    dataset: str
    tokenizer_cache_base: str
    eval_datasets_cache_path: str
    tokenizer_name: str
    eval_tasks: tuple[str, ...]


@dataclass(frozen=True)
class ValidationRunSpec:
    dataset: str
    subset_size: int
    policy: str
    selector_seed: int
    weight_config: WeightConfig
    run_name: str


@dataclass(frozen=True)
class ValidationLaunchPlan:
    dataset: str
    benchmark_output_dir: str
    policy: str
    name_prefix: str
    specs: list[ValidationRunSpec]


def _dataset_launch_config(dataset: str) -> DatasetLaunchConfig:
    if dataset == "two_phase_starcoder":
        return DatasetLaunchConfig(
            dataset=dataset,
            tokenizer_cache_base=TWO_PHASE_TOKENIZER_CACHE_BASE,
            eval_datasets_cache_path=TWO_PHASE_EVAL_DATASETS_CACHE_PATH,
            tokenizer_name=TWO_PHASE_TOKENIZER_NAME,
            eval_tasks=tuple(TWO_PHASE_EVAL_TASKS),
        )
    if dataset == "three_phase_starcoder":
        return DatasetLaunchConfig(
            dataset=dataset,
            tokenizer_cache_base=THREE_PHASE_TOKENIZER_CACHE_BASE,
            eval_datasets_cache_path=THREE_PHASE_EVAL_DATASETS_CACHE_PATH,
            tokenizer_name=THREE_PHASE_TOKENIZER_NAME,
            eval_tasks=tuple(THREE_PHASE_EVAL_TASKS),
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def _create_experiment(dataset: str, *, name_prefix: str):
    if dataset == "two_phase_starcoder":
        return create_two_phase_experiment(name=name_prefix)
    if dataset == "three_phase_starcoder":
        return create_three_phase_experiment(name=name_prefix)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _safe_name_prefix(name_prefix: str, *, run_names: list[str] | None = None) -> str:
    max_run_name_len = max((len(name) for name in run_names or []), default=0)
    max_prefix_len = 64 - (max_run_name_len + 1 if max_run_name_len else 0)
    max_prefix_len = max(max_prefix_len, 16)
    if len(name_prefix) <= max_prefix_len:
        return name_prefix
    digest = hashlib.sha1(name_prefix.encode("utf-8")).hexdigest()[:8]
    head_budget = max_prefix_len - len(digest) - 1
    if head_budget <= 0:
        truncated = digest[:max_prefix_len]
    else:
        truncated = f"{name_prefix[:head_budget]}_{digest}"
    logger.warning("Shortening name prefix for W&B tag compatibility: %s -> %s", name_prefix, truncated)
    return truncated


def _region_local_marin_path(default_path: str) -> str:
    current_prefix = marin_prefix().rstrip("/")
    if not default_path.startswith("gs://marin-") or not current_prefix.startswith("gs://marin-"):
        return default_path

    without_scheme = default_path[len("gs://") :]
    _, sep, object_key = without_scheme.partition("/")
    if not sep:
        return default_path
    return f"{current_prefix}/{object_key}"


def _choose_policy(
    selector_summary: pd.DataFrame,
    *,
    dataset: str,
    policy_override: str | None,
) -> str:
    if policy_override is not None:
        return policy_override

    frame = selector_summary[
        (selector_summary["dataset"] == dataset)
        & (selector_summary["mode"] == "retrospective")
        & (selector_summary["policy"].isin(STATIC_POLICIES))
    ].copy()
    if frame.empty:
        raise ValueError(f"No static selector summary rows found for dataset={dataset}")

    frame = frame.sort_values(
        by=["dsre_median_regret@1", "committee_mean_regret@1", "policy"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    return str(frame.iloc[0]["policy"])


def _load_predicted_optima_records(benchmark_output_dir: Path) -> list[dict]:
    path = benchmark_output_dir / "predicted_optima.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing predicted optimum export: {path}")

    records: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _build_validation_specs(
    benchmark_output_dir: Path,
    *,
    dataset: str,
    policy_override: str | None,
    run_id_base: int,
) -> tuple[str, list[ValidationRunSpec]]:
    selector_summary = pd.read_csv(benchmark_output_dir / "selector_summary.csv")
    policy = _choose_policy(selector_summary, dataset=dataset, policy_override=policy_override)
    records = _load_predicted_optima_records(benchmark_output_dir)

    filtered = [
        record
        for record in records
        if record["dataset"] == dataset
        and record["mode"] == "retrospective"
        and record["evaluation_model"] == "DS-RE-CEQ"
        and record["policy"] == policy
    ]
    if not filtered:
        raise ValueError(f"No predicted-optimum records found for dataset={dataset} policy={policy}")

    filtered.sort(key=lambda record: (int(record["subset_size"]), int(record["selector_seed"])))
    best_by_subset: dict[int, dict] = {}
    for record in filtered:
        subset_size = int(record["subset_size"])
        best_by_subset.setdefault(subset_size, record)

    specs: list[ValidationRunSpec] = []
    for subset_size in sorted(best_by_subset):
        record = best_by_subset[subset_size]
        source = WeightConfig.from_dict(record["weight_config"])
        weight_config = WeightConfig(
            run_id=run_id_base + subset_size,
            phase_weights=source.phase_weights,
        )
        run_name = f"{policy.replace('_observed', '')}_k{subset_size:03d}_optimum"
        specs.append(
            ValidationRunSpec(
                dataset=dataset,
                subset_size=subset_size,
                policy=policy,
                selector_seed=int(record["selector_seed"]),
                weight_config=weight_config,
                run_name=run_name,
            )
        )
    return policy, specs


def _write_launch_plan(
    output_path: Path,
    *,
    dataset: str,
    benchmark_output_dir: Path,
    policy: str,
    specs: list[ValidationRunSpec],
    name_prefix: str,
) -> None:
    payload = {
        "dataset": dataset,
        "benchmark_output_dir": str(benchmark_output_dir),
        "policy": policy,
        "name_prefix": name_prefix,
        "n_runs": len(specs),
        "runs": [
            {
                "subset_size": spec.subset_size,
                "selector_seed": spec.selector_seed,
                "run_name": spec.run_name,
                "weight_config": spec.weight_config.to_dict(),
            }
            for spec in specs
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _load_launch_plan_from_payload(payload: dict) -> ValidationLaunchPlan:
    specs = [
        ValidationRunSpec(
            dataset=str(payload["dataset"]),
            subset_size=int(run["subset_size"]),
            policy=str(payload["policy"]),
            selector_seed=int(run["selector_seed"]),
            weight_config=WeightConfig.from_dict(run["weight_config"]),
            run_name=str(run["run_name"]),
        )
        for run in payload["runs"]
    ]
    return ValidationLaunchPlan(
        dataset=str(payload["dataset"]),
        benchmark_output_dir=str(payload.get("benchmark_output_dir", "")),
        policy=str(payload["policy"]),
        name_prefix=str(payload.get("name_prefix", "")),
        specs=specs,
    )


def _load_launch_plan_from_base64(encoded_payload: str) -> ValidationLaunchPlan:
    payload = json.loads(base64.b64decode(encoded_payload).decode("utf-8"))
    return _load_launch_plan_from_payload(payload)


def _resolve_plan_dir(benchmark_output_dir: Path | None) -> Path:
    if benchmark_output_dir is None:
        return Path.cwd()

    if benchmark_output_dir.exists():
        return benchmark_output_dir

    logger.warning(
        "Benchmark output dir %s is not available in this runtime; writing launch plan to %s instead",
        benchmark_output_dir,
        Path.cwd(),
    )
    return Path.cwd()


def _run_validation_specs(
    *,
    dataset: str,
    benchmark_output_dir: Path | None,
    name_prefix: str,
    policy: str,
    specs: list[ValidationRunSpec],
    data_seed: int,
    dry_run: bool,
) -> Path:
    if not specs:
        raise ValueError(f"No validation runs selected for dataset={dataset}")

    safe_name_prefix = _safe_name_prefix(name_prefix, run_names=[spec.run_name for spec in specs])
    plan_dir = _resolve_plan_dir(benchmark_output_dir)
    plan_path = plan_dir / f"{dataset}_validation_launch_plan.json"
    _write_launch_plan(
        plan_path,
        dataset=dataset,
        benchmark_output_dir=benchmark_output_dir or plan_dir,
        policy=policy,
        specs=specs,
        name_prefix=safe_name_prefix,
    )
    logger.info("Prepared %d validation runs for %s using %s", len(specs), dataset, policy)
    if dry_run:
        return plan_path

    cfg = _dataset_launch_config(dataset)
    tokenizer_cache_base = _region_local_marin_path(cfg.tokenizer_cache_base)
    eval_datasets_cache_path = _region_local_marin_path(cfg.eval_datasets_cache_path)
    if dataset == "three_phase_starcoder":
        experiment = create_three_phase_experiment(
            name=safe_name_prefix,
            eval_datasets_cache_path=eval_datasets_cache_path,
        )
    else:
        experiment = _create_experiment(dataset, name_prefix=safe_name_prefix)

    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = tokenizer_cache_base
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    cache_tokenizer_step = create_cache_tokenizer_step(
        tokenizer_name=cfg.tokenizer_name,
        gcs_path=os.path.join(tokenizer_cache_base, cfg.tokenizer_name.replace("/", "--")),
        name_prefix=safe_name_prefix,
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=list(cfg.eval_tasks),
        gcs_path=eval_datasets_cache_path,
        name_prefix=safe_name_prefix,
    )
    weight_configs_step = experiment.create_weight_configs_step(
        configs=[spec.weight_config for spec in specs],
        summary={
            "source": "starcoder_generic_selector_benchmark",
            "dataset": dataset,
            "policy": policy,
            "subset_sizes": [spec.subset_size for spec in specs],
            "benchmark_output_dir": str(benchmark_output_dir) if benchmark_output_dir is not None else str(plan_dir),
        },
        seed=data_seed,
        name_prefix=safe_name_prefix,
    )
    training_steps = [
        experiment.create_training_step(
            spec.weight_config,
            name_prefix=safe_name_prefix,
            run_name=spec.run_name,
            data_seed=data_seed,
        )
        for spec in specs
    ]
    with _executor_cli_context():
        executor_main(
            ExecutorMainConfig(max_concurrent=len(training_steps) + 3),
            steps=[cache_tokenizer_step, cache_eval_datasets_step, weight_configs_step, *training_steps],
            description=f"{safe_name_prefix}: predicted optimum validation runs ({policy})",
        )
    return plan_path


@contextmanager
def _executor_cli_context():
    original_argv = sys.argv[:]
    sys.argv = [original_argv[0]]
    try:
        yield
    finally:
        sys.argv = original_argv


def _launch_validation_runs(
    *,
    dataset: str,
    benchmark_output_dir: Path,
    name_prefix: str,
    policy_override: str | None,
    run_id_base: int,
    data_seed: int,
    dry_run: bool,
) -> Path:
    policy, specs = _build_validation_specs(
        benchmark_output_dir,
        dataset=dataset,
        policy_override=policy_override,
        run_id_base=run_id_base,
    )
    return _run_validation_specs(
        dataset=dataset,
        benchmark_output_dir=benchmark_output_dir,
        name_prefix=name_prefix,
        policy=policy,
        specs=specs,
        data_seed=data_seed,
        dry_run=dry_run,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch StarCoder optimum validation runs from benchmark artifacts")
    parser.add_argument("--benchmark-output-dir", type=Path, default=None)
    parser.add_argument(
        "--launch-plan-json-base64",
        type=str,
        default=None,
        help="Base64-encoded launch plan JSON. Overrides benchmark-output-dir plan discovery.",
    )
    parser.add_argument(
        "--dataset",
        choices=("two_phase_starcoder", "three_phase_starcoder"),
        required=True,
    )
    parser.add_argument("--policy", type=str, default=None, help="Override the static policy to validate.")
    parser.add_argument("--run-id-base", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--name-prefix", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.launch_plan_json_base64 is not None:
        plan = _load_launch_plan_from_base64(args.launch_plan_json_base64)
        if plan.dataset != args.dataset:
            raise ValueError(f"Launch plan dataset mismatch: {plan.dataset} != {args.dataset}")
        name_prefix = args.name_prefix or plan.name_prefix
        benchmark_output_dir = Path(plan.benchmark_output_dir).resolve() if plan.benchmark_output_dir else None
        plan_path = _run_validation_specs(
            dataset=args.dataset,
            benchmark_output_dir=benchmark_output_dir,
            name_prefix=name_prefix,
            policy=plan.policy,
            specs=plan.specs,
            data_seed=args.data_seed,
            dry_run=args.dry_run,
        )
    else:
        if args.benchmark_output_dir is None:
            raise ValueError("Either --benchmark-output-dir or --launch-plan-json-base64 is required")
        benchmark_output_dir = args.benchmark_output_dir.resolve()
        run_id_base = args.run_id_base or DEFAULT_RUN_ID_BASE[args.dataset]
        default_name = f"pinlin_calvin_xu/data_mixture/{args.dataset}_selector_validation/{benchmark_output_dir.name}"
        name_prefix = args.name_prefix or default_name
        plan_path = _launch_validation_runs(
            dataset=args.dataset,
            benchmark_output_dir=benchmark_output_dir,
            name_prefix=name_prefix,
            policy_override=args.policy,
            run_id_base=run_id_base,
            data_seed=args.data_seed,
            dry_run=args.dry_run,
        )
    logger.info("Validation launch plan written to %s", plan_path)


if __name__ == "__main__":
    main()
