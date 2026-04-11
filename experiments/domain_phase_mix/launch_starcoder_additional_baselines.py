# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch proportional and Olmix StarCoder baselines on us-central1."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main
from marin.utils import create_cache_tokenizer_step

from experiments.domain_phase_mix import starcoder_additional_baselines as baselines
from experiments.domain_phase_mix.starcoder_additional_baselines import (
    DEFAULT_OLMIX_KL_LAMBDA,
    DEFAULT_REPETITION_FACTOR,
    TopologyName,
    compute_additional_baselines,
)
from experiments.domain_phase_mix.three_phase_starcoder_experiment import (
    EVAL_DATASETS_CACHE_PATH as THREE_PHASE_EVAL_DATASETS_CACHE_PATH,
    EVAL_TASKS as THREE_PHASE_EVAL_TASKS,
    NAME as THREE_PHASE_NAME,
    TOKENIZER_CACHE_BASE as THREE_PHASE_TOKENIZER_CACHE_BASE,
    TOKENIZER_NAME as THREE_PHASE_TOKENIZER_NAME,
    create_three_phase_experiment,
)
from experiments.domain_phase_mix.two_phase_starcoder_experiment import (
    EVAL_DATASETS_CACHE_PATH as TWO_PHASE_EVAL_DATASETS_CACHE_PATH,
    EVAL_TASKS as TWO_PHASE_EVAL_TASKS,
    NAME as TWO_PHASE_NAME,
    TOKENIZER_CACHE_BASE as TWO_PHASE_TOKENIZER_CACHE_BASE,
    TOKENIZER_NAME as TWO_PHASE_TOKENIZER_NAME,
    create_two_phase_experiment,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "all"
CENTRAL1_PREFIX = "gs://marin-us-central1"
DEFAULT_NAME_PREFIX = {
    "two_phase_starcoder": "pinlin_calvin_xu/dm/tp_sc_baselines_olmix1",
    "three_phase_starcoder": "pinlin_calvin_xu/dm/thp_sc_baselines_olmix1",
}


@dataclass(frozen=True)
class LaunchConfig:
    topology: TopologyName
    experiment_name: str
    tokenizer_cache_base: str
    tokenizer_name: str
    eval_datasets_cache_path: str
    eval_tasks: tuple


def _baseline_to_payload(baseline) -> dict[str, object]:
    return {
        "topology": baseline.topology,
        "label": baseline.label,
        "run_id": baseline.run_id,
        "run_name": baseline.run_name,
        "phase_starcoder_weights": list(baseline.phase_starcoder_weights),
        "predicted_objective": baseline.predicted_objective,
    }


def _baseline_from_payload(topology: TopologyName, payload: dict[str, object]):
    if str(payload["topology"]) != topology:
        raise ValueError(f"Mismatched topology in payload: expected {topology}, found {payload['topology']}")
    return baselines.StarcoderBaseline(
        topology=topology,
        label=str(payload["label"]),
        run_id=int(payload["run_id"]),
        run_name=str(payload["run_name"]),
        phase_starcoder_weights=tuple(float(value) for value in payload["phase_starcoder_weights"]),
        predicted_objective=(
            None if payload.get("predicted_objective") is None else float(payload["predicted_objective"])
        ),
    )


def build_baseline_payload(topologies: list[TopologyName]) -> dict[str, list[dict[str, object]]]:
    return {
        topology: [
            _baseline_to_payload(baseline)
            for baseline in compute_additional_baselines(
                topology,
                lambda_kl=DEFAULT_OLMIX_KL_LAMBDA,
                repetition_factor=DEFAULT_REPETITION_FACTOR,
            )
        ]
        for topology in topologies
    }


def encode_baseline_payload(payload: dict[str, list[dict[str, object]]]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_baseline_payload(payload_base64: str) -> dict[TopologyName, list[baselines.StarcoderBaseline]]:
    decoded = base64.urlsafe_b64decode(payload_base64.encode("ascii"))
    raw_payload = json.loads(decoded.decode("utf-8"))
    return {
        topology: [_baseline_from_payload(topology, item) for item in items] for topology, items in raw_payload.items()
    }


def _dataset_launch_config(topology: TopologyName) -> LaunchConfig:
    if topology == "two_phase_starcoder":
        return LaunchConfig(
            topology=topology,
            experiment_name=TWO_PHASE_NAME,
            tokenizer_cache_base=TWO_PHASE_TOKENIZER_CACHE_BASE,
            tokenizer_name=TWO_PHASE_TOKENIZER_NAME,
            eval_datasets_cache_path=TWO_PHASE_EVAL_DATASETS_CACHE_PATH,
            eval_tasks=tuple(TWO_PHASE_EVAL_TASKS),
        )
    return LaunchConfig(
        topology=topology,
        experiment_name=THREE_PHASE_NAME,
        tokenizer_cache_base=THREE_PHASE_TOKENIZER_CACHE_BASE,
        tokenizer_name=THREE_PHASE_TOKENIZER_NAME,
        eval_datasets_cache_path=THREE_PHASE_EVAL_DATASETS_CACHE_PATH,
        eval_tasks=tuple(THREE_PHASE_EVAL_TASKS),
    )


def _create_experiment(topology: TopologyName):
    if topology == "two_phase_starcoder":
        return create_two_phase_experiment(name=TWO_PHASE_NAME)
    return create_three_phase_experiment(name=THREE_PHASE_NAME)


def _safe_name_prefix(name_prefix: str, *, run_names: list[str]) -> str:
    max_run_name_len = max((len(name) for name in run_names), default=0)
    max_prefix_len = max(16, 64 - max_run_name_len - 1)
    if len(name_prefix) <= max_prefix_len:
        return name_prefix
    digest = hashlib.sha1(name_prefix.encode("utf-8")).hexdigest()[:8]
    return name_prefix[: max_prefix_len - 9] + "_" + digest


def prepare_launch(
    topology: TopologyName,
    *,
    name_prefix: str | None = None,
    baselines_override: list[baselines.StarcoderBaseline] | None = None,
) -> tuple[str, list[ExecutorStep], list[dict[str, object]]]:
    launch_config = _dataset_launch_config(topology)
    baseline_specs = baselines_override or compute_additional_baselines(
        topology,
        lambda_kl=DEFAULT_OLMIX_KL_LAMBDA,
        repetition_factor=DEFAULT_REPETITION_FACTOR,
    )
    run_names = [baseline.run_name for baseline in baseline_specs]
    safe_name_prefix = _safe_name_prefix(name_prefix or DEFAULT_NAME_PREFIX[topology], run_names=run_names)

    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = launch_config.tokenizer_cache_base
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    experiment = _create_experiment(topology)
    experiment.resources = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])

    cache_tokenizer_step = create_cache_tokenizer_step(
        tokenizer_name=launch_config.tokenizer_name,
        gcs_path=os.path.join(
            launch_config.tokenizer_cache_base,
            launch_config.tokenizer_name.replace("/", "--"),
        ),
        name_prefix=safe_name_prefix,
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=list(launch_config.eval_tasks),
        gcs_path=launch_config.eval_datasets_cache_path,
        name_prefix=safe_name_prefix,
    )
    weight_configs = [baseline.to_weight_config() for baseline in baseline_specs]
    summary_payload = [_baseline_to_payload(baseline) for baseline in baseline_specs]
    weight_configs_step = experiment.create_weight_configs_step(
        configs=weight_configs,
        summary={"additional_baselines": summary_payload},
        seed=0,
        name_prefix=safe_name_prefix,
    )
    training_steps = [
        experiment.create_training_step(
            weight_config,
            name_prefix=safe_name_prefix,
            run_name=baseline.run_name,
        )
        for baseline, weight_config in zip(baseline_specs, weight_configs, strict=True)
    ]
    steps = [cache_tokenizer_step, cache_eval_datasets_step, weight_configs_step, *training_steps]
    return safe_name_prefix, steps, summary_payload


@contextmanager
def _executor_cli_context():
    original_argv = sys.argv[:]
    sys.argv = [original_argv[0]]
    try:
        yield
    finally:
        sys.argv = original_argv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch proportional and Olmix StarCoder baselines on us-central1.")
    parser.add_argument(
        "--dataset",
        choices=[DEFAULT_DATASET, "two_phase_starcoder", "three_phase_starcoder"],
        default=DEFAULT_DATASET,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--emit-payload-base64", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--payload-base64", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if os.getenv("CI") is not None:
        logger.info("Skipping additional baseline launch in CI environment")
        return

    selected = ["two_phase_starcoder", "three_phase_starcoder"] if args.dataset == DEFAULT_DATASET else [args.dataset]
    if args.emit_payload_base64:
        print(encode_baseline_payload(build_baseline_payload(selected)))
        return

    payload_baselines = decode_baseline_payload(args.payload_base64) if args.payload_base64 is not None else {}
    for topology in selected:
        baselines_override = payload_baselines.get(topology)
        if args.payload_base64 is not None and baselines_override is None:
            raise ValueError(f"Missing payload baselines for topology={topology}")
        safe_name_prefix, steps, summary_payload = prepare_launch(topology, baselines_override=baselines_override)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dataset": topology,
                        "name_prefix": safe_name_prefix,
                        "prefix": CENTRAL1_PREFIX,
                        "baselines": summary_payload,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            continue
        with _executor_cli_context():
            executor_main(
                ExecutorMainConfig(
                    prefix=CENTRAL1_PREFIX,
                    max_concurrent=args.max_concurrent or len(steps),
                ),
                steps=steps,
                description=f"{safe_name_prefix}: proportional + Olmix baselines ({topology})",
            )


if __name__ == "__main__":
    main()
