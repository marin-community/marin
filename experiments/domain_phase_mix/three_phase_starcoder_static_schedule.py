# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch one fixed three-phase StarCoder schedule."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys

from rigging.filesystem import marin_prefix
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import executor_main
from marin.utils import create_cache_tokenizer_step

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.three_phase_starcoder_experiment import (
    EVAL_DATASETS_CACHE_PATH,
    EVAL_TASKS,
    TOKENIZER_CACHE_BASE,
    TOKENIZER_NAME,
    create_three_phase_experiment,
)

logger = logging.getLogger(__name__)


def _region_local_marin_path(default_path: str) -> str:
    """Map a Marin GCS path to the current region bucket when possible."""
    current_prefix = marin_prefix().rstrip("/")
    if not default_path.startswith("gs://marin-") or not current_prefix.startswith("gs://marin-"):
        return default_path

    without_scheme = default_path[len("gs://") :]
    _, sep, object_key = without_scheme.partition("/")
    if not sep:
        return default_path
    return f"{current_prefix}/{object_key}"


def _safe_name_prefix(name_prefix: str) -> str:
    """Shorten prefixes that would exceed W&B tag length limits."""
    if len(name_prefix) <= 64:
        return name_prefix
    digest = hashlib.sha1(name_prefix.encode("utf-8")).hexdigest()[:8]
    truncated = f"{name_prefix[:55]}_{digest}"
    logger.warning("Shortening name prefix for W&B tag compatibility: %s -> %s", name_prefix, truncated)
    return truncated


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch one fixed three-phase StarCoder schedule.")
    parser.add_argument(
        "--name-prefix",
        default="pinlin_calvin_xu/data_mixture/three_phase_starcoder_static_schedule",
        help="Prefix used for the launched run name.",
    )
    parser.add_argument(
        "--run-name",
        default="static_schedule",
        help="Run name suffix under the name prefix.",
    )
    parser.add_argument("--phase-0-starcoder", type=float, required=True, help="StarCoder weight for phase 0.")
    parser.add_argument("--phase-1-starcoder", type=float, required=True, help="StarCoder weight for phase 1.")
    parser.add_argument("--phase-2-starcoder", type=float, required=True, help="StarCoder weight for phase 2.")
    parser.add_argument("--run-id", type=int, default=91000, help="Run identifier used for bookkeeping.")
    parser.add_argument("--data-seed", type=int, default=0, help="Data seed for the training run.")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping fixed-schedule execution in CI environment.")
        return

    name_prefix = _safe_name_prefix(args.name_prefix)
    tokenizer_cache_base = _region_local_marin_path(TOKENIZER_CACHE_BASE)
    eval_datasets_cache_path = _region_local_marin_path(EVAL_DATASETS_CACHE_PATH)
    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = tokenizer_cache_base
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    experiment = create_three_phase_experiment(
        name=name_prefix,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )
    phase_weights = {
        "phase_0": {
            "nemotron_full": 1.0 - args.phase_0_starcoder,
            "starcoder": args.phase_0_starcoder,
        },
        "phase_1": {
            "nemotron_full": 1.0 - args.phase_1_starcoder,
            "starcoder": args.phase_1_starcoder,
        },
        "phase_2": {
            "nemotron_full": 1.0 - args.phase_2_starcoder,
            "starcoder": args.phase_2_starcoder,
        },
    }
    weight_config = WeightConfig(run_id=args.run_id, phase_weights=phase_weights)

    cache_tokenizer_step = create_cache_tokenizer_step(
        tokenizer_name=TOKENIZER_NAME,
        gcs_path=os.path.join(tokenizer_cache_base, TOKENIZER_NAME.replace("/", "--")),
        name_prefix=name_prefix,
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=EVAL_TASKS,
        gcs_path=eval_datasets_cache_path,
        name_prefix=name_prefix,
    )
    training_step = experiment.create_training_step(
        weight_config=weight_config,
        name_prefix=name_prefix,
        run_name=args.run_name,
        data_seed=args.data_seed,
    )

    logger.info(
        "Launching fixed schedule %s with weights=%s data_seed=%d",
        args.run_name,
        phase_weights,
        args.data_seed,
    )
    executor_main(
        steps=[cache_tokenizer_step, cache_eval_datasets_step, training_step],
        description=f"{name_prefix}: fixed schedule {args.run_name}",
    )


if __name__ == "__main__":
    main()
