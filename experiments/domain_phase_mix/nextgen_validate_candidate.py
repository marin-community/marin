# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch one full three-phase validation run for a nextgen candidate."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass

import fsspec

from rigging.filesystem import marin_prefix
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import executor_main
from marin.utils import create_cache_tokenizer_step

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.nextgen.utils import loop_root_path
from experiments.domain_phase_mix.three_phase_starcoder_experiment import (
    EVAL_DATASETS_CACHE_PATH,
    EVAL_TASKS,
    TOKENIZER_CACHE_BASE,
    TOKENIZER_NAME,
    create_three_phase_experiment,
)

logger = logging.getLogger(__name__)

CANDIDATES_JSON = "candidates.json"
CANDIDATE_ASSIGNMENTS_JSON = "candidate_assignments.json"


@dataclass(frozen=True)
class CandidateSelection:
    candidate_id: str
    phase_weights: dict[str, dict[str, float]]


def _default_fit_dir(loop_name: str, state_root: str) -> str:
    return os.path.join(loop_root_path(state_root, loop_name), "fit")


def _load_json(path: str, default):
    fs, _, _ = fsspec.get_fs_token_paths(path)
    if not fs.exists(path):
        return default
    with fsspec.open(path, "r") as f:
        return json.load(f)


def load_candidate_for_model(fit_dir: str, model_name: str) -> CandidateSelection:
    assignments = _load_json(os.path.join(fit_dir, CANDIDATE_ASSIGNMENTS_JSON), default={})
    candidate_rows = _load_json(os.path.join(fit_dir, CANDIDATES_JSON), default=[])

    candidate_id = assignments.get(model_name)
    if candidate_id is None:
        raise ValueError(f"Model '{model_name}' has no candidate assignment in {fit_dir}")

    row = next((item for item in candidate_rows if item.get("candidate_id") == candidate_id), None)
    if row is None:
        raise ValueError(f"Candidate '{candidate_id}' not found in {fit_dir}/{CANDIDATES_JSON}")
    if row.get("kind") != "schedule":
        raise ValueError(
            f"Candidate '{candidate_id}' for model '{model_name}' is kind={row.get('kind')!r}, expected 'schedule'"
        )

    phase_weights = row.get("phase_weights")
    if not isinstance(phase_weights, dict) or not phase_weights:
        raise ValueError(f"Candidate '{candidate_id}' missing phase weights")

    return CandidateSelection(candidate_id=candidate_id, phase_weights=phase_weights)


def _run_id_from_candidate(candidate_id: str) -> int:
    try:
        suffix = candidate_id.split("-", maxsplit=1)[-1]
        return int(suffix[:8], 16)
    except ValueError:
        return 0


def _validation_run_name(model_name: str, candidate_id: str) -> str:
    model_slug = model_name.lower().replace(" ", "_").replace("/", "_")
    return f"validate_{model_slug}_{candidate_id[:8]}"


def _validation_experiment_name(model_name: str, candidate_id: str, max_len: int = 64) -> str:
    model_slug = model_name.lower().replace(" ", "_").replace("/", "_")
    candidate_slug = candidate_id[:8]
    prefix = "nextgen_validation_"
    suffix = f"_{candidate_slug}"
    max_model_len = max(max_len - len(prefix) - len(suffix), 1)
    return f"{prefix}{model_slug[:max_model_len]}{suffix}"


def _region_local_marin_path(default_path: str) -> str:
    """Map a Marin GCS path to the current region bucket when possible."""
    current_prefix = marin_prefix().rstrip("/")
    if not default_path.startswith("gs://marin-") or not current_prefix.startswith("gs://marin-"):
        return default_path

    # Keep the object key, swap only the bucket prefix.
    without_scheme = default_path[len("gs://") :]
    _, sep, object_key = without_scheme.partition("/")
    if not sep:
        return default_path
    return f"{current_prefix}/{object_key}"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch one full three-phase validation run for a nextgen candidate.")
    parser.add_argument("--loop-name", required=True, help="Loop name used by nextgen state paths.")
    parser.add_argument(
        "--state-root",
        default="domain_phase_mix/nextgen",
        help="Root directory that stores nextgen loop artifacts.",
    )
    parser.add_argument(
        "--model-name",
        default="DS-RE-CEQ",
        help="Model name from fit_propose candidate assignments.",
    )
    parser.add_argument(
        "--fit-dir",
        default=None,
        help="Optional explicit fit output directory (defaults to derived loop fit path).",
    )
    parser.add_argument(
        "--name-prefix",
        default="pinlin_calvin_xu/data_mixture/three_phase_starcoder_dsre_ceq_validation",
        help="Prefix used for launched validation run names.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping validation execution in CI environment")
        return

    fit_dir = args.fit_dir or _default_fit_dir(args.loop_name, args.state_root)
    selected = load_candidate_for_model(fit_dir, args.model_name)

    tokenizer_cache_base = _region_local_marin_path(TOKENIZER_CACHE_BASE)
    eval_datasets_cache_path = _region_local_marin_path(EVAL_DATASETS_CACHE_PATH)
    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = tokenizer_cache_base
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    experiment = create_three_phase_experiment(
        name=_validation_experiment_name(args.model_name, selected.candidate_id),
        eval_datasets_cache_path=eval_datasets_cache_path,
    )
    run_name = _validation_run_name(args.model_name, selected.candidate_id)
    weight_config = WeightConfig(
        run_id=_run_id_from_candidate(selected.candidate_id),
        phase_weights=selected.phase_weights,
    )

    cache_tokenizer_step = create_cache_tokenizer_step(
        tokenizer_name=TOKENIZER_NAME,
        gcs_path=os.path.join(tokenizer_cache_base, TOKENIZER_NAME.replace("/", "--")),
        name_prefix=args.name_prefix,
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=EVAL_TASKS,
        gcs_path=eval_datasets_cache_path,
        name_prefix=args.name_prefix,
    )
    training_step = experiment.create_training_step(
        weight_config=weight_config,
        name_prefix=args.name_prefix,
        run_name=run_name,
    )

    logger.info(
        "Launching validation for model=%s candidate=%s run_name=%s",
        args.model_name,
        selected.candidate_id,
        run_name,
    )
    executor_main(
        steps=[cache_tokenizer_step, cache_eval_datasets_step, training_step],
        description=f"{args.name_prefix}: validation for {args.model_name}",
    )


if __name__ == "__main__":
    main()
