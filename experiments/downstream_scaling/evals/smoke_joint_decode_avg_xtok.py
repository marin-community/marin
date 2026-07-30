# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for cross-tokenizer joint-decode-avg on a small GSM8K slice.

Decoder = the smallest Delphi checkpoint (llama tokenizer); advisor =
Qwen3-0.6B-Base (Qwen tokenizer). Runs both selection rules; the validation
step checks completion counts and non-degenerate text.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, replace

import fsspec
from fray.cluster import ANY_REGION, ResourceConfig, get_tpu_topology
from rigging.filesystem import data_config
from rigging.log_setup import configure_logging
from thalas.execution.executor import ExecutorStep, executor_main, output_path_of
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path, versioned

from experiments.downstream_scaling.evals.algorithms.joint_decode_avg_xtok import (
    TOKEN_PATHS_FILENAME,
    JointDecodeCompletionAlgorithm,
    JointDecodeConfig,
    JointDecodeExecutionConfig,
    JointDecodeModelConfig,
    JointDecodePlacement,
    JointDecodePoolConfig,
    JointDecodeSamplingConfig,
    XtokSelectionRule,
    joint_decode_pool_configs,
)
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.schema import COMPLETIONS_FILENAME, read_completion_rows
from experiments.downstream_scaling.evals.framework.xregion.pool import EnginePlacement, WorkerPoolConfig
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, GSM8KTaskConfig
from experiments.downstream_scaling.evals.utils import version_path
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS
from experiments.models import ModelConfig, download_model_step

logger = logging.getLogger(__name__)

MODEL_KEY = "3e18"
N_PROBLEMS = 64
N_SAMPLES = 4
WORKERS_PER_TPU_TYPE = 2
AGGREGATE_WORKERS = 32
CHUNK_SIZE = 16
TPU_TYPES: tuple[str, ...] = ("v4-8", "v5p-8", "v6e-4", "v5litepod-4", "v6e-8", "v5litepod-8")
PLACEMENT_OVERRIDES: dict[tuple[str, str], tuple[JointDecodePlacement, ...]] = {
    ("1e23", "v6e-8"): (
        JointDecodePlacement(
            decoder=EnginePlacement((0, 1, 2, 3), (2, 2, 1), 2),
            advisor=EnginePlacement((4,), (1, 1, 1), 1),
        ),
    ),
}

BARRIER_TIMEOUT_S = 1200.0
HEARTBEAT_TIMEOUT = 120.0
POLL_BACKOFF = 10.0

MAX_TOKENS = 512
# 2x decoder cap: fertility headroom for the advisor on the same text
# (Qwen tokenizes digits singly; llama 3 groups up to three).
ADVISOR_MAX_TOKENS = 2 * MAX_TOKENS
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")

NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234

TEMPERATURE = 0.4
TOP_K_A = 16
# Deeper than same-tokenizer runs: byte overlap across vocabs is sparse and
# prefix mass improves with depth; cost is only JSON payload size.
TOP_K_B = 64
# Endpoint weights exercise the degenerate corners of both rules; the sweep
# runs all weights through one engine load per child.
ADVISOR_WEIGHTS: tuple[float, ...] = (0.0, 0.5, 1.0)
PREFIX_CREDIT = 1.0
SELECTION_RULES: tuple[XtokSelectionRule, ...] = (
    XtokSelectionRule.BYTES_UNION,
    XtokSelectionRule.ANCHORED_PREFIX_MASS,
)

# Same pin as experiments.models.qwen3_0_6b_base, as an ExecutorStep so the
# regional preseed machinery below applies to it.
QWEN3_0_6B_BASE = download_model_step(ModelConfig(hf_repo_id="Qwen/Qwen3-0.6B-Base", hf_revision="da87bfb"))

DEFAULT_PRESEED_REGIONS: tuple[str, ...] = ("us-central1",)
WORKER_REGIONS_ANY = "any"


@dataclass(frozen=True)
class ValidateCompletionsConfig:
    output_path: str
    completions_path: str
    token_paths_path: str
    expected_rows: int
    expected_advisor_weights: tuple[float, ...]
    expected_samples_per_weight: int


def make_task(n_problems: int) -> GSM8KTask:
    return GSM8KTask(
        config=GSM8KTaskConfig(
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=n_problems,
        )
    )


def make_algorithm(
    *,
    selection_rule: XtokSelectionRule,
    worker_pools: tuple[JointDecodePoolConfig, ...],
    chunk_size: int,
    n_samples: int,
    heartbeat_timeout: float,
    poll_backoff: float,
    advisor_model_path,
) -> JointDecodeCompletionAlgorithm:
    return JointDecodeCompletionAlgorithm(
        config=JointDecodeConfig(
            sampling=JointDecodeSamplingConfig(
                n_samples=n_samples,
                max_tokens=MAX_TOKENS,
                advisor_max_tokens=ADVISOR_MAX_TOKENS,
                top_k_a=TOP_K_A,
                top_k_b=TOP_K_B,
                seed=SEED,
                selection_rule=selection_rule,
                advisor_weights=ADVISOR_WEIGHTS,
                temperature=TEMPERATURE,
                prefix_credit=PREFIX_CREDIT,
                stop=STOP_TOKENS,
            ),
            advisor_model_path=advisor_model_path,
            decoder_model=JointDecodeModelConfig(apply_rpa_block_size_patch=True),
            # The RPA patch is delphi-specific and harms standard models; the
            # Qwen advisor runs unpatched.
            advisor_model=JointDecodeModelConfig(),
            execution=JointDecodeExecutionConfig(
                worker_pools=worker_pools,
                chunk_size=chunk_size,
                microbatch_size=None,
                heartbeat_timeout=heartbeat_timeout,
                poll_backoff=poll_backoff,
                barrier_timeout_s=BARRIER_TIMEOUT_S,
                aggregate_workers=AGGREGATE_WORKERS,
            ),
        )
    )


def _validate_token_path(key: tuple[str, int], path_row: dict, completion: dict) -> None:
    if path_row["advisor_weight"] != completion["metadata"]["advisor_weight"]:
        raise ValueError(f"{key}: token-path advisor_weight {path_row['advisor_weight']} != completion metadata")
    steps = path_row["steps"]
    if completion["text"] and not steps:
        raise ValueError(f"{key}: non-empty completion has no token-path steps")
    joined = bytearray()
    for step in steps:
        for side in ("tokens_a", "tokens_b"):
            if not step[side] or not all(isinstance(token, int) for token in step[side]):
                raise ValueError(f"{key}: token-path step has invalid {side}: {step[side]!r}")
        joined.extend(bytes.fromhex(step["bytes_hex"]))
    # The committed bytes legitimately extend past the text (stop-string
    # truncation; a length cap can cut a forced chunk mid-queue), so the
    # text must be a prefix of the bytes — not the other way around.
    if not bytes(joined).startswith(completion["text"].encode()):
        raise ValueError(f"{key}: completion text is not a prefix of the committed token-path bytes")


def validate_completions(config: ValidateCompletionsConfig) -> None:
    expected_weights = set(config.expected_advisor_weights)
    rows = list(read_completion_rows(config.completions_path))
    if len(rows) != config.expected_rows:
        raise ValueError(f"Expected {config.expected_rows} completion rows, got {len(rows)}")

    with fsspec.open(config.token_paths_path, "rt", compression="gzip") as f:
        path_rows = [json.loads(line) for line in f]
    paths_by_key = {(row["id"], row["completion_index"]): row for row in path_rows}
    if len(paths_by_key) != len(path_rows):
        raise ValueError("Duplicate (id, completion_index) keys in token paths")

    seen_ids: set[str] = set()
    empty: Counter[float] = Counter()
    total: Counter[float] = Counter()
    for row in rows:
        row_id = row["id"]
        if row_id in seen_ids:
            raise ValueError(f"Duplicate completion row id {row_id!r}")
        seen_ids.add(row_id)

        counts = Counter(completion["metadata"]["advisor_weight"] for completion in row["completions"])
        if set(counts) != expected_weights or any(
            count != config.expected_samples_per_weight for count in counts.values()
        ):
            raise ValueError(
                f"Row {row_id!r}: per-weight completion counts {dict(counts)} != "
                f"{config.expected_samples_per_weight} for each of {sorted(expected_weights)}"
            )
        for completion_index, completion in enumerate(row["completions"]):
            key = (row_id, completion_index)
            path_row = paths_by_key.pop(key, None)
            if path_row is None:
                raise ValueError(f"{key}: completion has no token-path row")
            _validate_token_path(key, path_row, completion)

            weight = completion["metadata"]["advisor_weight"]
            total[weight] += 1
            if not completion["text"].strip():
                empty[weight] += 1

    if paths_by_key:
        raise ValueError(f"Token-path rows without matching completions: {sorted(paths_by_key)[:5]}")

    # Occasional empty completions are legitimate (immediate stop/EOS); a
    # majority of them at any one weight means that weight's selection is
    # degenerate — checked per weight so one bad weight can't hide in the
    # sweep-wide average.
    for weight in sorted(expected_weights):
        if empty[weight] * 2 > total[weight]:
            raise ValueError(
                f"advisor_weight={weight}: {empty[weight]}/{total[weight]} completions are empty; "
                "selection looks degenerate"
            )

    path = os.path.join(config.output_path, "validation.SUCCESS")
    with fsspec.open(path, "wt") as f:
        f.write("ok\n")
    logger.info(
        "Validated %d completion rows (%d/%d empty) at %s",
        len(rows),
        sum(empty.values()),
        sum(total.values()),
        config.completions_path,
    )


def make_validation_step(
    *,
    name: str,
    completions_path,
    token_paths_path,
    n_problems: int,
    n_samples: int,
) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=remote(validate_completions, resources=ResourceConfig.with_cpu(cpu=1, ram="1g")),
        config=ValidateCompletionsConfig(
            output_path=this_output_path(),
            completions_path=version_path(completions_path),  # type: ignore[arg-type]
            token_paths_path=version_path(token_paths_path),  # type: ignore[arg-type]
            expected_rows=versioned(n_problems),  # type: ignore[arg-type]
            expected_advisor_weights=versioned(ADVISOR_WEIGHTS),  # type: ignore[arg-type]
            expected_samples_per_weight=versioned(n_samples),  # type: ignore[arg-type]
        ),
    )


def build_run_steps(
    *,
    model_key: str,
    worker_pools: tuple[WorkerPoolConfig, ...],
    chunk_size: int,
    n_problems: int,
    n_samples: int,
    heartbeat_timeout: float,
    poll_backoff: float,
) -> list[ExecutorStep]:
    if model_key not in DELPHI_HF_DOWNLOADS:
        raise ValueError(f"Unknown Delphi model key {model_key!r}; known: {sorted(DELPHI_HF_DOWNLOADS)}")

    pool_slug = "_".join(pool.pool_id for pool in worker_pools)
    model_path = output_path_of(DELPHI_HF_DOWNLOADS[model_key])
    advisor_model_path = output_path_of(QWEN3_0_6B_BASE)
    resolved_pools = joint_decode_pool_configs(model_key, worker_pools, PLACEMENT_OVERRIDES)

    steps = []
    for selection_rule in SELECTION_RULES:
        completions = make_eval_step(
            name=(
                f"downstream_scaling/evals/smoke/joint_decode_avg_xtok/"
                f"{selection_rule.value}/weight_sweep/{model_key}/{pool_slug}"
            ),
            model_path=model_path,
            task=make_task(n_problems),
            alg=make_algorithm(
                selection_rule=selection_rule,
                worker_pools=resolved_pools,
                chunk_size=chunk_size,
                n_samples=n_samples,
                heartbeat_timeout=heartbeat_timeout,
                poll_backoff=poll_backoff,
                advisor_model_path=advisor_model_path,
            ),
            skip_grades=True,
        )
        steps.append(
            make_validation_step(
                name=(
                    f"downstream_scaling/evals/smoke/joint_decode_avg_xtok/"
                    f"{selection_rule.value}/weight_sweep/{model_key}/{pool_slug}/validate"
                ),
                completions_path=output_path_of(completions) / COMPLETIONS_FILENAME,
                token_paths_path=output_path_of(completions) / TOKEN_PATHS_FILENAME,
                n_problems=n_problems,
                n_samples=n_samples,
            )
        )
    return steps


def _regional_download_step(base_step: ExecutorStep, *, name: str, region: str) -> ExecutorStep:
    if base_step.override_output_path is None:
        raise ValueError(f"Download step {base_step.name!r} does not define a stable relative output path")
    bucket = data_config().region_buckets[region]
    regional_step = replace(base_step, name=name)
    return regional_step.with_output_path(f"gs://{bucket.name}/{base_step.override_output_path}")


def build_preseed_steps(
    *,
    model_key: str,
    regions: list[str],
) -> list[ExecutorStep]:
    if model_key not in DELPHI_HF_DOWNLOADS:
        raise ValueError(f"Unknown Delphi model key {model_key!r}; known: {sorted(DELPHI_HF_DOWNLOADS)}")
    _validate_regions(regions, name="preseed regions")

    steps = []
    for region in regions:
        steps.append(
            _regional_download_step(
                DELPHI_HF_DOWNLOADS[model_key],
                name=f"downstream_scaling/evals/smoke/joint_decode_avg_xtok/preseed/{model_key}/{region}",
                region=region,
            )
        )
        steps.append(
            _regional_download_step(
                QWEN3_0_6B_BASE,
                name=f"downstream_scaling/evals/smoke/joint_decode_avg_xtok/preseed/qwen3-0.6b-base/{region}",
                region=region,
            )
        )
    return steps


def _validate_regions(regions: list[str], *, name: str) -> None:
    unknown_regions = sorted(set(regions) - set(data_config().region_buckets))
    if unknown_regions:
        raise ValueError(f"Unknown {name} {unknown_regions}; known: {sorted(data_config().region_buckets)}")


def resolve_worker_regions(worker_regions: list[str] | None, preseed_regions: list[str]) -> list[str]:
    if worker_regions is None:
        _validate_regions(preseed_regions, name="preseed regions")
        return preseed_regions

    if worker_regions == [WORKER_REGIONS_ANY]:
        # Fray's explicit "run anywhere, do not inherit the parent job's
        # region" marker; regions=None (UNSET) would inherit the coordinator
        # job's region for these nested worker pools.
        return [ANY_REGION]

    _validate_regions(worker_regions, name="worker regions")
    return worker_regions


def make_worker_pools(
    *,
    tpu_types: list[str],
    worker_regions: list[str],
    num_workers: int,
) -> tuple[WorkerPoolConfig, ...]:
    # One single-type pool per TPU type: with_tpu sizes the pool's cpu/ram
    # request from its primary type's host, and iris enforces those values as
    # container limits, so mixing host families in one pool either starves
    # the smaller host or caps the larger one. The shared chunk ledger
    # load-balances across pools.
    if num_workers <= 0:
        raise ValueError(f"--num-workers must be positive, got {num_workers}")
    if len(set(tpu_types)) != len(tpu_types):
        raise ValueError(f"duplicate TPU types: {tpu_types}")

    pools = []
    for tpu_type in tpu_types:
        topology = get_tpu_topology(tpu_type)
        if topology.vm_count != 1:
            raise ValueError(f"joint decode xtok supports only single-VM TPU types, got {tpu_type}")
        if topology.chips_per_vm % 2 != 0:
            raise ValueError(f"joint decode xtok needs even chips_per_vm, got {tpu_type}")
        pools.append(
            WorkerPoolConfig(
                pool_id=tpu_type,
                num_workers=num_workers,
                worker_resources=ResourceConfig.with_tpu(tpu_type, regions=worker_regions),
                vm_count=topology.vm_count,
                chips_per_vm=topology.chips_per_vm,
            )
        )
    return tuple(pools)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--mode", choices=("preseed", "run"), required=True)
    parser.add_argument("--model-key", default=MODEL_KEY)
    parser.add_argument("--preseed-regions", nargs="+", default=list(DEFAULT_PRESEED_REGIONS))
    parser.add_argument(
        "--worker-regions",
        nargs="+",
        default=None,
        help=(
            "TPU worker placement regions. Defaults to --preseed-regions. "
            f"Use '{WORKER_REGIONS_ANY}' for unrestricted placement."
        ),
    )
    parser.add_argument("--tpu-types", nargs="+", default=list(TPU_TYPES))
    parser.add_argument(
        "--num-workers",
        type=int,
        default=WORKERS_PER_TPU_TYPE,
        help="Workers per TPU type.",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--n-problems", type=int, default=N_PROBLEMS)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--heartbeat-timeout", type=float, default=HEARTBEAT_TIMEOUT)
    parser.add_argument("--poll-backoff", type=float, default=POLL_BACKOFF)

    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "preseed":
        steps = build_preseed_steps(
            model_key=args.model_key,
            regions=args.preseed_regions,
        )
        description = f"Joint-decode-avg-xtok smoke preseed for {args.model_key} + Qwen3-0.6B-Base."
    else:
        worker_regions = resolve_worker_regions(args.worker_regions, args.preseed_regions)
        worker_pools = make_worker_pools(
            tpu_types=args.tpu_types,
            worker_regions=worker_regions,
            num_workers=args.num_workers,
        )
        steps = build_run_steps(
            model_key=args.model_key,
            worker_pools=worker_pools,
            chunk_size=args.chunk_size,
            n_problems=args.n_problems,
            n_samples=args.n_samples,
            heartbeat_timeout=args.heartbeat_timeout,
            poll_backoff=args.poll_backoff,
        )
        description = "Joint-decode-avg-xtok smoke on a small GSM8K slice (both selection rules)."

    executor_main(
        steps=steps,
        description=description,
    )


if __name__ == "__main__":
    configure_logging()
    main()
