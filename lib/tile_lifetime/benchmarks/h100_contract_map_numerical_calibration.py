# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the CPU calibration for H100 Contract/Map numerical policy."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lib.tile_lifetime.benchmarks.h100_contract_map_backend_runner import _real_algebra_reference
from lib.tile_lifetime.benchmarks.h100_contract_map_backend_training import natural_jax_training_step
from tile_lifetime.bfloat16_metrics import bfloat16_ulp_distance
from tile_lifetime.h100_contract_map_benchmark import (
    REVIEWED_NUMERICAL_FLOORS_SHA256,
    default_h100_contract_map_benchmark_plan,
)

_OUTPUT_ROLES = ("forward", "dx", "dw0", "dw1")
_SEED_KINDS = ("canonical", "heldout_0", "heldout_1", "heldout_2")
_FIXTURE = Path(__file__).parents[1] / "tests/fixtures/h100_contract_map_numerical_calibration.json"


def _heldout_seed(case_id: str, index: int) -> int:
    digest = hashlib.sha256(f"{case_id}:numerical-calibration:{index}".encode()).hexdigest()
    return int(digest[:8], 16)


def _next_power_of_two(value: float) -> float:
    return 2.0 ** math.ceil(math.log2(value))


def calibration_record() -> dict[str, object]:
    """Measure ordinary JAX against uninterrupted FP64 algebra on CPU."""
    if jax.default_backend() != "cpu":
        raise RuntimeError("numerical calibration must run on CPU")
    records: list[dict[str, object]] = []
    for case in default_h100_contract_map_benchmark_plan().cases:
        seeds = (int(case.case_id[-8:], 16), *(_heldout_seed(case.case_id, index) for index in range(3)))
        for seed_kind, seed in zip(_SEED_KINDS, seeds, strict=True):
            rng = np.random.default_rng(seed)
            host_inputs = (
                rng.normal(scale=0.15, size=(case.rows, case.reduction)).astype(np.float32),
                rng.normal(scale=0.15, size=(case.reduction, case.features)).astype(np.float32),
                rng.normal(scale=0.15, size=(case.features, case.reduction)).astype(np.float32),
                rng.normal(scale=0.15, size=(case.rows, case.reduction)).astype(np.float32),
            )
            inputs = tuple(jax.device_put(jnp.asarray(value, dtype=jnp.bfloat16)) for value in host_inputs)
            actual = natural_jax_training_step(case.scalar_map, *inputs)
            jax.block_until_ready(actual)
            reference = _real_algebra_reference(
                case.scalar_map.value, *(np.asarray(value, dtype=np.float32) for value in inputs)
            )
            for role, observed_value, reference_value in zip(_OUTPUT_ROLES, actual, reference, strict=True):
                observed = np.asarray(observed_value)
                absolute = np.abs(observed.astype(np.float32) - np.asarray(reference_value, dtype=np.float32))
                ulp = bfloat16_ulp_distance(observed, reference_value)
                records.append(
                    {
                        "case_id": case.case_id,
                        "maximum_absolute_error": float(absolute.max(initial=0.0)),
                        "maximum_ulp_distance": int(ulp.max(initial=0)),
                        "mean_absolute_error": float(absolute.mean()),
                        "mean_ulp_distance": float(ulp.mean()),
                        "role": role,
                        "seed": seed,
                        "seed_kind": seed_kind,
                        "ulp_over_four_fraction": float(np.count_nonzero(ulp > 4) / ulp.size),
                        "ulp_p95_distance": int(np.percentile(ulp, 95, method="higher")),
                        "ulp_p99_distance": int(np.percentile(ulp, 99, method="higher")),
                    }
                )
    calibrated_floors = {}
    for role in _OUTPUT_ROLES:
        selected = [record for record in records if record["role"] == role]
        maximum = max(float(record["maximum_absolute_error"]) for record in selected)
        mean = max(float(record["mean_absolute_error"]) for record in selected)
        calibrated_floors[role] = {
            "observed_maximum_absolute_error": maximum,
            "observed_mean_absolute_error": mean,
            "predeclared_maximum_absolute_error": _next_power_of_two(maximum),
            "predeclared_mean_absolute_error": _next_power_of_two(mean),
        }
    return {
        "calibration": "ordinary_jax_bfloat16_vs_real_algebra_fp64",
        "jax_version": jax.__version__,
        "platform": jax.default_backend(),
        "records": records,
        "output_floors": calibrated_floors,
        "reviewed_numerical_floors_sha256": REVIEWED_NUMERICAL_FLOORS_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    serialized = json.dumps(calibration_record(), indent=2, sort_keys=True) + "\n"
    if args.write:
        _FIXTURE.write_text(serialized)
        return
    if not _FIXTURE.is_file() or _FIXTURE.read_text() != serialized:
        raise RuntimeError("numerical calibration fixture is stale; rerun with --write")


if __name__ == "__main__":
    main()
