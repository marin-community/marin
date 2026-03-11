# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the minimal Pallas CE reproducer on a TPU slice via Fray-on-Ray."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import ray
from fray.v1.cluster.ray.tpu import run_on_pod

_REPRO_PATH = Path(__file__).resolve().parents[3] / "lib" / "levanter" / "scripts" / "bench" / "repro_pallas_ce_vmem.py"
_SPEC = importlib.util.spec_from_file_location("repro_pallas_ce_vmem", _REPRO_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load repro module from {_REPRO_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
run_repro = _MODULE.run_repro


@dataclass(frozen=True)
class ReproConfig:
    tpu_type: str
    batch: int
    hidden: int
    vocab: int
    x_dtype: str
    w_dtype: str
    compute_dtype: str
    implementation: str
    v_block_divisor: int
    backward: bool
    seed: int


def _parse_args() -> ReproConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tpu-type", choices=("v5p-8", "v5p-64"), required=True)
    parser.add_argument("--batch", type=int, default=40_960)
    parser.add_argument("--hidden", type=int, default=2_048)
    parser.add_argument("--vocab", type=int, default=128_256)
    parser.add_argument("--x-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--w-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--compute-dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--implementation", default="pallas_tpu")
    parser.add_argument("--v-block-divisor", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return ReproConfig(
        tpu_type=args.tpu_type,
        batch=args.batch,
        hidden=args.hidden,
        vocab=args.vocab,
        x_dtype=args.x_dtype,
        w_dtype=args.w_dtype,
        compute_dtype=args.compute_dtype,
        implementation=args.implementation,
        v_block_divisor=args.v_block_divisor,
        backward=args.backward,
        seed=args.seed,
    )


def _submit_repro(config: ReproConfig) -> float:
    @ray.remote(max_calls=1)
    def remote_repro() -> float:
        return run_repro(
            batch=config.batch,
            hidden=config.hidden,
            vocab=config.vocab,
            x_dtype_name=config.x_dtype,
            w_dtype_name=config.w_dtype,
            compute_dtype_name=config.compute_dtype,
            implementation=config.implementation,
            v_block_divisor=config.v_block_divisor,
            backward=config.backward,
            seed=config.seed,
        )

    results = run_on_pod(
        remote_repro,
        config.tpu_type,
        num_slices=1,
        max_retries_preemption=0,
        max_retries_failure=0,
    )
    if isinstance(results, list):
        return float(results[0])
    return float(results)


def main() -> None:
    config = _parse_args()
    value = _submit_repro(config)
    print("repro_loss", value)


if __name__ == "__main__":
    main()
