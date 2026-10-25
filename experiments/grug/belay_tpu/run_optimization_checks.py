# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run independent TPU diagnostics serially and retain every phase's exit status."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_DEFAULT_PHASES = ("source", "qb", "transport", "kda-tile", "saved-moe")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--reference-step1")
    parser.add_argument("--first-port", type=int, required=True)
    parser.add_argument(
        "--phases", nargs="+", choices=(*_DEFAULT_PHASES, "saved-ce", "routing"), default=_DEFAULT_PHASES
    )
    args = parser.parse_args()
    if any(name in args.phases for name in ("source", "qb", "routing")) and not args.reference_step1:
        parser.error("--reference-step1 is required for source and QB full-update comparisons")
    module_prefix = "experiments.grug.belay_tpu."
    reference = ["--reference-step1", args.reference_step1]
    phases = {
        "source": ("belay_ckpt1_tpu_update_check", [*reference, "--source-qb"]),
        "qb": ("belay_ckpt1_tpu_update_check", reference),
        "routing": ("belay_ckpt1_tpu_update_check", [*reference, "--routing-ops"]),
        "transport": ("transport_kernel_check", []),
        "kda-tile": ("kda_tile_check", []),
        "saved-moe": ("saved_moe_check", []),
        "saved-ce": ("saved_ce_check", []),
    }
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    failed = False
    for index, name in enumerate(args.phases):
        module, extra_args = phases[name]
        phase_output = output / name
        phase_output.mkdir(parents=True, exist_ok=True)
        environment = {
            **os.environ,
            "BELAY_RUN_ID": f"{args.run_prefix}-{name}",
            "IRIS_PORT_JAX": str(args.first_port + 2 * index),
            "IRIS_OUTPUT_DIR": str(phase_output),
            "XLA_FLAGS": (
                f"--xla_allow_excess_precision=false --xla_dump_to={phase_output / 'xla'} --xla_dump_hlo_as_text"
            ),
        }
        result = subprocess.run(
            [sys.executable, "-u", "-m", module_prefix + "run_diagnostic_phase", module_prefix + module, *extra_args],
            env=environment,
        )
        failed |= result.returncode != 0
        record = {"stage": "diagnostic_phase_exit", "phase": name, "returncode": result.returncode}
        print(json.dumps(record), flush=True)
        with (output / "diagnostic-phases.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
