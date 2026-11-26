# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run nightly control cohorts through each model's regular launcher."""

import argparse
import json
import subprocess
import sys

from finestore.rl.rl_progress import completed_controls
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training import rl_compose, rl_gate, rl_index
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.snowball_rl import SNOWBALL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="Unique schedule or dispatch identifier.")
    args = parser.parse_args()
    configure_coreweave_s3()
    records = rl_index.list_attempts()
    started = []
    failed = False
    for table in (SNOWBALL, MOCHI):
        for preset in table.presets:
            if not preset.gated:
                continue
            try:
                version = rl_gate.config_version(rl_gate.compose_preset(preset))
                cohort = completed_controls(records, model=table.model, preset=preset.name, version=version)
                count = max(1, rl_gate.CONTROL_COUNT - len(cohort))
                print(
                    json.dumps({"model": table.model, "preset": preset.name, "version": version, "requested": count}),
                    flush=True,
                )
                started.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            f"experiments.post_training.{table.model}_rl",
                            "--preset",
                            preset.name,
                            "--control",
                            args.batch,
                            "--replicates",
                            str(count),
                            "--version",
                            rl_compose.RUN_ARTIFACT_VERSION,
                            "--run",
                            "--wait",
                        ]
                    )
                )
            except (ValueError, OSError, RuntimeError) as error:
                print(f"{table.model}/{preset.name}: {error}", file=sys.stderr)
                failed = True
    for child in started:
        failed |= child.wait() != 0
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
