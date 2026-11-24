# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe GB200 schedulability: run one CPU-cheap task on a requested GPU shape and exit.

Submitted as a real GB200 job; if it schedules, the node pool can host the shape and
the probe exits after printing nvidia-smi. Used to separate capacity problems from
resource-request problems during pool bring-up.
"""

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True
        ).stdout,
        flush=True,
    )
    print("probe-ok", flush=True)


if __name__ == "__main__":
    main()
