# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The one generic in-region CPU submit path for loop-guards analysis subcommands.

Every ad-hoc ``submit-*.py`` wrapper reassembled the same ``iris job run`` envelope
around an inline Python string. This subcommand keeps one checked-in envelope and
takes the program as ordinary arguments:

    uv run --no-sync python -m experiments.post_training.mochi_loop_guards cpu-submit \\
        --job-name mochi-loop-guards-postprocess --cpu 16 --memory 64GB --disk 128GB \\
        --timeout 14400 --env MARIN_PREFIX=s3://marin-us-east-08a/marin \\
        -- python -m experiments.post_training.mochi_loop_guards postprocess …

``--skyrl`` runs the arguments inside SkyRL's frozen CPU environment instead of the
Marin workspace environment; the vendored-SkyRL analyses (``counterfactual``,
``observation-census``, ``stitch-tb2-recovery``, ``tb2-repetition-ledger``) need it,
exactly as their original wrappers did.
"""

import argparse
import subprocess
from collections.abc import Sequence

from marin.rl.skyrl import skyrl_cpu_command

CLUSTER = "cw-us-east-08a"
PRIORITY = "production"
TIMEOUT_SECONDS = 3600
PACKAGE = "marin-core"


def submit_command(
    job_name: str,
    cpu: int,
    memory: str,
    disk: str,
    arguments: Sequence[str],
    *,
    timeout_seconds: int = TIMEOUT_SECONDS,
    cluster: str = CLUSTER,
    priority: str = PRIORITY,
    env: Sequence[str] = (),
    secrets: Sequence[str] = (),
    skyrl: bool = False,
) -> list[str]:
    """Assemble the ``iris job run`` argv for one in-region CPU job."""
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        raise ValueError("cpu-submit needs arguments after `--` to run in region")
    command = [
        "uv",
        "run",
        "--no-sync",
        "iris",
        "--cluster",
        cluster,
        "job",
        "run",
        "--no-wait",
        "--priority",
        priority,
        "--cpu",
        str(cpu),
        "--memory",
        memory,
        "--disk",
        disk,
        "--enable-extra-resources",
        "--timeout",
        str(timeout_seconds),
        "--sync-package",
        PACKAGE,
        "--job-name",
        job_name,
    ]
    for setting in env:
        name, separator, value = setting.partition("=")
        if not separator or not name or not value:
            raise ValueError(f"--env expects KEY=VALUE, got {setting!r}")
        command.extend(("-e", name, value))
    for secret in secrets:
        command.extend(("--secret", secret, f"env:{secret}"))
    command.append("--")
    if skyrl:
        command.extend(skyrl_cpu_command(*arguments))
    else:
        command.extend(("uv", "run", "--package", PACKAGE, *arguments))
    return command


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-name", required=True, help="In-region job name")
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--memory", required=True, help="Iris allocation, e.g. 64GB")
    parser.add_argument("--disk", required=True, help="Iris allocation, e.g. 128GB")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS, help="Job timeout in seconds")
    parser.add_argument("--cluster", default=CLUSTER)
    parser.add_argument("--priority", default=PRIORITY)
    parser.add_argument(
        "--env", action="append", default=[], metavar="KEY=VALUE", help="Forward an environment variable"
    )
    parser.add_argument("--secret", action="append", default=[], metavar="NAME", help="Deliver a named secret file")
    parser.add_argument("--skyrl", action="store_true", help="Run inside SkyRL's frozen CPU environment")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, metavar="-- ARGUMENTS")
    args = parser.parse_args(argv)
    command = submit_command(
        args.job_name,
        args.cpu,
        args.memory,
        args.disk,
        args.arguments,
        timeout_seconds=args.timeout,
        cluster=args.cluster,
        priority=args.priority,
        env=args.env,
        secrets=args.secrets,
        skyrl=args.skyrl,
    )
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
