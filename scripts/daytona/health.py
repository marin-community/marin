#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create, exercise, and clean up a short-lived Daytona sandbox."""

import argparse

from daytona import CreateSandboxFromImageParams, Resources
from marin.daytona.client import create_daytona_client
from marin.daytona.config import DaytonaConfig, resolve_daytona_credentials
from marin.daytona.health import run_health_probe


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded Daytona sandbox health probe.")
    parser.add_argument("--api-key-env", default="DAYTONA_API_KEY")
    parser.add_argument("--endpoint")
    parser.add_argument("--target")
    parser.add_argument("--image", required=True)
    parser.add_argument("--command", default="true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    client = create_daytona_client(
        resolve_daytona_credentials(DaytonaConfig(args.endpoint, args.target, args.api_key_env))
    )

    def create():
        return client.create(
            CreateSandboxFromImageParams(
                image=args.image,
                resources=Resources(cpu=1, memory=2048, disk=2048),
                ephemeral=True,
            ),
            timeout=args.timeout,
        )

    def execute(sandbox, command: str) -> tuple[int, str]:
        result = sandbox.process.exec(command=command, timeout=args.timeout)
        return int(getattr(result, "exit_code", 1)), str(getattr(result, "result", ""))

    result = run_health_probe(
        create=create,
        command=args.command,
        execute=execute,
        delete=lambda sandbox: sandbox.delete(),
    )
    print(
        f"create={result.create_seconds:.3f}s exec={result.exec_seconds:.3f}s "
        f"delete={result.delete_seconds:.3f}s exit_code={result.exit_code}"
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
