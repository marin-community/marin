#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect an existing Daytona sandbox without mutating its files."""

import argparse

from marin.daytona.client import create_daytona_client
from marin.daytona.config import DaytonaConfig, resolve_daytona_credentials


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a read-only inspection command in a Daytona sandbox.")
    parser.add_argument("sandbox", help="Existing Daytona sandbox id or name.")
    parser.add_argument("--api-key-env", default="DAYTONA_API_KEY")
    parser.add_argument("--endpoint")
    parser.add_argument("--target")
    parser.add_argument("--command", default="find / -maxdepth 2 -type d 2>/dev/null | sort")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    client = create_daytona_client(
        resolve_daytona_credentials(DaytonaConfig(args.endpoint, args.target, args.api_key_env))
    )
    sandbox = client.get(args.sandbox)
    result = sandbox.process.exec(command=args.command, timeout=args.timeout)
    print(getattr(result, "result", ""), end="")
    return int(getattr(result, "exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
