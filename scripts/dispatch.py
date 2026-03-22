#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the monitoring dispatcher CLI.

Usage:
    uv run scripts/dispatch.py register --name my-run ...
    uv run scripts/dispatch.py list
    uv run scripts/dispatch.py tick --event-kind scheduled_poll
"""

from marin.dispatch.cli import cli

if __name__ == "__main__":
    cli()
