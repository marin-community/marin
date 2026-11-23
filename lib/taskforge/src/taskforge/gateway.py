# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The command-line flags naming the GLM gateway, shared by the driver and its launcher; standard library only, so a
launcher imports it without the pipeline."""

import argparse

API_ROOT = "/v1"
"""The path suffix of an OpenAI-compatible base URL. The router serves ``/health`` at the URL without it."""


def base_url(value: str) -> str:
    if not value.endswith(API_ROOT):
        raise argparse.ArgumentTypeError(f"must be a URL ending in {API_ROOT}, got {value!r}")
    return value


def relay_job(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError(f"must name an Iris relay job, got {value!r}")
    return value


def add_gateway_flags(parser: argparse.ArgumentParser) -> None:
    """Exactly one of ``--glm-relay-job`` and ``--glm-base-url``; a blank or malformed value is a usage error, raised
    before anything is submitted or claimed."""
    gateway = parser.add_mutually_exclusive_group(required=True)
    gateway.add_argument("--glm-relay-job", type=relay_job, help="The Iris relay job that resolves the GLM gateway")
    gateway.add_argument("--glm-base-url", type=base_url, help="A direct GLM gateway endpoint ending in /v1")
