# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ray_run CLI entrypoint resource defaults."""

from marin.run.ray_run import build_parser


def test_entrypoint_num_cpus_defaults_to_one():
    """The entrypoint should request at least 1 CPU to avoid head-node scheduling."""
    parser = build_parser()
    args = parser.parse_args(["--", "echo", "hello"])
    assert args.entrypoint_num_cpus == 1


def test_entrypoint_num_cpus_override():
    """An explicit --entrypoint-num-cpus should override the default."""
    parser = build_parser()
    args = parser.parse_args(["--entrypoint-num-cpus", "4", "--", "echo", "hello"])
    assert args.entrypoint_num_cpus == 4
