# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pytest hooks for the replay golden tests."""


def pytest_addoption(parser) -> None:
    """Register ``--update-goldens`` to regenerate the committed golden files."""
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Regenerate replay golden files instead of asserting against them.",
    )
