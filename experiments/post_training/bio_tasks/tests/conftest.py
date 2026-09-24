# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def pytest_addoption(parser):
    parser.addoption("--bio-source-cache", type=Path, help="Offline SHA-256 file cache for full-study integration tests")
