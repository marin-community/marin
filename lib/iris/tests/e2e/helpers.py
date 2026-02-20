# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared job helper functions used across e2e test files."""

import time


def _quick():
    return 1


def _slow():
    time.sleep(120)


def _block(s):
    """Block until sentinel is signalled. Pass a SentinelFile instance."""
    s.wait()


def _failing():
    raise ValueError("intentional failure")
