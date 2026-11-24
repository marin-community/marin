# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Taskforge integration 9807c5cfb3, defaults/__init__.py.

"""The default policy directory, for ``taskforge.policy.load_policy``: ``policy.json`` and ``prompts/``."""

from pathlib import Path

DEFAULTS_DIR = Path(__file__).parent
