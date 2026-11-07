# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Taskforge integration 9807c5cfb3, defaults/__init__.py.

"""The package's default ``Policy``: ``policy.json`` and the prompt files under ``prompts/``."""

from pathlib import Path

from forge.policy import Policy, load_policy

DEFAULTS_DIR = Path(__file__).parent
DEFAULT_POLICY: Policy = load_policy(DEFAULTS_DIR)

# Experiments enable adversaries explicitly; an empty roster explicitly disables adversarial validation.
