# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for Marin IaC."""

import os
import sys

# Pulumi runs this file from infra/pulumi/ without installing marin-iac first.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from iac.program import build_stack

build_stack()
