# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared storage-layout constants for the canary ferry.

Kept separate from ``canary_ferry.py`` so consumers (e.g. the prune script)
can import the layout without triggering that module's heavy step-build at
import time.
"""

# Subdirectory of MARIN_PREFIX that the canary ferry writes its per-run output
# dirs into. Nesting keeps per-run dirs out of the MARIN_PREFIX root and lets
# scripts/canary/prune_canary_outputs.py sweep the whole subdir.
CANARY_OUTPUT_SUBDIR = "canary"
