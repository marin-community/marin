# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Make absolute imports of RolloutEngine test fixtures work in package-only runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
