# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

# Collection and spawned grading workers import the same test scorer package.
sys.path.insert(0, str(Path(__file__).parent))
