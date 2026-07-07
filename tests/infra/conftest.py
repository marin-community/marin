# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `infra.ci.*` is importable.
# Needed because pytest's --import-mode=importlib doesn't guarantee this even
# with `pythonpath = ["."]` in pyproject.toml during the collection phase.
_repo_root = str(Path(__file__).parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
