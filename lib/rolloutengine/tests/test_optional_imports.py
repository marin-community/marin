# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import contracts for optional platform dependencies."""

import subprocess
import sys


def test_rolloutengine_imports_without_linux_local_backend():
    # macOS cannot install bubblewrap_bin; check the same import boundary in a fresh interpreter.
    probe = """
import sys

class WithoutBubblewrap:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bubblewrap_bin":
            raise ModuleNotFoundError("No module named 'bubblewrap_bin'")

sys.meta_path.insert(0, WithoutBubblewrap())
import rolloutengine.engine
import rolloutengine.lowering
import rolloutengine.machines
import taskcompendium.runtime.local
"""
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
