# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright Prime Intellect. Licensed under Apache-2.0.
# Source: research-environments b41d8759070715dc102000e06fba0d73ebce9955.

"""Qiskit adapter omitted by multi-swe-bench 1.1.2's case-colliding source archive.

The adapter logic is from multi-swe-bench 1.1.2 (Apache-2.0).
"""

import importlib
import re
import sys
from pathlib import Path

from multi_swe_bench.harness.image import Config, Image, SWEImageDefault
from multi_swe_bench.harness.instance import Instance, TestResult
from multi_swe_bench.harness.pull_request import PullRequest
from multi_swe_bench.harness.test_result import TestStatus, mapping_to_testresult

# The source archive also overwrites the uppercase package's import list. Import its
# remaining adapters so every Qiskit dataset row keeps its upstream registration.
qiskit_package = sys.modules["multi_swe_bench.harness.repos.python.Qiskit"]
for path in Path(qiskit_package.__file__).parent.glob("*.py"):
    if path.stem != "__init__":
        importlib.import_module(f"{qiskit_package.__name__}.{path.stem}")


@Instance.register("qiskit", "qiskit")
class Qiskit(Instance):
    def __init__(self, pr: PullRequest, config: Config, *args, **kwargs):
        super().__init__()
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    def dependency(self) -> Image | None:
        return SWEImageDefault(self.pr, self._config)

    def fix_patch_run(self, fix_patch_run_cmd: str = "") -> str:
        return fix_patch_run_cmd or "bash /home/fix-run.sh"

    def parse_log(self, log: str) -> TestResult:
        test_status_map = {}
        escapes = "".join(chr(char) for char in range(1, 32))
        for line in log.split("\n"):
            line = re.sub(r"\[(\d+)m", "", line).translate(str.maketrans("", "", escapes))
            if any(line.startswith(status.value) for status in TestStatus):
                if line.startswith(TestStatus.FAILED.value):
                    line = line.replace(" - ", " ")
                test_case = line.split()
                if len(test_case) >= 2:
                    test_status_map[test_case[1]] = test_case[0]
            elif any(line.endswith(status.value) for status in TestStatus):
                test_case = line.split()
                if len(test_case) >= 2:
                    test_status_map[test_case[0]] = test_case[1]
        return mapping_to_testresult(test_status_map)
