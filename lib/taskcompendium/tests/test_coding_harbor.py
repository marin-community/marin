# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retained TaskTrove coding archives graded in real Docker Harbor trials."""

import json
import shlex
from pathlib import Path

import pytest

from taskcompendium.execution import (
    ChatWithTools,
    HarborExecutionConfig,
    HarnessToolBinding,
    environment_for_requirements,
)
from taskcompendium.harbor.runner import run_trial
from taskcompendium.importers.tasktrove import read_archive
from taskcompendium.importers.tasktrove_coding import import_task
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import FinalState, Rejected, Rendering

pytestmark = pytest.mark.docker
FIXTURES = Path(__file__).parent / "fixtures/coding"

# These archives contain no oracle. These are validation-only implementations
# written from the original instructions, not retained source solutions.
PYTHON_SOLUTION = """
import enum
import struct

class EnumWriteOps(enum.Enum):
    WRITE_VALUE = 1
class EnumItm(enum.Enum):
    ENG = 1
class EnumEngine(enum.Enum):
    ANY = 1

class CmdWriteData:
    def __init__(self, data=None, ops=None):
        self.data = [list(pair) for pair in (data or [])]
        self.ops = EnumWriteOps.WRITE_VALUE if ops is None else ops
    def append(self, address, value):
        self.data.append([address, value])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    @property
    def bytes(self):
        return 4
    def export(self):
        return struct.pack('<I', self.ops.value) + b''.join(struct.pack('<II', *pair) for pair in self.data)
    @classmethod
    def parse(cls, data):
        return cls(list(struct.iter_unpack('<II', data[4:])), EnumWriteOps(struct.unpack('<I', data[:4])[0]))
    def __eq__(self, other):
        return isinstance(other, CmdWriteData) and self.ops == other.ops and self.data == other.data

class CmdNop:
    def __init__(self, param):
        self.param = param
    @property
    def size(self):
        return 4
    def export(self):
        return struct.pack('<I', self.param)
    @classmethod
    def parse(cls, data):
        return cls(struct.unpack('<I', data)[0])
    def __eq__(self, other):
        return isinstance(other, CmdNop) and self.param == other.param
"""

CPP_SOLUTION = r"""
#include <iostream>
#include <cstdlib>
int main() {
    int n, d, m, x, y;
    std::cin >> n >> d >> m;
    while (m--) {
        std::cin >> x >> y;
        bool inside = d <= x+y && x+y <= 2*n-d && std::abs(x-y) <= d;
        std::cout << (inside ? "YES" : "NO") << '\n';
    }
}
"""


@pytest.mark.parametrize("family,row,paths", [("pytest", 1487, ("imx",)), ("stdio", 0, ("solution.cpp",))])
@pytest.mark.parametrize("attempt,reward", [("good", 1.0), ("bad", 0.0), ("empty", 0.0)])
async def test_real_coding_source_grader(tmp_path, runtime_image, family, row, paths, attempt, reward):
    archive = read_archive((FIXTURES / f"{family}-row-{row}.tar.gz").read_bytes(), str(row), "coding")
    spec = import_task(archive, python_image=runtime_image, native_image=runtime_image)
    assert not isinstance(spec, Rejected)
    commands = []
    if attempt != "empty":
        if family == "pytest":
            source = PYTHON_SOLUTION if attempt == "good" else PYTHON_SOLUTION.replace("return 4", "return 99")
            commands = ["mkdir -p imx; touch imx/__init__.py", f"printf '%s' {shlex.quote(source)} > imx/img.py"]
        else:
            source = (
                CPP_SOLUTION if attempt == "good" else CPP_SOLUTION.replace("bool inside =", "bool inside = false &&")
            )
            commands = [f"printf '%s' {shlex.quote(source)} > solution.cpp"]
    task = lower_to_harbor(
        spec,
        (Rendering("code", FinalState(paths)),),
        HarborExecutionConfig(
            "replay",
            environment_for_requirements(spec.requirements),
            interaction=(ChatWithTools((HarnessToolBinding("replay", "docker"),))),
        ),
        tmp_path / "task",
        agent_kwargs={"commands": commands},
    )
    execution = json.loads((task / "execution.json").read_text())
    result = await run_trial(task, execution, tmp_path / "trials", "coding")
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": reward}
    detail = json.loads((tmp_path / "trials/coding/verifier/taskcompendium-result.json").read_text())["detail"]
    if attempt == "good":
        assert detail["total"] == (5 if family == "pytest" else 17)
