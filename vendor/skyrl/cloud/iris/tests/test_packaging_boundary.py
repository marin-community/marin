"""Import-boundary tests for the CPU-only launcher installation."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize(
    "module,blocked",
    [
        ("cloud.iris.launch", ("flash_attn", "ray", "skyrl_train.objective", "skyrl_train.trainer", "torch", "vllm")),
        (
            "skyrl_train.entrypoints.main_base",
            ("flash_attn", "skyrl_train.objective", "skyrl_train.trainer", "torch", "vllm"),
        ),
    ],
)
def test_importing_entrypoints_does_not_import_training_stacks(module, blocked) -> None:
    program = f"""
import json
import sys

import {module}

blocked = {blocked!r}
print(json.dumps(sorted(name for name in blocked if name in sys.modules)))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


def test_importing_hf_export_does_not_import_training_stacks() -> None:
    """The launcher's export step imports skyrl_train.hf_export in a torch-free environment."""
    program = """
import json
import sys

import skyrl_train.hf_export

blocked = ("flash_attn", "ray", "torch", "vllm")
print(json.dumps(sorted(name for name in blocked if name in sys.modules)))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []
