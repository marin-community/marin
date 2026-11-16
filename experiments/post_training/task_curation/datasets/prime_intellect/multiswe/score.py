# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright Prime Intellect. Licensed under Apache-2.0.
# Source: research-environments b41d8759070715dc102000e06fba0d73ebce9955.

# Preserve upstream lazy imports after its case-collision shim; the benchmark is grader-only.
# ruff: noqa: PLC0415

import importlib
import sys

from . import _qiskit

# This is the source taskset's own fix for its pinned dependency's case-colliding sdist.
sys.modules.setdefault("multi_swe_bench.harness.repos.python.qiskit", _qiskit)
importlib.import_module("multi_swe_bench.harness.repos.python.Qiskit")


def make_instance(row: dict):
    from multi_swe_bench.harness.dataset import Dataset as MultiSWEDataset
    from multi_swe_bench.harness.image import Config
    from multi_swe_bench.harness.instance import Instance

    dataset = MultiSWEDataset.from_dict(row)
    config = Config(need_clone=False, global_env=None, clear_env=False)
    return dataset, Instance.create(pr=dataset, config=config)


def validate_report(report, multiswe_ds) -> bool:
    if not report.valid:
        return False
    for field in ("p2p_tests", "f2p_tests", "s2p_tests", "n2p_tests"):
        expected = getattr(multiswe_ds, field)
        actual = getattr(report, field)
        for test_name in expected:
            if test_name not in actual:
                return False
    return True


def calculate_reward(test_output: str, row: dict) -> float:
    if not test_output:
        return 0.0
    from multi_swe_bench.harness.report import generate_report

    multiswe_ds, instance = make_instance(row)
    report = generate_report(instance, multiswe_ds.run_result, multiswe_ds.test_patch_result, test_output)
    return float(validate_report(report, multiswe_ds))
