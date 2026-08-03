# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

import benchmark_trained_fast_student as benchmark_module  # noqa: E402


class Model:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def __call__(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.calls.append(self.name)
        return np.ones((len(texts), 2), dtype=np.float32)


def test_paired_rates_reject_unstable_measurement(monkeypatch) -> None:
    rates = {
        "student": iter([20.0, 21.0, 19.0, 20.5, 19.5]),
        "baseline": iter([1.0, 10.0, 10.0, 10.0, 10.0]),
    }

    def timed_rate(model: Model, texts: list[str], batch_size: int) -> tuple[float, float]:
        rate = next(rates[model.name])
        return len(texts) / rate, rate

    monkeypatch.setattr(benchmark_module, "SPEED_REPEATS", 5)
    monkeypatch.setattr(benchmark_module, "timed_rate", timed_rate)
    calls = []
    student = Model("student", calls)
    baseline = Model("baseline", calls)

    result = benchmark_module.paired_rates(student, baseline, ["a", "b"], batch_size=2)

    assert result["student_documents_per_second"] == 20.0
    assert result["baseline_documents_per_second"] == 10.0
    assert result["student_to_baseline_ratio"] == 2.0
    assert result["student_stability"]["passed"]
    assert not result["baseline_stability"]["passed"]
    assert not result["measurement_valid"]
