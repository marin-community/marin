# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

import benchmark_fast_embedding_accelerator as benchmark_module  # noqa: E402


class Model:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.calls += 1
        return np.ones((len(texts), 2), dtype=np.float32)


def test_accelerator_rates_use_full_warmup_and_stable_repeats(monkeypatch) -> None:
    rates = iter([rate for rate in [100.0, 102.0, 98.0, 101.0, 99.0] for _ in range(5)])
    timed_calls = 0

    def timed_rate(model: Model, texts: list[str], batch_size: int) -> tuple[float, float]:
        nonlocal timed_calls
        timed_calls += 1
        rate = next(rates)
        return len(texts) / rate, rate

    monkeypatch.setattr(benchmark_module, "SPEED_REPEATS", 5)
    monkeypatch.setattr(benchmark_module, "timed_rate", timed_rate)
    model = Model()

    result = benchmark_module.accelerator_rates(model, ["a", "b"], batch_size=2, calls_per_repeat=5)

    assert model.calls == 1
    assert timed_calls == 25
    assert result["student_documents_per_second"] == 100.0
    assert result["measurement_valid"] is True
