# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from experiments.ferries import datakit_nemotron_ferry, datakit_tier2_skewed_ferry


def _normalize_input(steps, module, monkeypatch) -> str:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(module, "normalize_to_parquet", lambda **kwargs: captured.update(kwargs))
    normalize = next(step for step in steps if step.name.endswith("/normalize"))
    assert normalize.fn is not None
    normalize.fn(normalize.output_path)
    return captured["input_path"]


def test_tier2_ferry_keeps_step_metadata_out_of_raw_storage(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-eu-west4")
    raw_root = "gs://marin-eu-west4/raw/datakit-tier2-skewed-v2-de656ef"
    output_root = "gs://marin-eu-west4/tmp/ttl=1d/datakit-tier2-skewed-smoke/storage-isolation"
    steps = datakit_tier2_skewed_ferry.build_steps("storage-isolation")

    assert all(step.output_path.startswith(output_root) for step in steps)
    assert all(not step.output_path.startswith(raw_root) for step in steps)
    assert _normalize_input(steps, datakit_tier2_skewed_ferry, monkeypatch) == f"{raw_root}/data"


def test_nemotron_ferry_keeps_step_metadata_out_of_raw_storage(monkeypatch):
    raw_root = datakit_nemotron_ferry.NEMOTRON_RAW_PATH
    output_root = "gs://marin-eu-west4/tmp/ttl=1d/datakit-nemotron-smoke/storage-isolation"
    steps = datakit_nemotron_ferry.build_steps(output_root)

    assert all(step.output_path.startswith(output_root) for step in steps)
    assert all(not step.output_path.startswith(raw_root) for step in steps)
    assert _normalize_input(steps, datakit_nemotron_ferry, monkeypatch) == (
        f"{raw_root}/{datakit_nemotron_ferry.NEMOTRON_DATA_SUBDIR}/"
        f"{datakit_nemotron_ferry.NEMOTRON_QUALITY_DIR}"
    )
