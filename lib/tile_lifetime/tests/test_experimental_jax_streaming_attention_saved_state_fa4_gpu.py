# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from lib.tile_lifetime.benchmarks.experimental_jax_streaming_attention_saved_state_fa4_gpu import (
    GENERATED_FORWARD_TARGET,
    GENERATED_REVERSE_TARGET,
    _require_fresh_directory,
    _source_audit,
    _verify_compiled_hlo_audit,
)

REPOSITORY = Path(__file__).resolve().parents[3]


def test_fa4_source_audit_covers_the_physical_kernel_sources() -> None:
    audit = _source_audit(REPOSITORY)
    audited_paths = {source["path"] for source in audit["sources"]}

    assert "lib/levanter/src/levanter/grug/attention/_fa4_cute_kernels.py" in audited_paths
    assert "lib/levanter/src/levanter/grug/attention/_fa4_cute_segmented_bwd.py" in audited_paths
    assert all(len(source["sha256"]) == 64 and source["bytes"] > 0 for source in audit["sources"])


def test_benchmark_requires_fresh_artifact_and_build_directories(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh"
    _require_fresh_directory(fresh, label="artifact directory")
    assert fresh.is_dir()

    (fresh / "stale.json").write_text("{}")
    with pytest.raises(ValueError, match="fresh empty directory"):
        _require_fresh_directory(fresh, label="artifact directory")


def test_compiled_boundary_rejects_missing_generated_target() -> None:
    audit = {
        "entry_layout": "HloModule generated",
        "contains_custom_call": True,
        "custom_call_targets": (GENERATED_FORWARD_TARGET,),
    }

    _verify_compiled_hlo_audit(
        audit,
        boundary_name="generated forward",
        expected_target=GENERATED_FORWARD_TARGET,
    )
    with pytest.raises(ValueError, match="does not contain generated target"):
        _verify_compiled_hlo_audit(
            audit,
            boundary_name="generated reverse",
            expected_target=GENERATED_REVERSE_TARGET,
        )
