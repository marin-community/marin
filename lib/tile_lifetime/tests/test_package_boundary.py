# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys

import tile_lifetime


def test_tile_lifetime_root_does_not_reexport_moved_shuttle_types() -> None:
    assert not hasattr(tile_lifetime, "DType")


def test_historical_importers_are_not_current_module_paths() -> None:
    assert importlib.util.find_spec("shuttle.stablehlo_import") is None
    assert importlib.util.find_spec("tile_lifetime.stablehlo_import") is None
    assert importlib.util.find_spec("tile_lifetime.pipeline") is None
    assert importlib.util.find_spec("tile_lifetime.stablehlo_scan_recovery") is None
    assert importlib.util.find_spec("tile_lifetime.stablehlo_row_normalization_backward") is None
    assert importlib.util.find_spec("tile_lifetime.stablehlo_streaming_attention_backward") is None
    assert importlib.util.find_spec("tile_lifetime.msa_recovery") is None
    assert importlib.util.find_spec("tile_lifetime.routed_attention_recovery") is None
    assert importlib.util.find_spec("tile_lifetime.semantic_recovery") is None
    assert importlib.util.find_spec("tile_lifetime.moe_recovery") is None


def test_tile_lifetime_root_does_not_export_recovery_or_hlo_rewrite_bridges() -> None:
    forbidden = (
        "compile_natural_projected_routed_attention",
        "compile_natural_routed_attention",
        "recover_projected_routed_attention_program",
        "recover_routed_attention_program",
        "recover_low_rank_gated_product_training",
        "plan_axis_fold_pipeline_hlo_replacement",
        "replace_axis_fold_pipeline_hlo_with_custom_call",
        "plan_streaming_attention_training_regions",
        "replace_streaming_attention_training_regions_with_custom_calls",
    )

    assert all(not hasattr(tile_lifetime, name) for name in forbidden)


def test_importing_tile_lifetime_does_not_load_experimental_frontends() -> None:
    script = """
import json
import sys
import tile_lifetime

print(json.dumps(sorted(
    name for name in sys.modules
    if name.startswith('shuttle.experimental') or name.startswith('tile_lifetime.experimental')
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
