# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training.snowball_isoflop import run
from experiments.post_training.snowball_isoflop.config import VERSION, Cell, Mask


def test_primary_launch_rejects_an_older_calibrated_runtime_before_dispatch(tmp_path, monkeypatch):
    cell = Cell(512, Mask.FULL, 0, 20, (20,), "A")
    report = tmp_path / "calibration-report.json"
    report.write_text(
        json.dumps(
            {
                "version": VERSION,
                "all_six_cells_valid": True,
                "expansion_gate_passed": True,
                "projected_primary_gpu_hours": 100,
                "inclusive_step_seconds": {"512": 1},
                "runtime_environment_by_cell": {
                    "512-full": {
                        "XLA_FLAGS": "--xla_gpu_shard_autotuning=false --xla_gpu_nccl_termination_timeout_seconds=600",
                        "CUDA_LAUNCH_BLOCKING": None,
                    }
                },
            }
        )
    )
    monkeypatch.setattr(run, "primary_cells", lambda: (cell,))
    monkeypatch.setattr(
        "sys.argv", ["run", "--phase", "A", "--run", "--calibration-report", str(report), "--h100-hour-limit", "100"]
    )
    monkeypatch.setattr(run, "budgeted_map", lambda *args, **kwargs: pytest.fail("GPU dispatch must remain gated"))
    with pytest.raises(ValueError, match="A runtime does not match its calibrated recipe"):
        run.main()
