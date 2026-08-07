# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks" / "h100_kv_major_slot_waves.py"


def test_slot_wave_planning_cli_preserves_edges_order_and_attention_semantics(tmp_path: Path) -> None:
    output = tmp_path / "slot-waves.json"
    subprocess.run(
        [
            sys.executable,
            str(BENCHMARK),
            "--sequence-length",
            "96",
            "--block-size",
            "32",
            "--selected-blocks",
            "3",
            "--query-heads",
            "4",
            "--key-value-heads",
            "2",
            "--head-dimension",
            "8",
            "--json-output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output.read_text())
    schedule = result["planning"]["slot_waves"]
    assert result["status"] == "planning_only"
    assert result["gpu"] is None
    assert result["cpu_check"]["allclose_rtol_2e-6_atol_2e-6"]
    assert schedule["edge_count"] == result["planning"]["relation_edges"] == 6
    for wave in schedule["waves"]:
        assert wave["key_value_blocks"] == sorted(wave["key_value_blocks"])
        assert len(wave["query_blocks"]) == len(set(wave["query_blocks"]))
        assert wave["destination_edge_offsets"][-1] == wave["edge_count"]
