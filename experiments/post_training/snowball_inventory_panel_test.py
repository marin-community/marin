# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.snowball_curriculum.bundle import file_sha256
from experiments.post_training.snowball_inventory_development import freeze_development_panel
from experiments.post_training.snowball_inventory_panel import (
    FAMILY_COUNTS,
    DevelopmentPanel,
    pairing_identity,
    select_development_rows,
)


def full_panel_rows():
    return [
        {
            "task_id": f"{family}-{index}",
            "evaluation_family": family,
            "source_contract_id": f"contract-{family}-{index}",
            "task_rollout_spec": '{"grader": "original", "prompt": "unchanged"}',
        }
        for family, count in FAMILY_COUNTS.items()
        for index in range(count)
    ]


def test_fast_selection_is_paired_across_input_orders_and_preserves_contracts():
    rows = full_panel_rows()
    selected = select_development_rows(rows, DevelopmentPanel.FAST)
    assert selected == select_development_rows(list(reversed(rows)), DevelopmentPanel.FAST)
    assert Counter(row["evaluation_family"] for row in selected) == dict.fromkeys(FAMILY_COUNTS, 20)
    originals = {row["task_id"]: row for row in rows}
    assert all(row == originals[row["task_id"]] for row in selected)
    assert select_development_rows(rows, DevelopmentPanel.FULL) == rows


def test_frozen_panel_round_trips_original_records_schema_and_source(tmp_path):
    source = tmp_path / "full.parquet"
    table = pa.Table.from_pylist(full_panel_rows()).replace_schema_metadata({b"source": b"frozen-panel"})
    pq.write_table(table, source)
    source_bytes = source.read_bytes()
    report = freeze_development_panel(source, file_sha256(StoragePath(str(source))), tmp_path / "fast")
    frozen = pq.read_table(tmp_path / "fast" / "validation.parquet")
    assert frozen.schema == table.schema
    assert frozen.to_pylist() == select_development_rows(table.to_pylist(), DevelopmentPanel.FAST)
    assert report["task_ids"] == [row["task_id"] for row in frozen.to_pylist()]
    assert report["pairing_identity"] == pairing_identity(
        [(row["task_id"], row["evaluation_family"], row["source_contract_id"]) for row in frozen.to_pylist()]
    )
    assert source.read_bytes() == source_bytes
