# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Historical ingest code is evidence, never an executable inventory input."""

import csv
import hashlib
from pathlib import Path

from experiments.post_training.task_curation.ingest_stages.inventory import inventory_rows, write_inventory


def test_inventory_preserves_original_and_copied_scripts_without_executing(tmp_path: Path):
    root = tmp_path / "history"
    lane = root / "agent-a"
    lane.mkdir(parents=True)
    marker = tmp_path / "executed"
    script = lane / "census.py"
    script.write_text(
        "from experiments.post_training.task_curation.published_census import measure_population\n"
        f"open({str(marker)!r}, 'w').write('executed')\n"
    )
    copy = lane / "upstream" / script.name
    copy.parent.mkdir()
    copy.write_bytes(script.read_bytes())
    friction = root / "FRICTION-A.md"
    friction.write_text("# A failed regional launch\n")
    checkout = Path(__file__).resolve().parents[4]

    output = tmp_path / "report"
    write_inventory(inventory_rows(root, checkout), output)

    with (output / "inventory.tsv").open() as stream:
        rows = {row["path"]: row for row in csv.DictReader(stream, delimiter="\t")}
    assert rows["agent-a/census.py"]["disposition"] == "reuse-review"
    assert rows["agent-a/upstream/census.py"]["disposition"] == "historical-copy"
    assert rows["agent-a/census.py"]["sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert rows["FRICTION-A.md"]["disposition"] == "friction-review"
    assert not marker.exists()
    assert script.read_bytes() == copy.read_bytes()
