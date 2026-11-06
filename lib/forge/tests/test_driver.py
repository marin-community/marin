# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from forge.driver import inputs, proposal_key


def test_manifest_selects_ranks_and_path_list_preserves_output_identity(tmp_path: Path) -> None:
    paths = tuple(f"s3://bucket/run/d02.swe.repair/slot-{slot}.proposal.json" for slot in (15, 8, 24))
    sample = tmp_path / "sample.json"
    sample.write_text(
        json.dumps(
            {
                "format": "acceptance-measure-sample-v1",
                "sample": [
                    {"rank": rank, "proposal": path, "capability": "d02.swe.repair", "slot": slot}
                    for rank, path, slot in zip((0, 1, 2), paths, (15, 8, 24), strict=True)
                ],
            }
        )
    )
    assert inputs(str(sample), (0, 2)) == (paths[0], paths[2])
    assert str(proposal_key(paths[0])) == "d02.swe.repair/15"
    manifest = tmp_path / "proposals.txt"
    manifest.write_text("\n".join(paths) + "\n")
    assert inputs(str(manifest)) == paths
    with pytest.raises(ValueError, match="missing"):
        inputs(str(sample), (3,))
    manifest.write_text(f"{paths[0]}\n{paths[0]}\n")
    with pytest.raises(ValueError, match="duplicate output"):
        inputs(str(manifest))


def test_empty_prefix_fails_instead_of_reporting_a_successful_empty_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No proposal"):
        inputs(str(tmp_path))
