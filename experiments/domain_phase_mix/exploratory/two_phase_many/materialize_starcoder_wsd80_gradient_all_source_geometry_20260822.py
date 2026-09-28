# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs>=2025.5.1",
#   "pandas>=2.2",
# ]
# ///
"""Materialize descriptive source-source geometry at every repaired checkpoint."""

import argparse
import json
import tempfile
from pathlib import Path

import gcsfs

from experiments.domain_phase_mix import starcoder_wsd80_gradient_mechanism_repair as runtime
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_gradient_mechanism_repair_20260818 as analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_mechanism_repair_20260818 as freeze,
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).parent / "reference_outputs" / "starcoder_wsd80_gradient_mechanism_all_source_geometry_20260822"
)
SCIENTIFIC_STATUS = "post_outcome_descriptive_extension_not_untouched_confirmation"


def materialize(output_dir: Path) -> None:
    release_sha256 = json.loads(freeze.RELEASE_PATH.read_text())["release_sha256"]
    runtime._load_release(release_sha256)
    documents = analysis.load_documents(gcsfs.GCSFileSystem(), release_sha256)
    geometry = analysis.flatten_h1(documents)

    expected_row_ids = {row["row_id"] for row in analysis._read_manifest()}
    observed_row_ids = set(geometry["row_id"])
    if observed_row_ids != expected_row_ids:
        raise RuntimeError("All-state source geometry does not cover the exact frozen repair manifest")
    if geometry.duplicated(["row_id", "statistic", "geometry", "component"]).any():
        raise RuntimeError("All-state source geometry contains duplicate statistic rows")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    geometry.to_csv(staging / "source_source_geometry_all_states.csv", index=False)
    audit = {
        "documents": len(documents),
        "endpoint_metrics_read": False,
        "manifest_rows": len(expected_row_ids),
        "release_sha256": release_sha256,
        "rows": len(geometry),
        "scientific_status": SCIENTIFIC_STATUS,
        "states": sorted(geometry["checkpoint_label"].unique()),
    }
    (staging / "materialization_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    analysis._publish_create_only(staging, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    materialize(args.output_dir)


if __name__ == "__main__":
    main()
