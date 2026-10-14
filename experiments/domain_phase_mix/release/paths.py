# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source and output locations of the dataset export. Every source is read-only."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RO = REPO / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs"
AG = REPO / ".agents/projects"
MIXSEL = Path.home() / "Projects/Work/Marin/mixture-selection"
DRIVE = Path(
    "/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/"
    "data_mixing_paper_one_phase"
)
BUILD = REPO / "release_build"
PRIVATE = Path.home() / ".config/mariner_release"  # salt and observation map; never inside the upload
D_TGT = 6_325_183_647_689
PARTITION_REVISION = "dolma3_dolmino_39bucket_2026-06"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"release source missing: {path}")
    return path
