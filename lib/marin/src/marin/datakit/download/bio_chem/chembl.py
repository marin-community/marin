# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ChEMBL compound and bioactivity record slices.

The ChEMBL release directory exposes per-version flat files. We pin a release
version explicitly so reruns are reproducible; bump ``CHEMBL_VERSION`` when
upstream data is refreshed.

Slices:

* ``chembl_<v>_chemreps.txt.gz`` — TSV with ``chembl_id``, ``canonical_smiles``,
  ``standard_inchi``, ``standard_inchi_key``. We keep the full row so IDs,
  SMILES, InChI strings, and key metadata remain in their original notation.
* ``chembl_<v>.sdf.gz`` — full SDF dump; capped to a small head.
"""

from __future__ import annotations

from marin.datakit.download.bio_chem._runtime import (
    NotationFormat,
    NotationSliceSpec,
    bio_chem_slice_step,
)
from marin.execution.executor import ExecutorStep
from marin.transform.bio_chem.splitters import SamplingCap

CHEMBL_BASE = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases"
CHEMBL_VERSION = "chembl_34"
CHEMBL_CHEMREPS_URL = f"{CHEMBL_BASE}/{CHEMBL_VERSION}/{CHEMBL_VERSION}_chemreps.txt.gz"
CHEMBL_SDF_URL = f"{CHEMBL_BASE}/{CHEMBL_VERSION}/{CHEMBL_VERSION}.sdf.gz"

CHEMBL_SLICES: tuple[NotationSliceSpec, ...] = (
    NotationSliceSpec(
        name="chembl_chemreps",
        urls=(CHEMBL_CHEMREPS_URL,),
        fmt=NotationFormat.SMILES,
        source_label=f"chembl:{CHEMBL_VERSION}/chemreps",
        sampling=SamplingCap(max_records=5000, max_bytes=4 * 1024 * 1024),
        skip_header_lines=1,
    ),
    NotationSliceSpec(
        name="chembl_sdf",
        urls=(CHEMBL_SDF_URL,),
        fmt=NotationFormat.SDF,
        source_label=f"chembl:{CHEMBL_VERSION}/sdf",
        sampling=SamplingCap(max_records=1000, max_bytes=64 * 1024 * 1024),
    ),
)


def chembl_step() -> ExecutorStep:
    return bio_chem_slice_step(name=f"raw/bio_chem/{CHEMBL_VERSION}", slices=CHEMBL_SLICES)
