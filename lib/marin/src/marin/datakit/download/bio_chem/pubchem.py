# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PubChem compound notation slices: SMILES (TSV) and SDF.

* ``CID-SMILES.gz`` is a 2-column TSV (CID, SMILES) — we keep the full line so
  the CID is preserved alongside the SMILES, mirroring PubChem's own export.
* ``Compound_000000001_000500000.sdf.gz`` is the first half-million-CID SDF
  shard. SDF entries are tens of KB each so the cap matters.
"""

from __future__ import annotations

from marin.datakit.download.bio_chem._runtime import (
    NotationFormat,
    NotationSliceSpec,
    bio_chem_slice_step,
)
from marin.execution.step_spec import StepSpec
from marin.transform.bio_chem.splitters import SamplingCap

PUBCHEM_BASE = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound"
PUBCHEM_CID_SMILES_URL = f"{PUBCHEM_BASE}/Extras/CID-SMILES.gz"
PUBCHEM_FIRST_SDF_SHARD_URL = f"{PUBCHEM_BASE}/CURRENT-Full/SDF/Compound_000000001_000500000.sdf.gz"

PUBCHEM_SLICES: tuple[NotationSliceSpec, ...] = (
    NotationSliceSpec(
        name="pubchem_cid_smiles",
        urls=(PUBCHEM_CID_SMILES_URL,),
        fmt=NotationFormat.SMILES,
        source_label="pubchem:CID-SMILES",
        sampling=SamplingCap(max_records=5000, max_bytes=4 * 1024 * 1024),
    ),
    NotationSliceSpec(
        name="pubchem_sdf",
        urls=(PUBCHEM_FIRST_SDF_SHARD_URL,),
        fmt=NotationFormat.SDF,
        source_label="pubchem:Compound/SDF/first-shard",
        sampling=SamplingCap(max_records=1000, max_bytes=64 * 1024 * 1024),
    ),
)


def pubchem_step() -> StepSpec:
    return bio_chem_slice_step(name="raw/bio_chem/pubchem", slices=PUBCHEM_SLICES)
