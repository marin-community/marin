# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoleculeNet SMILES slice via deepchem-hosted CSVs.

MoleculeNet's canonical exports live on the public deepchem S3 bucket. We pin
two small SMILES-bearing tasks (ESOL solubility and ClinTox toxicity) so the
slice exercises annotated SMILES rows in their natural CSV form (SMILES plus
target columns). ``skip_header_lines=1`` drops the CSV header.

The SMILES splitter keeps each non-comment line verbatim, so the model sees
the original ``smiles,label,...`` row format — that *is* the notation worth
modelling. To add more MoleculeNet tasks, append more ``NotationSliceSpec``
entries pointing at the deepchem CSV/CSV.gz URLs.
"""

from __future__ import annotations

from marin.datakit.download.bio_chem._runtime import (
    NotationFormat,
    NotationSliceSpec,
    bio_chem_slice_step,
)
from marin.execution.executor import ExecutorStep
from marin.transform.bio_chem.splitters import SamplingCap

MOLECULENET_BASE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"
MOLECULENET_ESOL_URL = f"{MOLECULENET_BASE}/delaney-processed.csv"
MOLECULENET_CLINTOX_URL = f"{MOLECULENET_BASE}/clintox.csv.gz"

MOLECULENET_SLICES: tuple[NotationSliceSpec, ...] = (
    NotationSliceSpec(
        name="moleculenet_esol_smiles",
        urls=(MOLECULENET_ESOL_URL,),
        fmt=NotationFormat.SMILES,
        source_label="moleculenet:esol/delaney",
        compression=None,
        sampling=SamplingCap(max_records=2000, max_bytes=2 * 1024 * 1024),
        skip_header_lines=1,
    ),
    NotationSliceSpec(
        name="moleculenet_clintox_smiles",
        urls=(MOLECULENET_CLINTOX_URL,),
        fmt=NotationFormat.SMILES,
        source_label="moleculenet:clintox",
        sampling=SamplingCap(max_records=2000, max_bytes=2 * 1024 * 1024),
        skip_header_lines=1,
    ),
)


def moleculenet_step() -> ExecutorStep:
    return bio_chem_slice_step(name="raw/bio_chem/moleculenet", slices=MOLECULENET_SLICES)
