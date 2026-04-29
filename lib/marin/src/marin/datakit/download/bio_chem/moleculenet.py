# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoleculeNet SMILES slices via pinned Hugging Face CSVs.

MoleculeNet's canonical exports are mirrored in small Hugging Face dataset
repos. We pin two SMILES-bearing tasks (ESOL solubility and ClinTox toxicity)
so the slice exercises annotated SMILES rows in their natural CSV form (SMILES
plus target columns). ``skip_header_lines=1`` drops each CSV header.

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

MOLECULENET_ESOL_REVISION = "8929f602b69f2a32ae07dd7a4015fa4cc64224ae"
MOLECULENET_CLINTOX_REVISION = "19f3c7b2cc41d158ff70a666de27f76098a1b2e6"
MOLECULENET_ESOL_URL = f"hf://datasets/scikit-fingerprints/MoleculeNet_ESOL@{MOLECULENET_ESOL_REVISION}/esol.csv"
MOLECULENET_CLINTOX_URLS = (
    f"hf://datasets/zpn/clintox@{MOLECULENET_CLINTOX_REVISION}/clintox_train.csv",
    f"hf://datasets/zpn/clintox@{MOLECULENET_CLINTOX_REVISION}/clintox_valid.csv",
    f"hf://datasets/zpn/clintox@{MOLECULENET_CLINTOX_REVISION}/clintox_test.csv",
)

MOLECULENET_SLICES: tuple[NotationSliceSpec, ...] = (
    NotationSliceSpec(
        name="moleculenet_esol_smiles",
        urls=(MOLECULENET_ESOL_URL,),
        fmt=NotationFormat.SMILES,
        source_label="hf:scikit-fingerprints/MoleculeNet_ESOL/esol.csv",
        compression=None,
        sampling=SamplingCap(max_records=2000, max_bytes=2 * 1024 * 1024),
        skip_header_lines=1,
    ),
    NotationSliceSpec(
        name="moleculenet_clintox_smiles",
        urls=MOLECULENET_CLINTOX_URLS,
        fmt=NotationFormat.SMILES,
        source_label="hf:zpn/clintox/csv",
        compression=None,
        sampling=SamplingCap(max_records=3000, max_bytes=2 * 1024 * 1024),
        skip_header_lines=1,
    ),
)


def moleculenet_step() -> ExecutorStep:
    return bio_chem_slice_step(name="raw/bio_chem/moleculenet", slices=MOLECULENET_SLICES)
