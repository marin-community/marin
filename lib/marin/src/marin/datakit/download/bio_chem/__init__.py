# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming downloaders for biology and chemistry notation slices.

Each submodule defines an ExecutorStep factory for one source family
(RefSeq, RNAcentral, UniProt, PubChem, RCSB PDB, ChEMBL, MoleculeNet) that
streams from the upstream mirror, splits the stream into format-preserving
records via :mod:`marin.transform.bio_chem`, packs short records into longer
documents for in-context-learning evaluation, and writes the result to
plain-text-in-parquet that Levanter can read directly.

The shared streaming primitives live in :mod:`._runtime`.
"""

from marin.datakit.download.bio_chem._runtime import (
    NotationFormat,
    NotationSliceSpec,
    PackingConfig,
    bio_chem_slice_step,
    run_notation_slice,
)

__all__ = [
    "NotationFormat",
    "NotationSliceSpec",
    "PackingConfig",
    "bio_chem_slice_step",
    "run_notation_slice",
]
