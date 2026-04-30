# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RNAcentral non-coding RNA FASTA slice.

Streams ``rnacentral_active.fasta.gz`` from the EBI mirror. The full release is
~10 GB compressed; sampling caps stop us from reading more than the head we
need to estimate perplexity on RNA notation.
"""

from __future__ import annotations

from marin.datakit.download.bio_chem._runtime import (
    NotationFormat,
    NotationSliceSpec,
    bio_chem_slice_step,
)
from marin.execution.step_spec import StepSpec
from marin.transform.bio_chem.splitters import SamplingCap

RNACENTRAL_BASE = "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences"
RNACENTRAL_FASTA_URL = f"{RNACENTRAL_BASE}/rnacentral_active.fasta.gz"

RNACENTRAL_SLICES: tuple[NotationSliceSpec, ...] = (
    NotationSliceSpec(
        name="rnacentral_active_fasta",
        urls=(RNACENTRAL_FASTA_URL,),
        fmt=NotationFormat.FASTA,
        source_label="rnacentral:active.fasta",
        sampling=SamplingCap(max_records=5000, max_bytes=64 * 1024 * 1024),
    ),
)


def rnacentral_step() -> StepSpec:
    return bio_chem_slice_step(name="raw/bio_chem/rnacentral", slices=RNACENTRAL_SLICES)
