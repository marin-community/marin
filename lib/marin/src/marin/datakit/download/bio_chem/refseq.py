# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NCBI RefSeq notation slices: viral genome FASTA and GFF annotations.

The NCBI viral release is intentionally chosen as the default: it is the
smallest curated RefSeq subdivision (a few GB compressed) and exercises both
short genomic FASTA records and dense GFF gene blocks. Sampling caps mean we
only stream the head of each file; the deterministic record limits keep the
slice reproducible across runs.

Bacterial / fungal RefSeq subdivisions are far larger; if you want them, build
another step with their FTP URLs.
"""

from __future__ import annotations

from marin.datakit.download.bio_chem._runtime import (
    NotationFormat,
    NotationSliceSpec,
    bio_chem_slice_step,
)
from marin.execution.step_spec import StepSpec
from marin.transform.bio_chem.splitters import SamplingCap

REFSEQ_BASE = "https://ftp.ncbi.nlm.nih.gov/refseq/release/viral"
REFSEQ_FASTA_URL = f"{REFSEQ_BASE}/viral.1.1.genomic.fna.gz"
REFSEQ_GFF_URL = f"{REFSEQ_BASE}/viral.1.gff.gz"

REFSEQ_SLICES: tuple[NotationSliceSpec, ...] = (
    NotationSliceSpec(
        name="refseq_viral_fasta",
        urls=(REFSEQ_FASTA_URL,),
        fmt=NotationFormat.FASTA,
        source_label="ncbi:refseq/viral/genomic.fna",
        sampling=SamplingCap(max_records=4000, max_bytes=64 * 1024 * 1024),
    ),
    NotationSliceSpec(
        name="refseq_viral_gff",
        urls=(REFSEQ_GFF_URL,),
        fmt=NotationFormat.GFF,
        source_label="ncbi:refseq/viral/annotations.gff",
        sampling=SamplingCap(max_records=2000, max_bytes=32 * 1024 * 1024),
    ),
)


def refseq_viral_step() -> StepSpec:
    return bio_chem_slice_step(name="raw/bio_chem/refseq_viral", slices=REFSEQ_SLICES)
