# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""UniProt/Swiss-Prot protein notation slices.

We use the Swiss-Prot subset (the manually-reviewed half of UniProtKB) because
it is small enough (~90 MB FASTA, ~700 MB DAT compressed) to stream a
representative head without dominating the pilot. UniRef and TrEMBL would be
preferable for scale; add them as separate slices if you need them.

* ``uniprot_sprot.fasta.gz``: protein FASTA — wide line-wrapped sequences with
  the rich ``>sp|...|``-style header that is itself a notation.
* ``uniprot_sprot.dat.gz``: flat-file metadata (``ID``/``AC``/``DE``/``FT``
  records). Each entry ends with ``//`` on its own line. We treat the whole
  thing as FASTA-style "records" by repurposing the SDF splitter on the ``//``
  delimiter — but to keep the splitter API tight, we split on ``//`` lines via
  the dedicated UniProt helper here.
"""

from __future__ import annotations

from marin.datakit.download.bio_chem._runtime import (
    NotationFormat,
    NotationSliceSpec,
    bio_chem_slice_step,
)
from marin.execution.executor import ExecutorStep
from marin.transform.bio_chem.splitters import SamplingCap

UNIPROT_BASE = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
UNIPROT_SPROT_FASTA_URL = f"{UNIPROT_BASE}/uniprot_sprot.fasta.gz"
UNIPROT_SPROT_DAT_URL = f"{UNIPROT_BASE}/uniprot_sprot.dat.gz"


UNIPROT_SLICES: tuple[NotationSliceSpec, ...] = (
    NotationSliceSpec(
        name="uniprot_sprot_fasta",
        urls=(UNIPROT_SPROT_FASTA_URL,),
        fmt=NotationFormat.FASTA,
        source_label="uniprot:sprot.fasta",
        sampling=SamplingCap(max_records=5000, max_bytes=32 * 1024 * 1024),
    ),
    NotationSliceSpec(
        name="uniprot_sprot_dat",
        urls=(UNIPROT_SPROT_DAT_URL,),
        fmt=NotationFormat.UNIPROT_DAT,
        source_label="uniprot:sprot.dat",
        sampling=SamplingCap(max_records=2500, max_bytes=32 * 1024 * 1024),
    ),
)


def uniprot_sprot_step() -> ExecutorStep:
    return bio_chem_slice_step(name="raw/bio_chem/uniprot_sprot", slices=UNIPROT_SLICES)
