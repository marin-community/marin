# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Format-preserving record splitters for biological and chemical notations.

Each splitter takes an iterator of input chunks (lines or text segments) and
yields raw record strings. Splitters do not normalize whitespace, change case,
strip IDs/coordinates, or rewrap sequence lines: the bytes inside each yielded
record are exactly the bytes that appeared in the input.
"""

from marin.transform.bio_chem.splitters import (
    iter_fasta_records,
    iter_gff_blocks,
    iter_mmcif_blocks,
    iter_sdf_records,
    iter_smiles_records,
    iter_uniprot_dat_records,
    pack_records_into_docs,
    take_until_cap,
)
