# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RCSB PDB mmCIF structure files.

RCSB does not expose a single bulk URL for streaming, so we pin a small
deterministic list of well-known PDB IDs and fetch one ``.cif.gz`` per entry
via the public download server. Each ``.cif.gz`` contains exactly one mmCIF
``data_<id>`` block, which the splitter recognises.

The default list is biased toward classic, small, biologically diverse
structures (insulin, hemoglobin, ribozyme, GFP, ribosome subunits, viral
capsid, etc.) so the slice exercises a range of mmCIF dialects without pulling
megabytes per entry. Override ``RCSB_DEFAULT_IDS`` to widen the slice.
"""

from __future__ import annotations

from marin.datakit.download.bio_chem._runtime import (
    NotationFormat,
    NotationSliceSpec,
    bio_chem_slice_step,
)
from marin.execution.step_spec import StepSpec
from marin.transform.bio_chem.splitters import SamplingCap

RCSB_DOWNLOAD_BASE = "https://files.rcsb.org/download"

# Curated small/diverse PDB IDs. Keep these alphabetised so changes show up in
# diffs, and keep the list small (~64 entries, ~10-30 KB each compressed) so
# the slice fits the "streaming-not-mirroring" rule.
RCSB_DEFAULT_IDS: tuple[str, ...] = (
    "101M",
    "102D",
    "1AKI",
    "1ALK",
    "1AOI",
    "1ATN",
    "1BNA",
    "1CRN",
    "1D66",
    "1EHZ",
    "1F88",
    "1FAT",
    "1G6N",
    "1GFL",
    "1HHO",
    "1HIV",
    "1IGT",
    "1J4N",
    "1KX5",
    "1LMB",
    "1LYZ",
    "1M14",
    "1MBN",
    "1NCD",
    "1OCA",
    "1OEL",
    "1PGB",
    "1PRH",
    "1Q2W",
    "1RNA",
    "1RUV",
    "1SVA",
    "1TIM",
    "1UBQ",
    "1V0G",
    "1W0E",
    "1YJP",
    "2BEG",
    "2BG9",
    "2CAG",
    "2DEZ",
    "2GS2",
    "2HHB",
    "2J67",
    "2LYZ",
    "2MGT",
    "2P4N",
    "2PTC",
    "2WDK",
    "3FXI",
    "3GBI",
    "3HAQ",
    "3J3Q",
    "3PQR",
    "4HHB",
    "4HOJ",
    "4LZT",
    "4O9S",
    "4UN3",
    "5IRE",
    "5J7V",
    "6CMO",
    "6VXX",
    "7BV2",
)


def _rcsb_urls(ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{RCSB_DOWNLOAD_BASE}/{pdb_id}.cif.gz" for pdb_id in ids)


RCSB_SLICES: tuple[NotationSliceSpec, ...] = (
    NotationSliceSpec(
        name="rcsb_mmcif",
        urls=_rcsb_urls(RCSB_DEFAULT_IDS),
        fmt=NotationFormat.MMCIF,
        source_label="rcsb:pdb/mmCIF",
        sampling=SamplingCap(max_records=len(RCSB_DEFAULT_IDS), max_bytes=64 * 1024 * 1024),
    ),
)


def rcsb_pdb_step() -> StepSpec:
    return bio_chem_slice_step(name="raw/bio_chem/rcsb_pdb", slices=RCSB_SLICES)
