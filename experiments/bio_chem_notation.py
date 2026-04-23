# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bio/chem notation PPL slices.

Wires the streaming downloaders in :mod:`marin.datakit.download.bio_chem`
through to the same ``RawTextEvaluationDataset`` / ``default_tokenize`` flow
that Paloma and Uncheatable Eval use, so the bio/chem slices land in the
existing perplexity-gap and tokenized-validation pipelines without bespoke
wiring.

Each slice in :data:`BIO_CHEM_SLICES` corresponds to one parquet shard set
written by a download step. The keys in the returned dict are
``bio_chem/<family>/<slice>`` so they namespace cleanly against
``paloma/...`` and ``uncheatable_eval/...`` in the gap report.

See issue #5058 for context.
"""

from __future__ import annotations

import os.path
from dataclasses import dataclass

from experiments.llama import llama3_tokenizer
from marin.datakit.download.bio_chem.chembl import chembl_step
from marin.datakit.download.bio_chem.moleculenet import moleculenet_step
from marin.datakit.download.bio_chem.pubchem import pubchem_step
from marin.datakit.download.bio_chem.rcsb_pdb import rcsb_pdb_step
from marin.datakit.download.bio_chem.refseq import refseq_viral_step
from marin.datakit.download.bio_chem.rnacentral import rnacentral_step
from marin.datakit.download.bio_chem.uniprot import uniprot_sprot_step
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.tokenize import TokenizeConfig
from marin.processing.tokenize.data_configs import TokenizerStep


@dataclass(frozen=True)
class BioChemSlice:
    """One eval slice: which download step it lives in and which file glob inside that step."""

    family: str
    """Source family (e.g. ``refseq``, ``uniprot``)."""

    slice_name: str
    """Slice slot — matches ``NotationSliceSpec.name`` so we can find the shard files."""

    step: ExecutorStep
    """The download step whose output dir holds the parquet shards."""


def _build_slices() -> tuple[BioChemSlice, ...]:
    """Single source of truth for which slices we evaluate.

    Each entry must reference a slice that the named download step actually
    produces — see ``NotationSliceSpec.name`` in the per-family modules under
    ``marin.datakit.download.bio_chem``.
    """
    refseq = refseq_viral_step()
    rnacentral = rnacentral_step()
    uniprot = uniprot_sprot_step()
    pubchem = pubchem_step()
    rcsb = rcsb_pdb_step()
    chembl = chembl_step()
    moleculenet = moleculenet_step()
    return (
        BioChemSlice("refseq", "refseq_viral_fasta", refseq),
        BioChemSlice("refseq", "refseq_viral_gff", refseq),
        BioChemSlice("rnacentral", "rnacentral_active_fasta", rnacentral),
        BioChemSlice("uniprot", "uniprot_sprot_fasta", uniprot),
        BioChemSlice("uniprot", "uniprot_sprot_dat", uniprot),
        BioChemSlice("pubchem", "pubchem_cid_smiles", pubchem),
        BioChemSlice("pubchem", "pubchem_sdf", pubchem),
        BioChemSlice("rcsb", "rcsb_mmcif", rcsb),
        BioChemSlice("chembl", "chembl_chemreps", chembl),
        BioChemSlice("chembl", "chembl_sdf", chembl),
        BioChemSlice("moleculenet", "moleculenet_esol_smiles", moleculenet),
        BioChemSlice("moleculenet", "moleculenet_clintox_smiles", moleculenet),
    )


BIO_CHEM_SLICES: tuple[BioChemSlice, ...] = _build_slices()


def _slice_glob(slice_: BioChemSlice) -> str:
    return f"{slice_.slice_name}-*.parquet"


def _slice_key(slice_: BioChemSlice) -> str:
    return os.path.join("bio_chem", slice_.family, slice_.slice_name)


def bio_chem_tokenized(
    *, tokenizer: str = llama3_tokenizer, slices: tuple[BioChemSlice, ...] = BIO_CHEM_SLICES
) -> dict[str, TokenizerStep]:
    """Tokenize every bio/chem slice for the regular validation-loss flow."""
    from experiments.defaults import default_tokenize

    out: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for slice_ in slices:
        key = _slice_key(slice_)
        out[key] = default_tokenize(
            name=key,
            dataset=slice_.step.cd(_slice_glob(slice_)),
            tokenizer=tokenizer,
            is_validation=True,
        )
    return out


def bio_chem_raw_validation_sets(
    slices: tuple[BioChemSlice, ...] = BIO_CHEM_SLICES,
):
    """Wire bio/chem slices into the perplexity-gap raw-text dataset registry."""
    from marin.evaluation.perplexity_gap import raw_text_dataset

    return {_slice_key(slice_): raw_text_dataset(slice_.step.cd(_slice_glob(slice_))) for slice_ in slices}


if __name__ == "__main__":
    # Materialise every download step so the slices exist on disk.
    download_steps = []
    seen: set[int] = set()
    for slice_ in BIO_CHEM_SLICES:
        if id(slice_.step) in seen:
            continue
        seen.add(id(slice_.step))
        download_steps.append(slice_.step)
    executor_main(steps=download_steps)
