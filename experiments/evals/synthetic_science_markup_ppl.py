# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HF-backed target-only PPL probes for scientific records and markup trees.

The generated rows are supervised records with ``input`` and ``target`` fields.
Scoring uses target-only loss via :func:`supervised_text_dataset`, matching the
base-model prompt style used by the numeric-format PPL probes.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import random
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, supervised_text_dataset
from marin.processing.tokenize import HfDatasetSpec

EPIC_5005 = 5005
SCIENCE_MARKUP_ISSUE = 5618
SCIENCE_MARKUP_HF_DATASET_ID = "marin-community/synth-science-markup-tree-ppl"
SCIENCE_MARKUP_SOURCE = "generated_science_markup_tree_ppl_v1"
SCIENCE_MARKUP_HF_REVISION = "ff4f38e047c8950725566a22488cbe567e275d2c"
SCIENCE_MARKUP_SEED = 5615
EXAMPLES_PER_CONFIG = 1000


class ScienceMarkupFamily(StrEnum):
    SCIENTIFIC_RECORDS = "scientific_records"
    MARKUP_BIBLIOGRAPHY = "markup_bibliography"
    TREE_CLOSURE = "tree_closure"


@dataclass(frozen=True)
class ScienceMarkupPplSlice:
    family: ScienceMarkupFamily
    task_name: str
    hf_config_name: str

    @property
    def registry_key(self) -> str:
        return posixpath.join("synthetic_science_markup_ppl", self.family.value, self.task_name)

    @property
    def tags(self) -> tuple[str, ...]:
        return (
            "synthetic_science_markup_ppl",
            f"epic:{EPIC_5005}",
            f"issue:{SCIENCE_MARKUP_ISSUE}",
            f"family:{self.family.value}",
            f"task:{self.task_name}",
            f"seed:{SCIENCE_MARKUP_SEED}",
            f"examples:{EXAMPLES_PER_CONFIG}",
            f"source:{SCIENCE_MARKUP_SOURCE}",
            f"hf_revision:{SCIENCE_MARKUP_HF_REVISION}",
            "loss:target_only",
        )

    def to_raw_text_dataset(self, *, hf_dataset_id: str) -> RawTextEvaluationDataset:
        return supervised_text_dataset(
            HfDatasetSpec(
                id=hf_dataset_id,
                name=self.hf_config_name,
                revision=SCIENCE_MARKUP_HF_REVISION,
            ),
            input_key="input",
            target_key="target",
            split="validation",
            tags=self.tags,
        )


def _slice(family: ScienceMarkupFamily, task_name: str) -> ScienceMarkupPplSlice:
    return ScienceMarkupPplSlice(family=family, task_name=task_name, hf_config_name=task_name)


SCIENCE_MARKUP_PPL_SLICES: tuple[ScienceMarkupPplSlice, ...] = (
    _slice(ScienceMarkupFamily.SCIENTIFIC_RECORDS, "smiles_formula_completion"),
    _slice(ScienceMarkupFamily.SCIENTIFIC_RECORDS, "sdf_record_closure"),
    _slice(ScienceMarkupFamily.SCIENTIFIC_RECORDS, "pdb_atom_records"),
    _slice(ScienceMarkupFamily.SCIENTIFIC_RECORDS, "mmcif_loop_completion"),
    _slice(ScienceMarkupFamily.SCIENTIFIC_RECORDS, "fasta_sequence_record"),
    _slice(ScienceMarkupFamily.SCIENTIFIC_RECORDS, "genbank_feature_record"),
    _slice(ScienceMarkupFamily.MARKUP_BIBLIOGRAPHY, "bibtex_entry_completion"),
    _slice(ScienceMarkupFamily.MARKUP_BIBLIOGRAPHY, "latex_table_rows"),
    _slice(ScienceMarkupFamily.MARKUP_BIBLIOGRAPHY, "latex_reference_closure"),
    _slice(ScienceMarkupFamily.TREE_CLOSURE, "xml_tag_closure"),
    _slice(ScienceMarkupFamily.TREE_CLOSURE, "html_entity_attribute_closure"),
    _slice(ScienceMarkupFamily.TREE_CLOSURE, "svg_group_path_closure"),
    _slice(ScienceMarkupFamily.TREE_CLOSURE, "mathml_nested_tree"),
    _slice(ScienceMarkupFamily.TREE_CLOSURE, "nested_attribute_tree"),
    _slice(ScienceMarkupFamily.TREE_CLOSURE, "entity_escaping"),
)


def science_markup_raw_validation_sets(
    *,
    hf_dataset_id: str = SCIENCE_MARKUP_HF_DATASET_ID,
) -> dict[str, RawTextEvaluationDataset]:
    return {
        slice_.registry_key: slice_.to_raw_text_dataset(hf_dataset_id=hf_dataset_id)
        for slice_ in SCIENCE_MARKUP_PPL_SLICES
    }


def _row(
    *,
    row_id: str,
    subset: str,
    task: str,
    seed: int,
    input_text: str,
    target: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "input": input_text,
        "target": target,
        "id": row_id,
        "subset": subset,
        "task": task,
        "seed": seed,
        "metadata": metadata,
    }


def _escape_xml_text(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_xml_attr(value: str) -> str:
    return _escape_xml_text(value).replace('"', "&quot;")


def _sample_science_record(task: str, rng: random.Random, row_id: str, seed: int) -> dict[str, Any]:
    compound = rng.choice(
        [
            ("ethanol", "CCO", "C2H6O", 46.069, ("C", "O")),
            ("glycine", "NCC(=O)O", "C2H5NO2", 75.067, ("N", "C", "O")),
            ("benzene", "c1ccccc1", "C6H6", 78.114, ("C",)),
            ("alanine", "CC(N)C(=O)O", "C3H7NO2", 89.094, ("C", "N", "O")),
        ]
    )
    name, smiles, formula, mass, atoms = compound
    chain = rng.choice(["A", "B", "C"])
    residue = rng.choice(["ALA", "GLY", "SER", "LYS"])
    accession = f"SCM{rng.randrange(100000, 999999)}"

    if task == "smiles_formula_completion":
        input_text = f"Complete the molecular record.\nname: {name}\nsmiles: {smiles}\n"
        target = f"formula: {formula}\nmonoisotopic_mass: {mass:.3f}\n"
        metadata = {"format": "smiles", "compound": name}
    elif task == "sdf_record_closure":
        atom_lines = "\n".join(
            f"{idx * 1.25:10.4f}{(idx % 2) * 0.85:10.4f}{0.0:10.4f} {atom:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
            for idx, atom in enumerate(atoms, start=1)
        )
        input_text = (
            f"{name}\n  MarinSynthetic\n\n"
            f"{len(atoms):>3}{max(len(atoms) - 1, 0):>3}  0  0  0  0            999 V2000\n"
            f"{atom_lines}\n"
        )
        target = "M  END\n>  <SMILES>\n" f"{smiles}\n\n$$$$\n"
        metadata = {"format": "sdf", "compound": name, "atoms": len(atoms)}
    elif task == "pdb_atom_records":
        input_text = f"HEADER    SYNTHETIC PEPTIDE\nTITLE     {residue} CHAIN {chain} COORDINATES\n"
        target = (
            f"ATOM      1  N   {residue} {chain}   1      11.104  13.207   2.100  1.00 20.00           N\n"
            f"ATOM      2  CA  {residue} {chain}   1      12.410  13.660   2.520  1.00 20.00           C\n"
            f"TER\nEND\n"
        )
        metadata = {"format": "pdb", "chain": chain, "residue": residue}
    elif task == "mmcif_loop_completion":
        input_text = (
            f"data_{accession.lower()}\n#\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
            "_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.Cartn_x\n"
            "_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        )
        target = (
            f"ATOM 1 N N {residue} {chain} 11.104 13.207 2.100\n"
            f"ATOM 2 C CA {residue} {chain} 12.410 13.660 2.520\n#\n"
        )
        metadata = {"format": "mmcif", "chain": chain, "residue": residue}
    elif task == "fasta_sequence_record":
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        seq = "".join(rng.choice(alphabet) for _ in range(24))
        input_text = f">{accession}|synthetic peptide|chain={chain}\n"
        target = f"{seq[:12]}\n{seq[12:]}\n"
        metadata = {"format": "fasta", "length": len(seq), "chain": chain}
    elif task == "genbank_feature_record":
        gene = rng.choice(["recA", "dnaK", "rpoB", "gyrA"])
        input_text = (
            f"LOCUS       {accession:<12} 96 bp    DNA     linear   SYN 01-JAN-2026\n"
            "FEATURES             Location/Qualifiers\n"
        )
        target = (
            f"     gene            1..96\n"
            f'                     /gene="{gene}"\n'
            f"     CDS             1..96\n"
            f'                     /product="synthetic {gene} fragment"\n'
            "ORIGIN\n        1 atggctgctg aacgtctgaa cctgactgaa\n//\n"
        )
        metadata = {"format": "genbank", "accession": accession, "gene": gene}
    else:
        raise ValueError(f"unknown scientific record task: {task}")

    return _row(
        row_id=row_id,
        subset=ScienceMarkupFamily.SCIENTIFIC_RECORDS.value,
        task=task,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata=metadata,
    )


def _sample_markup_bibliography(task: str, rng: random.Random, row_id: str, seed: int) -> dict[str, Any]:
    key = rng.choice(["knuth1984texbook", "lamport1994latex", "witten2017attention", "felsenstein1985phylogenies"])
    author = rng.choice(["Donald E. Knuth", "Leslie Lamport", "Ashish Vaswani and Noam Shazeer", "Joseph Felsenstein"])
    title = rng.choice(["Typesetting Structured Records", "Markup Grammars in Practice", "Tree Closure and Escaping"])
    year = rng.randrange(1984, 2026)

    if task == "bibtex_entry_completion":
        input_text = f"Complete the BibTeX entry.\n@article{{{key},\n  author = {{{author}}},\n"
        target = f"  title = {{{title}}},\n  journal = {{Journal of Synthetic Formats}},\n  year = {{{year}}}\n}}\n"
        metadata = {"format": "bibtex", "key": key}
    elif task == "latex_table_rows":
        col_a = rng.choice(["sample", "record", "parser"])
        input_text = (
            "\\begin{tabular}{llr}\nName & Format & Count \\\\\n\\hline\n"
            f"{col_a} & FASTA & {rng.randrange(10, 99)} \\\\\n"
        )
        target = (
            f"tree & XML & {rng.randrange(10, 99)} \\\\\n"
            f"chem & SDF & {rng.randrange(10, 99)} \\\\\n\\end{{tabular}}\n"
        )
        metadata = {"format": "latex_table", "columns": 3}
    elif task == "latex_reference_closure":
        label = f"sec:{rng.choice(['methods', 'results', 'appendix'])}"
        input_text = "In Section~\\ref{" f"{label}" "} we compare \\cite{"
        target = f"{key}}} against the grammar in Appendix~\\ref{{app:formats}}.\n"
        metadata = {"format": "latex_reference", "citation_key": key, "label": label}
    else:
        raise ValueError(f"unknown markup bibliography task: {task}")

    return _row(
        row_id=row_id,
        subset=ScienceMarkupFamily.MARKUP_BIBLIOGRAPHY.value,
        task=task,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata=metadata,
    )


def _sample_tree_closure(task: str, rng: random.Random, row_id: str, seed: int) -> dict[str, Any]:
    tag = rng.choice(["sample", "assay", "record", "entry"])
    escaped = rng.choice(["alpha & beta", "x < y", '"quoted" value', "A&B < C"])

    if task == "xml_tag_closure":
        input_text = f'<dataset><{tag} id="{rng.randrange(100, 999)}"><name>'
        target = f"{tag.title()} {row_id}</name><value>{rng.randrange(1, 100)}</value></{tag}></dataset>\n"
        metadata = {"format": "xml", "open_tag": tag}
    elif task == "html_entity_attribute_closure":
        input_text = f'<article data-title="{_escape_xml_attr(escaped)}"><p>'
        target = "Measured &amp; reviewed &lt;synthetic&gt; record.</p></article>\n"
        metadata = {"format": "html", "requires_entity_escaping": True}
    elif task == "svg_group_path_closure":
        color = rng.choice(["#1f77b4", "#2ca02c", "#d62728"])
        input_text = (
            f'<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">'
            f'<g id="layer-{rng.randrange(1, 5)}" fill="{color}">'
        )
        target = f'<path d="M4 4 L28 4 L{rng.randrange(12, 24)} 28 Z"/>' '<circle cx="16" cy="16" r="4"/></g></svg>\n'
        metadata = {"format": "svg", "color": color}
    elif task == "mathml_nested_tree":
        variable = rng.choice(["x", "y", "n"])
        input_text = "<math><mrow><msup><mi>" f"{variable}</mi><mn>"
        target = (
            f"{rng.randrange(2, 5)}</mn></msup><mo>+</mo>"
            f"<mfrac><mn>1</mn><mi>{variable}</mi></mfrac></mrow></math>\n"
        )
        metadata = {"format": "mathml", "variable": variable}
    elif task == "nested_attribute_tree":
        input_text = (
            f'<node kind="root" data-id="{rng.randrange(1000, 9999)}">' '<node kind="branch" data-path="a.b"><leaf '
        )
        target = f'name="{tag}" value="{rng.randrange(10, 99)}"/></node></node>\n'
        metadata = {"format": "xml_attributes", "open_tags": ["node", "node", "leaf"]}
    elif task == "entity_escaping":
        input_text = f'Escape text for XML attribute and body.\nraw: {escaped}\nxml: <field value="'
        attr = _escape_xml_attr(escaped)
        body = _escape_xml_text(escaped)
        target = f'{attr}">{body}</field>\n'
        metadata = {"format": "entity_escaping", "raw": escaped}
    else:
        raise ValueError(f"unknown tree closure task: {task}")

    return _row(
        row_id=row_id,
        subset=ScienceMarkupFamily.TREE_CLOSURE.value,
        task=task,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata=metadata,
    )


def generate_rows(slice_: ScienceMarkupPplSlice, *, examples: int = EXAMPLES_PER_CONFIG) -> Iterable[dict[str, Any]]:
    for index in range(examples):
        example_seed = SCIENCE_MARKUP_SEED + index + 10_000 * SCIENCE_MARKUP_PPL_SLICES.index(slice_)
        rng = random.Random(example_seed)
        row_id = f"{slice_.hf_config_name}-{index:06d}"
        if slice_.family == ScienceMarkupFamily.SCIENTIFIC_RECORDS:
            yield _sample_science_record(slice_.task_name, rng, row_id, example_seed)
        elif slice_.family == ScienceMarkupFamily.MARKUP_BIBLIOGRAPHY:
            yield _sample_markup_bibliography(slice_.task_name, rng, row_id, example_seed)
        elif slice_.family == ScienceMarkupFamily.TREE_CLOSURE:
            yield _sample_tree_closure(slice_.task_name, rng, row_id, example_seed)
        else:
            raise ValueError(f"unknown family: {slice_.family}")


def write_jsonl_configs(output_dir: Path, *, examples: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for slice_ in SCIENCE_MARKUP_PPL_SLICES:
        output_path = output_dir / f"{slice_.hf_config_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as out:
            for row in generate_rows(slice_, examples=examples):
                out.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    write_jsonl_configs(args.output_dir, examples=args.examples)


if __name__ == "__main__":
    main()
