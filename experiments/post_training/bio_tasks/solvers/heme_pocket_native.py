# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native Biopython heme-pocket distances and sampled solvent accessibility."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import Atom, Chain, Model, NeighborSearch, Residue, Structure
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.SASA import ShrakeRupley


def cif_atoms(path: Path) -> list[dict]:
    """Resolve author residue IDs and atom-wise alternate conformers from mmCIF."""
    data = MMCIF2Dict(str(path))
    fields = {key.removeprefix("_atom_site."): values for key, values in data.items() if key.startswith("_atom_site.")}
    selected = {}
    for row in zip(*fields.values(), strict=True):
        atom = dict(zip(fields, row, strict=True))
        protein = atom["group_PDB"] == "ATOM"
        if atom["pdbx_PDB_model_num"] != "1" or (not protein and atom["auth_comp_id"] != "HEM"):
            continue
        if atom["type_symbol"].upper() in {"H", "D"} or float(atom["occupancy"]) <= 0:
            continue
        chain = atom["auth_asym_id"]
        sequence = int(atom["auth_seq_id"])
        insertion = "" if atom["pdbx_PDB_ins_code"] in {".", "?"} else atom["pdbx_PDB_ins_code"]
        alternate = "" if atom["label_alt_id"] in {".", "?"} else atom["label_alt_id"]
        kind = "protein" if protein else "HEM"
        residue = f"{kind}:{chain}:{sequence}:{insertion or '.'}:{atom['auth_comp_id']}"
        identifier = f"{residue}:{atom['auth_atom_id']}"
        record = {
            "id": identifier,
            "residue": residue,
            "chain": chain,
            "sequence": sequence,
            "insertion": insertion,
            "name": atom["auth_comp_id"],
            "atom": atom["auth_atom_id"],
            "element": atom["type_symbol"].upper(),
            "protein": protein,
            "coordinate": np.array([float(atom[f"Cartn_{axis}"]) for axis in "xyz"], dtype=np.float64),
            "rank": (-float(atom["occupancy"]), alternate),
        }
        if identifier not in selected or record["rank"] < selected[identifier]["rank"]:
            selected[identifier] = record
    if not selected:
        raise ValueError("No eligible atoms in the supplied structure")
    return [selected[key] for key in sorted(selected)]


def bio_structure(records: list[dict]) -> tuple[Structure.Structure, dict[str, Atom.Atom]]:
    structure = Structure.Structure("selected")
    model = Model.Model(0)
    structure.add(model)
    chains, residues, atoms = {}, {}, {}
    for serial, record in enumerate(records, 1):
        chain_id = record["chain"]
        if chain_id not in chains:
            chains[chain_id] = Chain.Chain(chain_id)
            model.add(chains[chain_id])
        residue_id = record["residue"]
        if residue_id not in residues:
            hetero = " " if record["protein"] else "H_HEM"
            key = (hetero, record["sequence"], record["insertion"] or " ")
            residues[residue_id] = Residue.Residue(key, record["name"], "")
            chains[chain_id].add(residues[residue_id])
        atom = Atom.Atom(
            record["atom"], record["coordinate"].copy(), 0.0, 1.0, " ", record["atom"], serial, record["element"]
        )
        atom.xtra["identity"] = record["id"]
        residues[residue_id].add(atom)
        atoms[record["id"]] = atom
    return structure, atoms


def native_surface(records: list[dict], query: dict) -> dict[str, float]:
    if any(record["element"] not in query["radii"] for record in records):
        raise ValueError("An atom element has no declared solvent-accessibility radius")
    structure, atoms = bio_structure(records)
    ShrakeRupley(
        probe_radius=query["probe_radius"], n_points=query["surface_points"], radii_dict=query["radii"]
    ).compute(structure, level="A")
    return {identifier: float(atom.sasa) for identifier, atom in atoms.items()}


def pocket_artifacts(inputs: Path) -> dict:
    query = json.loads((inputs / "query.json").read_text())
    records = cif_atoms(inputs / "structure.cif")
    proteins = [record for record in records if record["protein"]]
    ligands = defaultdict(list)
    residues = defaultdict(list)
    for record in records:
        (residues if record["protein"] else ligands)[record["residue"]].append(record)
    if len(ligands) != query["expected_hemes"] or len(residues) != query["expected_protein_residues"]:
        raise ValueError("Structure selection differs from the frozen study")
    _, protein_atoms = bio_structure(proteins)
    search = NeighborSearch(list(protein_atoms.values()))
    by_id = {record["id"]: record for record in proteins}
    protein_surface = native_surface(proteins, query)
    all_contacts, all_residues, all_ligand_atoms, summaries = {}, {}, {}, {}
    for ligand_id, ligand in sorted(ligands.items()):
        complex_surface = native_surface(proteins + ligand, query)
        free_surface = native_surface(ligand, query)
        contact_counts = defaultdict(int)
        chain_contacts = set()
        for ligand_atom in ligand:
            for protein_atom in search.search(ligand_atom["coordinate"], query["contact_cutoff"], level="A"):
                protein_id = protein_atom.xtra["identity"]
                protein_record = by_id[protein_id]
                contact_id = f"{ligand_atom['id']}|{protein_id}"
                all_contacts[contact_id] = {
                    "heme": ligand_id,
                    "heme_atom": ligand_atom["atom"],
                    "protein_atom": protein_id,
                    "protein_residue": protein_record["residue"],
                    "distance": math.dist(ligand_atom["coordinate"], protein_record["coordinate"]),
                }
                contact_counts[protein_record["residue"]] += 1
                chain_contacts.add(protein_record["chain"])
            nearest = min(
                (math.dist(ligand_atom["coordinate"], record["coordinate"]), record["id"]) for record in proteins
            )
            all_ligand_atoms[ligand_atom["id"]] = {
                "heme": ligand_id,
                "element": ligand_atom["element"],
                "nearest_protein_atom": nearest[1],
                "minimum_distance": nearest[0],
                "sasa_free": free_surface[ligand_atom["id"]],
                "sasa_complex": complex_surface[ligand_atom["id"]],
                "buried_area": free_surface[ligand_atom["id"]] - complex_surface[ligand_atom["id"]],
            }
        for residue_id, atoms in sorted(residues.items()):
            nearest = min((math.dist(a["coordinate"], b["coordinate"]), a["id"], b["id"]) for a in atoms for b in ligand)
            baseline = math.fsum(protein_surface[atom["id"]] for atom in atoms)
            bound = math.fsum(complex_surface[atom["id"]] for atom in atoms)
            all_residues[f"{ligand_id}|{residue_id}"] = {
                "heme": ligand_id,
                "residue": residue_id,
                "chain": atoms[0]["chain"],
                "residue_name": atoms[0]["name"],
                "nearest_protein_atom": nearest[1],
                "nearest_heme_atom": nearest[2],
                "minimum_distance": nearest[0],
                "contact_pairs": contact_counts[residue_id],
                "in_pocket": int(nearest[0] <= query["contact_cutoff"]),
                "sasa_without_heme": baseline,
                "sasa_with_heme": bound,
                "buried_area": baseline - bound,
            }
        irons = [atom for atom in ligand if atom["element"] == "FE"]
        if len(irons) != 1:
            raise ValueError("Every HEM must contain exactly one iron atom")
        iron = all_ligand_atoms[irons[0]["id"]]
        protein_burial = math.fsum(protein_surface[a["id"]] - complex_surface[a["id"]] for a in proteins)
        ligand_burial = math.fsum(free_surface[a["id"]] - complex_surface[a["id"]] for a in ligand)
        summaries[ligand_id] = {
            "protein_residues": len(residues),
            "pocket_residues": sum(value > 0 for value in contact_counts.values()),
            "contact_pairs": sum(contact_counts.values()),
            "contacting_chains": len(chain_contacts),
            "heme_heavy_atoms": len(ligand),
            "protein_buried_area": protein_burial,
            "heme_buried_area": ligand_burial,
            "total_buried_area": protein_burial + ligand_burial,
            "nearest_iron_protein_atom": iron["nearest_protein_atom"],
            "iron_minimum_distance": iron["minimum_distance"],
        }
    return {"summaries": summaries, "contacts": all_contacts, "residues": all_residues, "heme_atoms": all_ligand_atoms}


def write_artifacts(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, field in (("contacts.tsv", "contacts"), ("residues.tsv", "residues"), ("heme_atoms.tsv", "heme_atoms")):
        rows = [{"id": key, **value} for key, value in sorted(result[field].items())]
        if not rows:
            raise ValueError("A required artifact has no rows")
        with (output / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    answer = [{"id": key, **value} for key, value in sorted(result["summaries"].items())]
    (output / "answer.json").write_text(json.dumps(answer, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_artifacts(pocket_artifacts(args.inputs), args.output)


if __name__ == "__main__":
    main()
