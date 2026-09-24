# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental mmCIF tasks with references computed from the paired PDB deposit."""

import json
import random
from functools import cache, partial

import numpy as np

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, Recipe

STRUCTURES = ("1UBQ", "1CRN", "4HHB")


@cache
def deposited_coordinates(accession: str) -> dict[tuple[str, int, str], tuple[float, float, float]]:
    """Read alpha carbons from the authoring-only PDB representation."""
    atoms = {}
    model = 1
    for line in source_text(f"PDB:{accession}", f"{accession.lower()}.pdb.gz").splitlines():
        if line.startswith("MODEL "):
            model = int(line[10:14])
        if model != 1 or not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        key = (line[21], int(line[22:26]), line[26].strip())
        rank = (-float(line[54:60]), line[16].strip())
        coordinates = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
        if key not in atoms or rank < atoms[key][0]:
            atoms[key] = (rank, coordinates)
    return {key: value[1] for key, value in atoms.items()}


def generate_real_structure(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    accession = rng.choice(STRUCTURES)
    atoms = deposited_coordinates(accession)
    keys = list(atoms)
    coordinates = np.array(list(atoms.values()))
    inputs = {"structure.cif": source_text(f"PDB:{accession}", f"{accession.lower()}.cif.gz")}
    prompt = (
        f"Analyze the unchanged experimental structure {accession} in /app/inputs/structure.cif. "
        "Use only model 1, ATOM records, and protein alpha carbons (auth_atom_id CA). "
        "Identify residues by auth_asym_id, auth_seq_id, and pdbx_PDB_ins_code; '.' and '?' mean no insertion code. "
        "For alternate conformers, retain highest occupancy; ties use lexicographically smallest label_alt_id "
        "with '.' or '?' treated as empty. Coordinates and distances are in angstroms. "
    )
    expected = {}
    if operation.endswith("chain-geometry"):
        for chain in sorted({key[0] for key in keys}):
            points = coordinates[[i for i, key in enumerate(keys) if key[0] == chain]]
            center = points.mean(axis=0)
            radius = np.sqrt(((points - center) ** 2).sum(axis=1).mean())
            expected[chain] = {
                **{axis: float(center[j]) for j, axis in enumerate("xyz")},
                "radius": float(radius),
                "n_residues": len(points),
            }
        columns = {
            axis: Column(
                kind="number", description=f"CA centroid {axis} coordinate", unit="angstrom", atol=1e-7, rtol=1e-9
            )
            for axis in "xyz"
        }
        columns["radius"] = Column(
            kind="number", description="unweighted CA radius of gyration", unit="angstrom", atol=1e-7, rtol=1e-9
        )
        columns["n_residues"] = Column(kind="integer", description="selected alpha carbons", unit="residues")
        prompt += (
            "For each author chain, report its CA centroid, unweighted radius of gyration "
            "about that centroid, and residue count."
        )
        wrong = [{"id": key, **value, "radius": value["radius"] ** 2} for key, value in expected.items()]
        reason = "mean_squared_distance_without_square_root"
    else:
        cutoff = rng.choice((6, 8, 10))
        inputs["query.json"] = json.dumps({"cutoff": cutoff}) + "\n"
        distances = ((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(axis=2)
        eligible = np.array([[a[0] != b[0] or abs(a[1] - b[1]) > 1 for b in keys] for a in keys])
        contacts = ((distances <= cutoff**2) & eligible).sum(axis=1)
        expected = {
            f"{chain}:{residue}{insertion}": {"contacts": int(contacts[i])}
            for i, (chain, residue, insertion) in enumerate(keys)
        }
        columns = {
            "contacts": Column(kind="integer", description="nonadjacent CA neighbors within cutoff", unit="residues")
        }
        prompt += (
            "For every selected residue, report its number of other CA atoms within the inclusive cutoff in query.json. "
            "Exclude pairs from the same author chain with absolute auth_seq_id difference <=1, including self pairs. "
            "Include inter-chain contacts. IDs must be 'chain:residue' with the insertion code appended when present."
        )
        wrong_counts = (distances <= cutoff**2).sum(axis=1)
        wrong = [{"id": key, "contacts": int(value)} for key, value in zip(expected, wrong_counts, strict=True)]
        reason = "included_self_and_sequence_neighbors"
    return Instance(
        prompt,
        inputs,
        Contract(columns=columns, expected=expected),
        {reason: wrong},
        data_origin=DataOrigin.REAL,
        source_ids=(f"PDB:{accession}",),
        derivation=(
            "Full deposited mmCIF supplied unchanged. References use the independently parsed paired "
            "PDB representation; model, atom and conformer selection is explicit."
        ),
    )


RECIPES = tuple(
    Recipe(
        name,
        "1",
        ("experimental-structures", "author-residue-identifiers", "alternate-conformers", "coordinate-geometry"),
        ("mmCIF",),
        tuple(f"https://www.rcsb.org/structure/{code}" for code in STRUCTURES),
        partial(generate_real_structure, operation=name),
    )
    for name in ("real-mmcif-chain-geometry", "real-mmcif-contact-degree")
)
