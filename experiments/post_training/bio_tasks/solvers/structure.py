# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-column and CIF-loop structural reference calculations."""

import math
import shlex
from collections import defaultdict
from functools import partial
from itertools import combinations
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import fasta, table


def pdb_atoms(path: Path) -> dict[tuple[str, int, str, str], tuple[float, float, float]]:
    model = 1
    selected = {}
    for line in path.read_text().splitlines():
        if line.startswith("MODEL "):
            model = int(line[10:14])
        if not line.startswith("ATOM  ") or model != 1:
            continue
        key = (line[21], int(line[22:26]), line[26].strip(), line[12:16].strip())
        rank = (-float(line[54:60]), line[16])
        point = tuple(float(line[start : start + 8]) for start in [30, 38, 46])
        if key not in selected or rank < selected[key][0]:
            selected[key] = (rank, point)
    return {key: value[1] for key, value in selected.items()}


def dot(a: tuple, b: tuple) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def minus(a: tuple, b: tuple) -> tuple:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def cross(a: tuple, b: tuple) -> tuple:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dihedral(points: list[tuple]) -> float:
    b0, b1, b2 = minus(points[0], points[1]), minus(points[2], points[1]), minus(points[3], points[2])
    norm = math.sqrt(dot(b1, b1))
    b1 = tuple(x / norm for x in b1)
    v = minus(b0, tuple(dot(b0, b1) * x for x in b1))
    w = minus(b2, tuple(dot(b2, b1) * x for x in b1))
    angle = math.degrees(math.atan2(dot(cross(b1, v), w), dot(v, w)))
    return 180.0 if angle == -180 else angle


def solve_structure(inputs: Path, operation: str) -> list[dict]:
    answer = []
    if operation == "pdb-backbone-dihedrals":
        atoms = pdb_atoms(inputs / "backbone.pdb")
        phi = [atoms["A", residue, "", atom] for residue, atom in [(1, "C"), (2, "N"), (2, "CA"), (2, "C")]]
        psi = [atoms["A", residue, "", atom] for residue, atom in [(2, "N"), (2, "CA"), (2, "C"), (3, "N")]]
        return [{"id": "A:2", "phi": dihedral(phi), "psi": dihedral(psi)}]
    if operation.startswith("pdb"):
        atoms = {
            f"{chain}:{residue}{insertion}": (point, chain, residue)
            for (chain, residue, insertion, atom), point in pdb_atoms(inputs / "structure.pdb").items()
            if atom == "CA"
        }
        if operation == "pdb-radius-gyration":
            points = [row[0] for row in atoms.values()]
            centroid = tuple(sum(point[axis] for point in points) / len(points) for axis in range(3))
            radius = math.sqrt(sum(math.dist(point, centroid) ** 2 for point in points) / len(points))
            return [{"id": "structure", "radius": radius, "n_atoms": len(points)}]
        for a, b in combinations(sorted(atoms), 2):
            distance = math.dist(atoms[a][0], atoms[b][0])
            if operation == "pdb-ca-distances":
                answer.append({"id": a + "|" + b, "distance": distance})
            else:
                nonlocal_pair = atoms[a][1] != atoms[b][1] or abs(atoms[a][2] - atoms[b][2]) > 1
                answer.append(
                    {
                        "id": a + "|" + b,
                        "contact": int(nonlocal_pair and distance <= float((inputs / "cutoff.txt").read_text())),
                    }
                )
    elif operation == "mmcif-chain-centroids":
        fields = []
        groups = defaultdict(list)
        for line in (inputs / "structure.cif").read_text().splitlines():
            if line.startswith("_atom_site."):
                fields.append(line.split(".", 1)[1])
            elif line.startswith("ATOM "):
                row = dict(zip(fields, shlex.split(line), strict=True))
                if row["auth_atom_id"] == "CA" and row["pdbx_PDB_model_num"] == "1":
                    groups[row["auth_asym_id"]].append(tuple(float(row["Cartn_" + axis]) for axis in "xyz"))
        for chain, points in groups.items():
            answer.append(
                {
                    "id": chain,
                    **{axis: sum(point[i] for point in points) / len(points) for i, axis in enumerate("xyz")},
                    "n": len(points),
                }
            )
    elif operation == "peptide-target-decoy-fdr":
        rows = sorted(table(inputs / "psms.csv"), key=lambda row: float(row["score"]), reverse=True)
        targets = decoys = 0
        raw = []
        for row in rows:
            targets += row["type"] == "target"
            decoys += row["type"] == "decoy"
            raw.append(min(1, decoys / targets) if targets else 1.0)
        qvalues = [min(raw[i:]) for i in range(len(raw))]
        for row, q in zip(rows, qvalues, strict=True):
            answer.append({"id": row["psm"], "qvalue": q, "accept": int(row["type"] == "target" and q <= 0.25)})
    else:
        assignments = table(inputs / "peptides.csv")
        for name, sequence in fasta(inputs / "proteins.fa").items():
            covered = set()
            for row in assignments:
                if row["protein"] != name:
                    continue
                start = sequence.find(row["peptide"])
                while start >= 0:
                    covered.update(range(start, start + len(row["peptide"])))
                    start = sequence.find(row["peptide"], start + 1)
            answer.append(
                {"id": name, "covered": len(covered), "length": len(sequence), "fraction": len(covered) / len(sequence)}
            )
    return answer


NAMES = (
    "pdb-ca-distances",
    "mmcif-chain-centroids",
    "pdb-contact-map",
    "pdb-backbone-dihedrals",
    "pdb-radius-gyration",
    "peptide-target-decoy-fdr",
    "protein-coverage",
)
SOLVERS = {name: partial(solve_structure, operation=name) for name in NAMES}
