# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read deposited mmCIF atom rows and compute geometry without array libraries."""

import json
import math
import shlex
from collections import defaultdict
from functools import partial
from itertools import combinations
from pathlib import Path


def solve_real_structure(inputs: Path, operation: str) -> list[dict]:
    fields = []
    atoms = {}
    for line in (inputs / "structure.cif").read_text().splitlines():
        if line.startswith("_atom_site."):
            fields.append(line.split(".", 1)[1].strip())
        elif line.startswith("ATOM "):
            row = dict(zip(fields, shlex.split(line), strict=True))
            if row["pdbx_PDB_model_num"] != "1" or row["auth_atom_id"] != "CA":
                continue
            insertion = row["pdbx_PDB_ins_code"]
            alternate = row["label_alt_id"]
            key = (row["auth_asym_id"], int(row["auth_seq_id"]), "" if insertion in (".", "?") else insertion)
            rank = (-float(row["occupancy"]), "" if alternate in (".", "?") else alternate)
            point = tuple(float(row["Cartn_" + axis]) for axis in "xyz")
            if key not in atoms or rank < atoms[key][0]:
                atoms[key] = (rank, point)
    points = {key: record[1] for key, record in atoms.items()}
    if operation.endswith("chain-geometry"):
        chains = defaultdict(list)
        for (chain, _residue, _insertion), point in points.items():
            chains[chain].append(point)
        answer = []
        for chain, members in chains.items():
            center = [math.fsum(point[j] for point in members) / len(members) for j in range(3)]
            radius = math.sqrt(math.fsum(math.dist(point, center) ** 2 for point in members) / len(members))
            answer.append(
                {"id": chain, **dict(zip("xyz", center, strict=True)), "radius": radius, "n_residues": len(members)}
            )
        return answer
    cutoff = json.loads((inputs / "query.json").read_text())["cutoff"]
    counts = dict.fromkeys(points, 0)
    for first, second in combinations(points, 2):
        if first[0] == second[0] and abs(first[1] - second[1]) <= 1:
            continue
        if math.dist(points[first], points[second]) <= cutoff:
            counts[first] += 1
            counts[second] += 1
    return [
        {"id": f"{chain}:{residue}{insertion}", "contacts": count}
        for (chain, residue, insertion), count in counts.items()
    ]


SOLVERS = {
    name: partial(solve_real_structure, operation=name)
    for name in ("real-mmcif-chain-geometry", "real-mmcif-contact-degree")
}
