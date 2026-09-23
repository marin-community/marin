# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structural geometry and peptide/protein evidence recipes."""

import math
import random
from functools import partial
from itertools import combinations

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Difficulty, Instance, Recipe, csv_text


def pdb_atom(
    serial: int,
    atom: str,
    chain: str,
    residue: int,
    insertion: str,
    xyz: tuple,
    altloc: str = " ",
    occupancy: float = 1.0,
) -> str:
    x, y, z = xyz
    element = atom[0]
    return (
        f"ATOM  {serial:5d} {atom:^4s}{altloc}ALA {chain}{residue:4d}{insertion:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{20:6.2f}          {element:>2s}  \n"
    )


def generate_structure(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    expected, inputs = {}, {}
    if operation in {"pdb-ca-distances", "pdb-contact-map", "pdb-radius-gyration"}:
        scale = rng.randint(1, 3)
        offset = rng.randint(-8, 8)
        positions = [(0, 0, 0), (3 * scale, 0, 0), (3 * scale, 4 * scale, 0), (0, 0, 12 * scale)]
        identifiers = ["A:1", "A:2A", "A:3", "B:1"]
        coordinates = [(x + offset, y + offset, z + offset) for x, y, z in positions]
        lines = ["MODEL        1\n"]
        for i, point in enumerate(coordinates):
            chain = "A" if i < 3 else "B"
            residue = i + 1 if i < 3 else 1
            insertion = "A" if i == 1 else " "
            lines.append(
                pdb_atom(
                    i * 3 + 1, "CA", chain, residue, insertion, point, "A" if i == 1 else " ", 0.6 if i == 1 else 1.0
                )
            )
            lines.append(pdb_atom(i * 3 + 2, "N", chain, residue, insertion, (point[0] + 1, point[1], point[2])))
            if i == 1:
                lines.append(pdb_atom(i * 3 + 3, "CA", chain, residue, insertion, (99, 99, 99), "B", 0.4))
        lines.extend(
            ["ENDMDL\n", "MODEL        2\n", pdb_atom(20, "CA", "A", 1, " ", (99, 99, 99)), "ENDMDL\n", "END\n"]
        )
        inputs = {"structure.pdb": "".join(lines)}
        distances = [3, 5, 12, 4, math.sqrt(153), 13]
        if operation == "pdb-radius-gyration":
            expected = {"structure": {"radius": math.sqrt(32.25) * scale, "n_atoms": 4}}
            columns = {
                "radius": Column(
                    kind="number",
                    unit="angstrom",
                    description="equal-weight CA radius of gyration about centroid",
                    atol=1e-8,
                    rtol=1e-8,
                ),
                "n_atoms": Column(kind="integer", unit="CA atoms", description="selected atoms"),
            }
            prompt = (
                "Compute the equal-weight CA radius of gyration sqrt(mean squared distance to the CA "
                "centroid)) for structure.pdb, with id=structure."
            )
            wrong = [{"id": "structure", "radius": math.sqrt(32.25) * scale * 10, "n_atoms": 4}]
            reason = "converted_angstrom_to_wrong_scale"
        else:
            for (i, j), distance in zip(combinations(range(4), 2), distances, strict=True):
                if operation == "pdb-ca-distances":
                    expected[identifiers[i] + "|" + identifiers[j]] = {"distance": distance * scale}
                else:
                    expected[identifiers[i] + "|" + identifiers[j]] = {"contact": int(i == 0 and j == 2)}
            if operation == "pdb-ca-distances":
                columns = {
                    "distance": Column(
                        kind="number", unit="angstrom", description="Euclidean CA distance", atol=1e-8, rtol=1e-8
                    )
                }
                prompt = (
                    "Compute all unordered residue-pair CA distances in structure.pdb. Residue IDs are "
                    "chain:integer_resSeq followed immediately by nonblank insertion code. Use lexically "
                    "ordered residue1|residue2 IDs."
                )
                wrong = [{"id": k, "distance": v["distance"] / 10} for k, v in expected.items()]
                reason = "reported_nanometers_as_angstroms"
            else:
                inputs["cutoff.txt"] = str(5 * scale) + "\n"
                columns = {
                    "contact": Column(
                        kind="integer", unit="decision", description="1 if nonlocal CA pair within inclusive cutoff"
                    )
                }
                prompt = (
                    "Build a nonlocal CA contact map for every unordered residue pair in structure.pdb. A "
                    "contact requires distance <= cutoff.txt angstroms and either different chains or "
                    "abs(resSeq difference)>1. Use resSeq numeric difference for the local-neighbor exclusion "
                    "(insertion code does not change it). Return 0 or 1 for every pair, with lexically ordered "
                    "chain:resSeqInsertion|chain:resSeqInsertion IDs."
                )
                wrong = [{"id": k, "contact": 0} for k in expected]
                reason = "exclusive_contact_distance_cutoff"
        prompt += (
            " Select only MODEL 1 ATOM records named CA. For alternate locations of the same "
            "chain/resSeq/insertion/atom, choose highest occupancy, then lexically smallest altloc. "
            "Include all chains; do not count N atoms or other models."
        )
    elif operation == "pdb-backbone-dihedrals":
        sign = rng.choice([-1, 1])
        shift = rng.randint(-4, 4)
        atoms = [
            (1, "C", (0, 1, 0)),
            (2, "N", (0, 0, 0)),
            (2, "CA", (1, 0, 0)),
            (2, "C", (1, 0, sign)),
            (3, "N", (2, 0, sign)),
        ]
        inputs = {
            "backbone.pdb": (
                "".join(
                    pdb_atom(i + 1, atom, "A", residue, " ", tuple(value + shift for value in point))
                    for i, (residue, atom, point) in enumerate(atoms)
                )
                + "END\n"
            )
        }
        expected = {"A:2": {"phi": 90.0 * sign, "psi": 180.0}}
        columns = {
            name: Column(kind="number", unit="degrees", description=description, atol=1e-8, rtol=1e-8)
            for name, description in [
                ("phi", "C(previous),N,CA,C signed dihedral"),
                ("psi", "N,CA,C,N(next) signed dihedral"),
            ]
        }
        prompt = (
            "Compute phi and psi for residue A:2 in backbone.pdb using phi=(C1,N2,CA2,C2), "
            "psi=(N2,CA2,C2,N3). For four points p0..p3, use b0=p0-p1, unit b1=(p2-p1)/norm, b2=p3-p2; "
            "project b0,b2 perpendicular to b1 as v,w; angle=atan2(dot(cross(b1,v),w),dot(v,w)). Report"
            " degrees in (-180,180], mapping -180 to 180. Use id=A:2."
        )
        wrong = [{"id": "A:2", "phi": -90.0 * sign, "psi": 180.0}]
        reason = "reversed_dihedral_handedness"
    elif operation == "mmcif-chain-centroids":
        shift = rng.randint(-5, 5)
        headers = [
            "group_PDB",
            "id",
            "type_symbol",
            "label_atom_id",
            "label_alt_id",
            "label_comp_id",
            "label_asym_id",
            "label_entity_id",
            "label_seq_id",
            "pdbx_PDB_ins_code",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
            "occupancy",
            "B_iso_or_equiv",
            "auth_seq_id",
            "auth_comp_id",
            "auth_asym_id",
            "auth_atom_id",
            "pdbx_PDB_model_num",
        ]
        points = [("X", "A", 1, 0, 0, 0), ("X", "A", 2, 2, 4, 6), ("Y", "B", 1, 6, 0, 0), ("Y", "B", 2, 8, 2, 4)]
        rows = []
        for i, (label, auth, residue, x, y, z) in enumerate(points, 1):
            rows.append(
                f"ATOM {i} C CA . ALA {label} 1 {residue} ? {x+shift} {y+shift} {z+shift} "
                f"1.0 20 {residue} ALA {auth} CA 1"
            )
        rows.append("ATOM 5 C CA . ALA X 1 1 ? 99 99 99 1.0 20 1 ALA A CA 2")
        inputs = {
            "structure.cif": (
                "data_fixture\n#\nloop_\n"
                + "".join("_atom_site." + header + "\n" for header in headers)
                + "\n".join(rows)
                + "\n#\n"
            )
        }
        expected = {
            "A": {"x": 1.0 + shift, "y": 2.0 + shift, "z": 3.0 + shift, "n": 2},
            "B": {"x": 7.0 + shift, "y": 1.0 + shift, "z": 2.0 + shift, "n": 2},
        }
        columns = {
            axis: Column(kind="number", unit="angstrom", description="CA centroid " + axis, atol=1e-8, rtol=1e-8)
            for axis in "xyz"
        }
        columns["n"] = Column(kind="integer", unit="CA atoms", description="selected atoms per author chain")
        prompt = (
            "Compute unweighted CA centroids per author chain in structure.cif. Read the atom_site loop"
            " by column names; select ATOM, auth_atom_id=CA, and pdbx_PDB_model_num=1. Group by "
            "auth_asym_id, not label_asym_id; report Cartesian x,y,z means and n atoms. Use author "
            "chain id. This file uses whitespace-separated CIF values without multiline text."
        )
        wrong = [{"id": {"A": "X", "B": "Y"}[k], **v} for k, v in expected.items()]
        reason = "used_label_chain_instead_of_author_chain"
    elif operation == "peptide-target-decoy-fdr":
        shift = rng.randint(0, 20)
        types = ["target", "target", "decoy", "target", "target", "decoy", "target", "target"]
        rows = [{"psm": f"p{i}", "score": 100 - i + shift, "type": kind} for i, kind in enumerate(types)]
        expected_values = [0, 0, 1 / 4, 1 / 4, 1 / 4, 1 / 3, 1 / 3, 1 / 3]
        expected = {
            f"p{i}": {
                "qvalue": float(expected_values[i]),
                "accept": int(types[i] == "target" and expected_values[i] <= 0.25),
            }
            for i in range(8)
        }
        rng.shuffle(rows)
        inputs = {"psms.csv": csv_text(rows)}
        columns = {
            "qvalue": Column(
                kind="number",
                unit="fraction",
                description="monotone minimum of cumulative decoy/target FDR at lower score thresholds",
                atol=1e-10,
                rtol=1e-8,
            ),
            "accept": Column(kind="integer", unit="decision", description="target and q <=0.25"),
        }
        prompt = (
            "Estimate PSM q-values in psms.csv by score-descending target-decoy competition results. "
            "These are already competed PSMs with unique scores. At each threshold FDR=min(1, "
            "cumulative_decoys/cumulative_targets); define q at a score as the minimum FDR over "
            "thresholds at that score or lower. No pseudocount. Report q and accept=1 only for target "
            "PSMs with q<=0.25, for every psm id."
        )
        wrong = [{"id": k, **v, "accept": 1} for k, v in expected.items()]
        reason = "accepted_decoy_psms"
    else:
        assert operation == "protein-coverage"
        n = rng.randint(2, 5)
        sequence = "A" * n + "PEPTIDE" + "GG" + "PEPTIDE" + "K"
        inputs = {
            "proteins.fa": ">p0\n" + sequence + "\n>p1\nMMMMMM\n",
            "peptides.csv": csv_text(
                [
                    {"protein": "p0", "peptide": "PEPTIDE"},
                    {"protein": "p0", "peptide": "TIDEGG"},
                    {"protein": "p0", "peptide": "PEPTIDE"},
                ]
            ),
        }
        expected = {
            "p0": {"covered": 16, "length": len(sequence), "fraction": 16 / len(sequence)},
            "p1": {"covered": 0, "length": 6, "fraction": 0.0},
        }
        columns = {
            "covered": Column(kind="integer", unit="residues", description="union of all exact peptide-match positions"),
            "length": Column(kind="integer", unit="residues", description="protein length"),
            "fraction": Column(
                kind="number", unit="fraction", description="covered / protein length", atol=1e-10, rtol=1e-8
            ),
        }
        prompt = (
            "Compute residue coverage for every proteins.fa entry from its assigned peptides.csv "
            "peptides. Match exact sequence, include every occurrence of repeated peptides, and take "
            "the union of covered positions. Duplicate peptide rows do not add coverage. Return all "
            "proteins, including those with no evidence, using protein id."
        )
        wrong = [{"id": k, **v, "covered": 7 if k == "p0" else 0} for k, v in expected.items()]
        reason = "used_only_first_peptide_occurrence"
    return Instance(
        prompt + " Inputs are in /app/inputs.", inputs, Contract(columns=columns, expected=expected), {reason: wrong}
    )


SKILLS = {
    "pdb-ca-distances": ("fixed-width-atoms", "alternate-locations", "insertion-codes"),
    "mmcif-chain-centroids": ("atom-site-loop", "author-versus-label-ids", "models"),
    "pdb-contact-map": ("residue-contacts", "local-neighbor-exclusion", "distance-units"),
    "pdb-backbone-dihedrals": ("backbone-geometry", "handedness", "angle-conventions"),
    "pdb-radius-gyration": ("centering", "structural-size", "atom-selection"),
    "peptide-target-decoy-fdr": ("target-decoy", "monotone-qvalues", "competition"),
    "protein-coverage": ("peptide-mapping", "repeated-sequences", "union-coverage"),
}
RECIPES = tuple(
    Recipe(
        name,
        "1",
        Difficulty.MEDIUM,
        skills,
        (
            ("pdb3.3-atom-profile",)
            if name.startswith("pdb")
            else (
                ("mmcif-atom-site",)
                if name.startswith("mmcif")
                else ("fasta", "csv-header") if name == "protein-coverage" else ("csv-header",)
            )
        ),
        (
            ("https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html",)
            if name.startswith("pdb")
            else (
                ("https://mmcif.wwpdb.org/docs/user-guide/guide.html",)
                if name.startswith("mmcif")
                else ("https://pyteomics.readthedocs.io/en/latest/",)
            )
        ),
        partial(generate_structure, operation=name),
    )
    for name, skills in SKILLS.items()
)
