# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a paired-PDB reference without using the native mmCIF/SASA oracle."""

import argparse
import gzip
import hashlib
import json
import math
import resource
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

PROTOCOL = Path(__file__).with_suffix(".json")


def pdb_atoms(content: str) -> list[dict]:
    """Read deposited decimal coordinates and keep eligible author atom identities."""
    selected = {}
    model = 1
    for line in content.splitlines():
        if line.startswith("MODEL "):
            model = int(line[10:14])
        if model != 1 or line[:6] not in {"ATOM  ", "HETATM"}:
            continue
        protein = line.startswith("ATOM  ")
        name = line[17:20].strip()
        element = line[76:78].strip().upper()
        occupancy = float(line[54:60])
        if (not protein and name != "HEM") or element in {"H", "D"} or occupancy <= 0:
            continue
        chain, sequence, insertion = line[21], int(line[22:26]), line[26].strip()
        atom_name = line[12:16].strip()
        residue = f"{'protein' if protein else 'HEM'}:{chain}:{sequence}:{insertion or '.'}:{name}"
        identity = f"{residue}:{atom_name}"
        record = {
            "id": identity,
            "residue": residue,
            "chain": chain,
            "name": name,
            "atom": atom_name,
            "protein": protein,
            "element": element,
            "point": [float(line[start : start + 8]) for start in (30, 38, 46)],
            "rank": (-occupancy, line[16].strip()),
        }
        if identity not in selected or record["rank"] < selected[identity]["rank"]:
            selected[identity] = record
    return [selected[identity] for identity in sorted(selected)]


def sampled_surface(records: list[dict], query: dict) -> np.ndarray:
    """Count unoccluded sampling points by bounded NumPy distance arithmetic."""
    count = query["surface_points"]
    # Freeze the declared golden spiral and float32 unit directions. The
    # independent path uses direct squared distances, with no Bio.PDB KD trees.
    longitude, z = 0.0, 1 - 1 / count
    angle_step = math.pi * (3 - math.sqrt(5))
    directions = np.empty((count, 3), dtype=np.float32)
    for index in range(count):
        radial = math.sqrt(1 - z * z)
        directions[index] = (radial * math.cos(longitude), radial * math.sin(longitude), z)
        longitude += angle_step
        z -= 2 / count
    centers = np.array([record["point"] for record in records], dtype=np.float64)
    radii = np.array([query["radii"][record["element"]] + query["probe_radius"] for record in records])
    areas = np.empty(len(records), dtype=np.float64)
    for index, center in enumerate(centers):
        squared = np.sum((centers - center) ** 2, axis=1)
        neighbors = np.flatnonzero(squared < (radii[index] + radii) ** 2)
        sphere = directions * radii[index] + center
        exposed = np.ones(count, dtype=bool)
        for neighbor in neighbors:
            if neighbor != index:
                exposed &= np.sum((sphere - centers[neighbor]) ** 2, axis=1) > radii[neighbor] ** 2
        areas[index] = int(exposed.sum()) * (4 * math.pi / count) * radii[index] ** 2
    return areas


def independent_reference(content: str, query: dict) -> dict:
    records = pdb_atoms(content)
    proteins = [record for record in records if record["protein"]]
    hemes = defaultdict(list)
    residue_indices = defaultdict(list)
    for index, record in enumerate(proteins):
        residue_indices[record["residue"]].append(index)
    for record in records:
        if not record["protein"]:
            hemes[record["residue"]].append(record)
    if len(hemes) != query["expected_hemes"] or len(residue_indices) != query["expected_protein_residues"]:
        raise ValueError("Paired PDB does not contain the frozen complete structure")
    coordinates = np.array([record["point"] for record in proteins])
    protein_surface = sampled_surface(proteins, query)
    result = {name: {} for name in ("summaries", "contacts", "residues", "heme_atoms")}
    for heme_id, heme in sorted(hemes.items()):
        complex_surface = sampled_surface(proteins + heme, query)
        free_surface = sampled_surface(heme, query)
        # 4,000-ish by 43 distances are bounded; no full atom-by-atom matrix is made.
        heme_coordinates = np.array([record["point"] for record in heme])
        distances = np.sqrt(np.sum((coordinates[:, None, :] - heme_coordinates[None, :, :]) ** 2, axis=2))
        contacting = distances <= query["contact_cutoff"]
        for protein_index, heme_index in zip(*np.nonzero(contacting), strict=True):
            protein_atom, heme_atom = proteins[protein_index], heme[heme_index]
            result["contacts"][f"{heme_atom['id']}|{protein_atom['id']}"] = {
                "heme": heme_id,
                "heme_atom": heme_atom["atom"],
                "protein_atom": protein_atom["id"],
                "protein_residue": protein_atom["residue"],
                "distance": float(distances[protein_index, heme_index]),
            }
        for index, atom in enumerate(heme):
            closest = min(range(len(proteins)), key=lambda i: (distances[i, index], proteins[i]["id"]))
            bound = float(complex_surface[len(proteins) + index])
            result["heme_atoms"][atom["id"]] = {
                "heme": heme_id,
                "element": atom["element"],
                "nearest_protein_atom": proteins[closest]["id"],
                "minimum_distance": float(distances[closest, index]),
                "sasa_free": float(free_surface[index]),
                "sasa_complex": bound,
                "buried_area": float(free_surface[index]) - bound,
            }
        for residue, indices in sorted(residue_indices.items()):
            first, second = min(
                ((i, j) for i in indices for j in range(len(heme))),
                key=lambda pair: (distances[pair], proteins[pair[0]]["id"], heme[pair[1]]["id"]),
            )
            baseline = math.fsum(float(protein_surface[i]) for i in indices)
            bound = math.fsum(float(complex_surface[i]) for i in indices)
            result["residues"][f"{heme_id}|{residue}"] = {
                "heme": heme_id,
                "residue": residue,
                "chain": proteins[indices[0]]["chain"],
                "residue_name": proteins[indices[0]]["name"],
                "nearest_protein_atom": proteins[first]["id"],
                "nearest_heme_atom": heme[second]["id"],
                "minimum_distance": float(distances[first, second]),
                "contact_pairs": int(contacting[indices].sum()),
                "in_pocket": int(contacting[indices].any()),
                "sasa_without_heme": baseline,
                "sasa_with_heme": bound,
                "buried_area": baseline - bound,
            }
        irons = [atom for atom in heme if atom["element"] == "FE"]
        if len(irons) != 1:
            raise ValueError("HEM does not have exactly one iron")
        iron = result["heme_atoms"][irons[0]["id"]]
        protein_burial = math.fsum(
            float(a - b) for a, b in zip(protein_surface, complex_surface[: len(proteins)], strict=True)
        )
        heme_burial = math.fsum(
            float(a - b) for a, b in zip(free_surface, complex_surface[len(proteins) :], strict=True)
        )
        result["summaries"][heme_id] = {
            "protein_residues": len(residue_indices),
            "pocket_residues": sum(bool(contacting[indices].any()) for indices in residue_indices.values()),
            "contact_pairs": int(contacting.sum()),
            "contacting_chains": len({proteins[i]["chain"] for i in np.flatnonzero(contacting.any(axis=1))}),
            "heme_heavy_atoms": len(heme),
            "protein_buried_area": protein_burial,
            "heme_buried_area": heme_burial,
            "total_buried_area": protein_burial + heme_burial,
            "nearest_iron_protein_atom": iron["nearest_protein_atom"],
            "iron_minimum_distance": iron["minimum_distance"],
        }
    return {"query": query, **result}


def prepare(source: Path, output: Path) -> None:
    started = time.time()
    protocol = json.loads(PROTOCOL.read_text())
    for package, expected in protocol["required_packages"].items():
        if version(package) != expected:
            raise ValueError(f"Package version differs from frozen protocol: {package}")
    contents = {}
    for extension in ("pdb", "cif"):
        contents[extension] = gzip.decompress((source / f"4hhb.{extension}.gz").read_bytes())
        if hashlib.sha256(contents[extension]).hexdigest() != protocol[f"{extension}_sha256"]:
            raise ValueError(f"Changed deposited {extension} observations")
    output.mkdir(parents=True, exist_ok=False)
    inputs, private = output / "inputs", output / "reference"
    inputs.mkdir()
    private.mkdir()
    (inputs / "structure.cif").write_bytes(contents["cif"])
    (inputs / "query.json").write_text(json.dumps(protocol["query"], indent=2) + "\n")
    reference = independent_reference(contents["pdb"].decode(), protocol["query"])
    content = (json.dumps(reference, indent=2, allow_nan=False) + "\n").encode()
    compressed = gzip.compress(content, mtime=0)
    filename = "4hhb-heme-pocket-reference.json.gz"
    (private / filename).write_bytes(compressed)
    asset = {
        "url": "https://doi.org/10.2210/pdb4HHB/pdb",
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "vendored_sha256": hashlib.sha256(compressed).hexdigest(),
        "content_bytes": len(content),
        "vendored_bytes": len(compressed),
        "transformation": (
            "Private paired-PDB parse, complete heavy-atom distances and independent sampled SASA reference; "
            "never supplied to solver."
        ),
    }
    (private / "source-asset.json").write_text(json.dumps({filename: asset}, indent=2) + "\n")
    (private / "preparation.json").write_text(
        json.dumps(
            {
                "started_at_unix": started,
                "ended_at_unix": time.time(),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "packages": {name: version(name) for name in ("numpy", "biopython")},
                "tables": {key: len(value) for key, value in reference.items() if key != "query"},
                "native_oracle": "pending",
                "harbor": "pending",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory with pinned 4hhb.pdb.gz and 4hhb.cif.gz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
