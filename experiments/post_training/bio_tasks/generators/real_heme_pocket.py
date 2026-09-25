# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete observed heme-pocket and solvent-exposure artifact contracts."""

import json

from experiments.post_training.bio_tasks.contract import Column, Contract, TableContract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, OracleRuntime, Recipe, WorkflowScope

SOURCE_ID = "PDB:4HHB"
REFERENCE_ASSET = "4hhb-heme-pocket-reference.json.gz"


def heme_pocket_contract(reference: dict) -> Contract:
    """Check every ligand contact, residue and ligand atom, including zero contacts."""
    text = {
        name: Column(kind="text", unit="author identity", description=description)
        for name, description in (
            ("heme", "complete HEM residue identity"),
            ("heme_atom", "HEM author atom name"),
            ("protein_atom", "complete author protein atom identity"),
            ("protein_residue", "complete author protein residue identity"),
            ("residue", "complete author protein residue identity"),
            ("chain", "protein author chain identity"),
            ("residue_name", "deposited residue name"),
            ("element", "deposited uppercase element symbol"),
            ("nearest_protein_atom", "nearest protein atom identity; exact ties use identity order"),
            ("nearest_heme_atom", "nearest HEM atom identity; exact ties use identity order"),
            ("nearest_iron_protein_atom", "nearest protein atom to the HEM iron, without a bond claim"),
        )
    }
    distances = {
        name: Column(kind="number", unit="angstrom", description=description, atol=1e-8, rtol=1e-10)
        for name, description in (
            ("distance", "distance between the named heavy atoms"),
            ("minimum_distance", "minimum heavy-atom distance over the declared atom universe"),
            ("iron_minimum_distance", "distance from iron to the nearest selected protein heavy atom"),
        )
    }
    areas = {
        name: Column(kind="number", unit="square angstrom", description=description, atol=1e-7, rtol=1e-10)
        for name, description in (
            ("sasa_free", "isolated HEM atom sampled solvent-accessible area"),
            ("sasa_complex", "HEM atom sampled area with all globin chains present"),
            ("sasa_without_heme", "protein residue sampled area in the complete globin complex without HEM"),
            ("sasa_with_heme", "protein residue sampled area after adding the named HEM"),
            ("buried_area", "sampled area lost on adding the partner, without a factor of one half"),
            ("protein_buried_area", "sum of sampled area loss over all globin atoms"),
            ("heme_buried_area", "sum of sampled area loss over the named HEM atoms"),
            ("total_buried_area", "protein plus HEM sampled area loss, without dividing by two"),
        )
    }
    integers = {
        name: Column(kind="integer", unit=unit, description=description)
        for name, unit, description in (
            ("protein_residues", "residues", "all selected protein residues, not just the pocket"),
            ("pocket_residues", "residues", "distinct residues with at least one passing atom pair"),
            ("contact_pairs", "atom pairs", "distinct protein-HEM heavy-atom pairs inside the inclusive cutoff"),
            ("contacting_chains", "chains", "distinct author protein chains among passing contacts"),
            ("heme_heavy_atoms", "atoms", "all selected heavy atoms in the named HEM"),
            ("in_pocket", "indicator", "one if minimum distance is at most the inclusive cutoff, otherwise zero"),
        )
    }
    columns = {**text, **distances, **areas, **integers}
    schemas = {
        "contacts.tsv": ("contacts", ("heme", "heme_atom", "protein_atom", "protein_residue", "distance")),
        "residues.tsv": (
            "residues",
            (
                "heme",
                "residue",
                "chain",
                "residue_name",
                "nearest_protein_atom",
                "nearest_heme_atom",
                "minimum_distance",
                "contact_pairs",
                "in_pocket",
                "sasa_without_heme",
                "sasa_with_heme",
                "buried_area",
            ),
        ),
        "heme_atoms.tsv": (
            "heme_atoms",
            ("heme", "element", "nearest_protein_atom", "minimum_distance", "sasa_free", "sasa_complex", "buried_area"),
        ),
    }
    summary = (
        "protein_residues",
        "pocket_residues",
        "contact_pairs",
        "contacting_chains",
        "heme_heavy_atoms",
        "protein_buried_area",
        "heme_buried_area",
        "total_buried_area",
        "nearest_iron_protein_atom",
        "iron_minimum_distance",
    )
    return Contract(
        columns={name: columns[name] for name in summary},
        expected=reference["summaries"],
        tables={
            filename: TableContract(
                columns={name: columns[name] for name in names}, expected=reference[field], max_bytes=8 * 1024 * 1024
            )
            for filename, (field, names) in schemas.items()
        },
    )


def generate_heme_pocket(_seed: int) -> Instance:
    reference = json.loads(source_text(SOURCE_ID, REFERENCE_ASSET))
    contract = heme_pocket_contract(reference)
    return Instance(
        "Audit the four heme pockets of observed human deoxyhemoglobin 4HHB using the unchanged mmCIF "
        "in /app/inputs/structure.cif and the frozen numerical method in query.json. Determine which "
        "globin residues approach each HEM, and how much sampled solvent-accessible area the protein "
        "and that HEM lose when brought together at the deposited coordinates. Read model 1 only. "
        "Select protein ATOM records and HETATM HEM records, omit hydrogen/deuterium and nonpositive "
        "occupancy, and resolve alternate locations per author atom identity using highest occupancy "
        "then lexicographically smallest alternate ID ('.' and '?' mean blank). Use auth_asym_id, "
        "auth_seq_id, insertion code, auth_comp_id and auth_atom_id, preserving all author identities. "
        "Parse deposited decimal coordinates as float64. Residue IDs are "
        "kind:chain:sequence:insertion:residue_name, with kind protein or HEM and '.' for blank insertion; "
        "atom IDs append :atom_name. Keep all four protein chains; do not add crystal symmetry mates. "
        "For each HEM separately, enumerate every protein-HEM heavy-atom pair at distance <=4.0 A in "
        "contacts.tsv, with id=heme_atom_ID|protein_atom_ID. Include every protein residue, including "
        "zero-contact residues, in residues.tsv with id=heme_residue_ID|protein_residue_ID. Report its "
        "nearest atom pair (distance ties ordered by protein then HEM atom identity), contact-pair count, "
        "pocket indicator, and solvent areas. In heme_atoms.tsv report every selected HEM atom keyed by "
        "its atom ID, nearest protein atom (distance ties ordered by identity), minimum distance, and "
        "its isolated and complex solvent areas. Use Biopython 1.88 ShrakeRupley, 960 golden-spiral "
        "sample points, 1.4 A probe and the exact element radii in query.json. Baseline protein is all "
        "selected globin chains with all HEMs removed. For each complex add only the named HEM; omit "
        "other HEMs, water and phosphate. Isolated HEM uses its unchanged coordinates. Per-partner "
        "burial is SASA without partner minus SASA with partner; total burial is the sum of protein "
        "and HEM losses, without dividing by two. Report answer.json for all four HEM residues, "
        "including distinct pocket residues, atom-pair count, contacting chain count, selected atom "
        "counts, buried areas and the nearest protein atom/distance to the HEM iron. Preserve full "
        "precision in artifacts. These are geometric proximity and specified sampled-exposure "
        "measurements: they do not establish hydrogen bonds, salt bridges, binding affinities, "
        "thermodynamic free energies or relaxed apo/holo structural changes.",
        {
            "structure.cif": source_text(SOURCE_ID, "4hhb.cif.gz"),
            "query.json": json.dumps(reference["query"], indent=2) + "\n",
        },
        contract,
        {
            "halved_total_buried_area": [
                {**row, "total_buried_area": row["total_buried_area"] / 2} for row in contract.answer()
            ],
            "reversed_protein_burial": [
                {**row, "protein_buried_area": -row["protein_buried_area"]} for row in contract.answer()
            ],
        },
        data_origin=DataOrigin.REAL,
        source_ids=(SOURCE_ID,),
        derivation=(
            "Complete deposited 4HHB mmCIF; four observed HEM cofactors and all globin chains. "
            "Private reference independently parses paired PDB decimal coordinates and evaluates "
            "distances and sampled SASA with NumPy. Native oracle uses Biopython mmCIF parsing, "
            "NeighborSearch and ShrakeRupley. No new biological observations or benchmark inputs."
        ),
        workflow_scope=WorkflowScope.CONNECTED,
    )


RECIPES = (
    Recipe(
        id="real-heme-pocket-burial",
        version="1",
        skills=("author atom identities", "ligand pocket extraction", "solvent exposure", "complete structural audit"),
        formats=("mmCIF", "TSV", "JSON query"),
        sources=("https://www.rcsb.org/structure/4HHB",),
        generate=generate_heme_pocket,
        oracle_timeout=1200,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
