# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reaction networks and signed or weighted biological interaction graphs."""

import random
import xml.etree.ElementTree as ET
from functools import partial

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Difficulty, Instance, Recipe, csv_text


def sbml_text(species: list[str], reactions: list[tuple], boundary: tuple[str, ...] = ()) -> str:
    root = ET.Element("sbml", xmlns="http://www.sbml.org/sbml/level3/version2/core", level="3", version="2")
    model = ET.SubElement(root, "model", id="fixture")
    compartments = ET.SubElement(model, "listOfCompartments")
    ET.SubElement(compartments, "compartment", id="cell", constant="true", size="1", spatialDimensions="3")
    entries = ET.SubElement(model, "listOfSpecies")
    for name in species:
        ET.SubElement(
            entries,
            "species",
            id=name,
            compartment="cell",
            initialAmount="0",
            hasOnlySubstanceUnits="true",
            boundaryCondition=str(name in boundary).lower(),
            constant="false",
        )
    entries = ET.SubElement(model, "listOfReactions")
    for name, reactants, products, reversible in reactions:
        reaction = ET.SubElement(entries, "reaction", id=name, reversible=str(reversible).lower())
        for tag, terms in [("listOfReactants", reactants), ("listOfProducts", products)]:
            if terms:
                group = ET.SubElement(reaction, tag)
                for species_id, coefficient in terms.items():
                    ET.SubElement(
                        group, "speciesReference", species=species_id, stoichiometry=str(coefficient), constant="true"
                    )
    return ET.tostring(root, encoding="unicode") + "\n"


def generate_networks(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    inputs, expected = {}, {}
    if operation == "pathway-reachability":
        prefix = f"m{rng.randint(10,99)}_"
        names = {letter: prefix + letter for letter in "ABCDEFGH"}
        reactions = [
            ("r1", {"A": 1, "B": 1}, {"C": 1}, False),
            ("r2", {"A": 1}, {"D": 1}, False),
            ("r3", {"D": 1}, {"E": 1}, False),
            ("r4", {"G": 1}, {"H": 1}, True),
        ]
        reactions = [
            (rid, {names[k]: v for k, v in left.items()}, {names[k]: v for k, v in right.items()}, rev)
            for rid, left, right, rev in reactions
        ]
        rng.shuffle(reactions)
        inputs = {
            "network.xml": sbml_text(list(names.values()), reactions),
            "seeds.txt": names["A"] + "\n" + names["H"] + "\n",
        }
        expected = {name: {"reachable": int(letter in "ADEGH")} for letter, name in names.items()}
        columns = {
            "reachable": Column(
                kind="integer", unit="decision", description="availability closure under feasible reaction directions"
            )
        }
        prompt = (
            "Compute qualitative metabolite availability from seeds.txt in the SBML network.xml. A "
            "reaction direction can add every product only when all its reactants are available; "
            "available metabolites are not consumed. Reversible reactions also permit the reverse rule."
            " Iterate to closure and report every species id with reachable=0/1. Do not reduce "
            "multi-reactant reactions to independent graph edges."
        )
        wrong = [
            {"id": name, "reachable": 1 if name.endswith("_C") else row["reachable"]} for name, row in expected.items()
        ]
        reason = "ignored_missing_cosubstrate"
    elif operation == "stoichiometric-balance":
        flow, delta = rng.randint(2, 7), rng.randint(1, 3)
        reactions = [
            ("source", {"X": 1}, {"A": 1}, False),
            ("split", {"A": 1}, {"B": 2}, False),
            ("convert", {"B": 1}, {"C": 1}, False),
            ("sink", {"C": 1}, {}, False),
        ]
        inputs = {
            "network.xml": sbml_text(["X", "A", "B", "C"], reactions, ("X",)),
            "fluxes.csv": csv_text(
                [
                    {"reaction": name, "flux": value}
                    for name, value in [
                        ("source", flow),
                        ("split", flow),
                        ("convert", 2 * flow + delta),
                        ("sink", 2 * flow),
                    ]
                ]
            ),
        }
        expected = {
            "A": {"residual": 0.0, "balanced": 1},
            "B": {"residual": float(-delta), "balanced": 0},
            "C": {"residual": float(delta), "balanced": 0},
        }
        columns = {
            "residual": Column(
                kind="number",
                unit="mmol per gram dry weight per hour",
                description="production minus consumption S*v",
                atol=1e-10,
                rtol=1e-8,
            ),
            "balanced": Column(kind="integer", unit="decision", description="1 if abs(residual)<=1e-9"),
        }
        prompt = (
            "Audit the supplied flux vector in fluxes.csv against SBML network.xml. For every "
            "non-boundary species, compute S*v with products positive and reactants negative, "
            "respecting stoichiometric coefficients. Report balanced=1 for abs(residual)<=1e-9. Exclude"
            " boundaryCondition=true species. All supplied fluxes use mmol/gDW/hour and the written "
            "reaction direction; do not optimize or refit them."
        )
        wrong = [{"id": k, **v, "residual": -v["residual"]} for k, v in expected.items()]
        reason = "reversed_stoichiometric_sign"
    elif operation == "reaction-mass-balance":
        factor = rng.randint(1, 4)
        reactions = [
            ("balanced", {"glucose": factor, "oxygen": 6 * factor}, {"co2": 6 * factor, "water": 6 * factor}, False),
            ("unbalanced", {"glucose": factor, "oxygen": 6 * factor}, {"co2": 6 * factor, "water": 5 * factor}, False),
            ("salt", {"sodium": 1, "chloride": 1}, {"salt": 1}, False),
        ]
        metadata = [
            ("glucose", "C6H12O6", 0),
            ("oxygen", "O2", 0),
            ("co2", "CO2", 0),
            ("water", "H2O", 0),
            ("sodium", "Na", 1),
            ("chloride", "Cl", -1),
            ("salt", "NaCl", 0),
        ]
        inputs = {
            "network.xml": sbml_text([row[0] for row in metadata], reactions),
            "metabolites.csv": csv_text(
                [{"species": name, "formula": formula, "charge": charge} for name, formula, charge in metadata]
            ),
        }
        for name in ["balanced", "unbalanced", "salt"]:
            expected[name] = {
                "balanced": int(name != "unbalanced"),
                "H": -2 * factor if name == "unbalanced" else 0,
                "O": -factor if name == "unbalanced" else 0,
                "charge": 0,
            }
        columns = {
            name: Column(
                kind="integer",
                unit="atoms" if name in {"H", "O"} else "elementary charges" if name == "charge" else "decision",
                description=(
                    "product minus reactant residual" if name != "balanced" else "1 if all elements and charge balance"
                ),
            )
            for name in ["balanced", "H", "O", "charge"]
        }
        prompt = (
            "Check elemental and charge balance of every SBML network.xml reaction using "
            "metabolites.csv formulas and integer charges. Formulas contain element symbols and "
            "optional positive integer counts, without parentheses. Include all elements when deciding "
            "balanced; report H, O and charge residuals (products minus reactants) plus balanced=0/1. "
            "Use reaction id. Stoichiometric coefficients apply to both atoms and charge."
        )
        wrong = [{"id": k, **v, "balanced": 1} for k, v in expected.items()]
        reason = "assumed_carbon_balance_implies_mass_balance"
    elif operation == "interaction-components":
        offset = rng.randint(10, 99)
        names = [f"p{offset+i}" for i in range(7)]
        a, b, c, d, e, f, g = names
        edges = [(a, b), (b, c), (c, a), (a, b), (d, e), (e, e)]
        rng.shuffle(edges)
        inputs = {
            "network.sif": "".join(f"{left}\tpp\t{right}\n" for left, right in edges),
            "nodes.txt": "\n".join(names) + "\n",
        }
        for members in [names[:3], names[3:5], [f], [g]]:
            for name in members:
                expected[name] = {"component": min(members), "size": len(members)}
        columns = {
            "component": Column(
                kind="text", unit="node identifier", description="lexically smallest node in undirected component"
            ),
            "size": Column(kind="integer", unit="nodes", description="unique nodes in component"),
        }
        prompt = (
            "Find connected components of the undirected protein interaction network.sif (source, "
            "interaction type, target). Include isolated nodes from nodes.txt, ignore duplicate edges, "
            "and let self-loops leave connectivity unchanged. Report every node with component labeled "
            "by its lexically smallest member and component size."
        )
        wrong = [{"id": k, **v, "size": v["size"] + 1} for k, v in expected.items()]
        reason = "counted_duplicate_or_self_edges_as_nodes"
    elif operation == "regulatory-path-signs":
        sign = rng.choice([-1, 1])
        first_sign = rng.choice([-1, 1])
        rows = [("A", "B", first_sign), ("B", "D", -1), ("A", "C", -1), ("C", "D", sign), ("D", "E", -1)]
        inputs = {
            "network.csv": csv_text([{"source": a, "target": b, "sign": value} for a, b, value in rows]),
            "source.txt": "A\n",
            "nodes.txt": "A\nB\nC\nD\nE\nF\n",
        }
        expected = {
            "A": {"signs": "+"},
            "B": {"signs": "+" if first_sign == 1 else "-"},
            "C": {"signs": "-"},
            "D": {"signs": "+,-" if first_sign != sign else "-" if sign == 1 else "+"},
            "E": {"signs": "+,-" if first_sign != sign else "+" if sign == 1 else "-"},
            "F": {"signs": "none"},
        }
        columns = {"signs": Column(kind="text", unit="path-sign set", description="+, -, +,-, or none")}
        prompt = (
            "Propagate possible regulation signs from source.txt through the directed acyclic "
            "network.csv. Path sign is the product of edge signs; report all distinct path signs per "
            "node, including + for the source empty path. Encode as +, -, +,-, or none. Multiple "
            "contradictory paths remain both signs; do not cancel them. Include all nodes.txt nodes."
        )
        wrong = [{"id": k, **v, "signs": "none" if k == "D" else v["signs"]} for k, v in expected.items()]
        reason = "discarded_reachable_regulatory_paths"
    else:
        assert operation == "network-shortest-paths"
        weight = rng.randint(2, 7)
        scale = rng.randint(1, 6)
        edges = [("A", "B", weight), ("A", "C", 1), ("C", "B", 1), ("B", "D", 2), ("C", "D", 10), ("D", "E", 1)]
        inputs = {
            "edges.csv": csv_text([{"source": a, "target": b, "cost": c * scale} for a, b, c in edges]),
            "source.txt": "A\n",
            "nodes.txt": "A\nB\nC\nD\nE\nF\n",
        }
        expected = {
            name: {"distance": value * scale if value is not None else None}
            for name, value in [("A", 0), ("B", 2), ("C", 1), ("D", 4), ("E", 5), ("F", None)]
        }
        columns = {
            "distance": Column(
                kind="number",
                unit="path cost",
                description="minimum sum of directed edge costs; null if unreachable",
                nullable=True,
                atol=1e-10,
                rtol=1e-8,
            )
        }
        prompt = (
            "Compute minimum total directed path cost from source.txt to every nodes.txt node using "
            "edges.csv. Costs are positive; they are costs to sum, not strengths to invert. Keep edge "
            "direction, source distance 0, and null for unreachable nodes. Use node id."
        )
        wrong = [{"id": k, "distance": 0 if v["distance"] is None else v["distance"]} for k, v in expected.items()]
        reason = "encoded_unreachable_as_zero_distance"
    return Instance(
        prompt + " Inputs are in /app/inputs.", inputs, Contract(columns=columns, expected=expected), {reason: wrong}
    )


SKILLS = {
    "pathway-reachability": ("cosubstrate-logic", "reversible-reactions", "closure"),
    "stoichiometric-balance": ("stoichiometric-matrix", "flux-residual", "boundary-species"),
    "reaction-mass-balance": ("chemical-formulas", "element-balance", "charge"),
    "interaction-components": ("undirected-network", "isolates", "duplicate-edges"),
    "regulatory-path-signs": ("signed-paths", "contradictory-regulation", "dag"),
    "network-shortest-paths": ("weighted-paths", "directionality", "unreachable-nodes"),
}
RECIPES = tuple(
    Recipe(
        name,
        "1",
        Difficulty.MEDIUM,
        skills,
        (
            ("sbml-level3-version2-core", "csv-header")
            if name in {"pathway-reachability", "stoichiometric-balance", "reaction-mass-balance"}
            else ("sif",) if name == "interaction-components" else ("csv-header",)
        ),
        (
            ("https://sbml.org/documents/specifications/level-3/version-2/",)
            if name in {"pathway-reachability", "stoichiometric-balance", "reaction-mass-balance"}
            else ("https://networkx.org/documentation/stable/reference/algorithms/index.html",)
        ),
        partial(generate_networks, operation=name),
    )
    for name, skills in SKILLS.items()
)
