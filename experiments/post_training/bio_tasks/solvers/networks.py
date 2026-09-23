# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input-derived reaction and interaction graph checks."""

import heapq
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import partial
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import tab_rows, table


def reaction_network(path: Path) -> tuple[dict, list[tuple]]:
    root = ET.parse(path).getroot()
    species = {item.attrib["id"]: item.attrib for item in root.findall(".//{*}species")}
    reactions = []
    for item in root.findall(".//{*}reaction"):
        sides = []
        for tag in ["listOfReactants", "listOfProducts"]:
            sides.append(
                {
                    ref.attrib["species"]: float(ref.attrib.get("stoichiometry", "1"))
                    for ref in item.findall("{*}" + tag + "/{*}speciesReference")
                }
            )
        reactions.append((item.attrib["id"], *sides, item.attrib["reversible"] == "true"))
    return species, reactions


def solve_networks(inputs: Path, operation: str) -> list[dict]:
    answer = []
    if operation in {"pathway-reachability", "stoichiometric-balance", "reaction-mass-balance"}:
        species, reactions = reaction_network(inputs / "network.xml")
        if operation == "pathway-reachability":
            available = set((inputs / "seeds.txt").read_text().splitlines())
            while True:
                before = len(available)
                for _, left, right, reversible in reactions:
                    if set(left) <= available:
                        available.update(right)
                    if reversible and set(right) <= available:
                        available.update(left)
                if len(available) == before:
                    break
            return [{"id": name, "reachable": int(name in available)} for name in species]
        if operation == "stoichiometric-balance":
            fluxes = {row["reaction"]: float(row["flux"]) for row in table(inputs / "fluxes.csv")}
            residuals = dict.fromkeys(species, 0.0)
            for name, left, right, _ in reactions:
                for sign, side in [(-1, left), (1, right)]:
                    for metabolite, coefficient in side.items():
                        residuals[metabolite] += sign * coefficient * fluxes[name]
            return [
                {"id": name, "residual": value, "balanced": int(abs(value) <= 1e-9)}
                for name, value in residuals.items()
                if species[name]["boundaryCondition"] == "false"
            ]
        compositions = {}
        for row in table(inputs / "metabolites.csv"):
            atoms = defaultdict(int)
            for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", row["formula"]):
                atoms[element] += int(count or "1")
            atoms["charge"] = int(row["charge"])
            compositions[row["species"]] = atoms
        for name, left, right, _ in reactions:
            difference = defaultdict(float)
            for sign, side in [(-1, left), (1, right)]:
                for metabolite, coefficient in side.items():
                    for element, count in compositions[metabolite].items():
                        difference[element] += sign * coefficient * count
            answer.append(
                {
                    "id": name,
                    "balanced": int(all(value == 0 for value in difference.values())),
                    **{key: int(difference[key]) for key in ["H", "O", "charge"]},
                }
            )
        return answer
    nodes = (inputs / "nodes.txt").read_text().splitlines()
    graph = defaultdict(list)
    if operation == "interaction-components":
        for left, _, right in tab_rows(inputs / "network.sif"):
            graph[left].append(right)
            graph[right].append(left)
        unseen = set(nodes)
        while unseen:
            start = min(unseen)
            component = {start}
            stack = [start]
            while stack:
                for neighbor in graph[stack.pop()]:
                    if neighbor not in component:
                        component.add(neighbor)
                        stack.append(neighbor)
            unseen -= component
            answer.extend({"id": name, "component": min(component), "size": len(component)} for name in component)
    elif operation == "regulatory-path-signs":
        for row in table(inputs / "network.csv"):
            graph[row["source"]].append((row["target"], int(row["sign"])))
        start = (inputs / "source.txt").read_text().strip()
        signs = defaultdict(set)
        queue = [(start, 1)]
        while queue:
            node, sign = queue.pop()
            if sign in signs[node]:
                continue
            signs[node].add(sign)
            queue.extend((target, sign * edge_sign) for target, edge_sign in graph[node])
        for name in nodes:
            value = ",".join(symbol for number, symbol in [(1, "+"), (-1, "-")] if number in signs[name]) or "none"
            answer.append({"id": name, "signs": value})
    else:
        for row in table(inputs / "edges.csv"):
            graph[row["source"]].append((row["target"], float(row["cost"])))
        start = (inputs / "source.txt").read_text().strip()
        distances = {start: 0.0}
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distances[node]:
                continue
            for target, weight in graph[node]:
                candidate = cost + weight
                if candidate < distances.get(target, float("inf")):
                    distances[target] = candidate
                    heapq.heappush(queue, (candidate, target))
        answer = [{"id": name, "distance": distances.get(name)} for name in nodes]
    return answer


NAMES = (
    "pathway-reachability",
    "stoichiometric-balance",
    "reaction-mass-balance",
    "interaction-components",
    "regulatory-path-signs",
    "network-shortest-paths",
)
SOLVERS = {name: partial(solve_networks, operation=name) for name in NAMES}
