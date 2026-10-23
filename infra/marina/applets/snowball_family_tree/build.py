# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Snowball family tree applet from its reviewed graph and frontend."""

import json
import shutil
from graphlib import TopologicalSorter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"

data = json.loads((ROOT / "graph-data.json").read_text())
nodes = {node["id"]: node for node in data["nodes"]}
if len(nodes) != len(data["nodes"]):
    raise ValueError("Graph contains duplicate node IDs")
for node in nodes.values():
    for parent_id in node["parents"]:
        if parent_id not in nodes:
            raise ValueError(f"Graph node {node['id']} has missing parent {parent_id}")
        parent = nodes[parent_id]
        if parent["lane"] != "shared" and node["lane"] != parent["lane"]:
            raise ValueError(f"Graph node {node['id']} crosses cooldown lanes from parent {parent_id}")
TopologicalSorter({node_id: node["parents"] for node_id, node in nodes.items()}).prepare()

if DIST.exists():
    shutil.rmtree(DIST)
DIST.mkdir()
serialized = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
html = (ROOT / "index.html").read_text().replace("__DATA__", serialized)
(DIST / "index.html").write_text(html)
for asset in ["app.js", "search.js", "timeline.js", "style.css"]:
    shutil.copyfile(ROOT / asset, DIST / asset)
(DIST / "evidence").mkdir(exist_ok=True)
for evidence in (ROOT / "evidence").glob("*.json"):
    shutil.copyfile(evidence, DIST / "evidence" / evidence.name)
