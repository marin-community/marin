"""Choose a reproducible, diverse construction pilot from accepted blueprints.

Select only existing viable pairings. Never invent a proposal to fill a cell.
Among equal coverage gains, prefer stronger reviews and deeper construction DAGs.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from .catalog import load_pilot
from .inference import atomic_json, digest
from .prompts import ENVIRONMENTS, VERIFIERS
from .validation import AXES


def attach_provenance(items, pilot):
    """Backfill early experiment inputs from their recorded, validated manifest."""
    records = {record["capability_id"]: record for record in pilot["capabilities"]}
    result = []
    for item in items:
        record = records[item["proposal"]["capability_id"]]
        provenance = {
            "catalog_source": pilot["source"],
            "capability_record": record,
            "capability_record_hash": digest(record),
        }
        if "provenance" in item and item["provenance"] != provenance:
            raise ValueError("accepted provenance differs from recorded pilot")
        result.append(dict(item, provenance=provenance))
    return result


def select(items, count=9):
    if count < 1:
        raise ValueError("count must be positive")
    candidates = []
    for item in items:
        proposal = item["proposal"]
        review = item["review"]
        if digest(proposal) != item["proposal_hash"]:
            raise ValueError("accepted proposal fingerprint mismatch")
        if (
            review["verdict"] != "accept"
            or set(review["scores"]) != AXES
            or any(
                type(score) is not int or score not in (4, 5)
                for score in review["scores"].values()
            )
            or review["critical_failures"]
            or review.get("required_changes")
        ):
            raise ValueError("construction selection requires passing reviews")
        candidates.append(item)
    if len({i["proposal_hash"] for i in candidates}) != len(candidates):
        raise ValueError("duplicate proposal in candidate pool")
    environments, verifiers, capabilities, pairs = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    selected = []
    while candidates and len(selected) < count:

        def score(item):
            p = item["proposal"]
            e, v, c = p["environment"], p["verification"], p["capability_id"]
            return (
                int(environments[e] == 0) + int(verifiers[v] == 0),
                int(environments[e] < 2) + int(verifiers[v] < 2),
                int(capabilities[c] == 0),
                int(pairs[e, v] == 0),
                sum(item["review"]["scores"].values()),
                len(p["builder_plan"]),
                item["proposal_hash"],
            )

        chosen = max(candidates, key=score)
        candidates.remove(chosen)
        selected.append(chosen)
        p = chosen["proposal"]
        environments[p["environment"]] += 1
        verifiers[p["verification"]] += 1
        capabilities[p["capability_id"]] += 1
        pairs[p["environment"], p["verification"]] += 1
    report = {
        "requested": count,
        "selected": len(selected),
        "candidate_pool": len(items),
        "input_hash": digest(items),
        "output_hash": digest(selected),
        "environments": dict(environments),
        "verifiers": dict(verifiers),
        "capabilities": dict(capabilities),
        "pairs": {f"{e}/{v}": n for (e, v), n in pairs.items()},
        "missing_environments": sorted(set(ENVIRONMENTS) - environments.keys()),
        "missing_verifiers": sorted(set(VERIFIERS) - verifiers.keys()),
        "policy": "coverage, replicate coverage, capability diversity, valid pair diversity, quality, DAG depth; no invented cells",
    }
    return selected, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accepted", type=Path)
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--pilot", type=Path, help="original input_pilot.json for source provenance"
    )
    args = parser.parse_args(argv)
    candidates = json.loads(args.accepted.read_text())
    items, report = select(candidates, args.count)
    if args.pilot:
        items = attach_provenance(items, load_pilot(args.pilot))
        report["provenance_manifest"] = str(args.pilot)
        report["output_hash"] = digest(items)
    atomic_json(args.out, items)
    atomic_json(args.out.with_suffix(".selection.json"), report)
    print(json.dumps(report, indent=2))
    return (
        0
        if len(items) == args.count
        and not report["missing_environments"]
        and not report["missing_verifiers"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
