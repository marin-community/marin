# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# H3 readout: pull the 3 launched runs' final uncheatable bpb from W&B, compare to the
# pre-registered rule and the historical anchor. The runs live in the PRIVATE entity
# stanford-mercury/marin, so a WANDB_API_KEY is required (source wandb.env first).
# Usage: source <wandb.env> && .venv/bin/python scratch/mixture_features/h3/readout.py
import base64
import json
import os
import urllib.request
from pathlib import Path

WANDB_GRAPHQL_URL = "https://api.wandb.ai/graphql"
WANDB_ENTITY = "stanford-mercury"  # actual entity of the launched runs (from the run URL)
WANDB_PROJECT = "marin"
USER_AGENT = "mve-h3-readout/1.0"
PRIMARY = "eval/uncheatable_eval/bpb"
DISPLAY_REGEX = "rav_mve_h3"  # matches the 3 launched runs
OUT = Path(__file__).resolve().parent
RUNS = {
    "rav_mve_h3_proposal": "PROPOSAL",
    "rav_mve_h3_olmix": "OLMIX_REUSE",
    "rav_mve_h3_tokprop": "TOKEN_PROPORTIONAL",
}

RUNS_QUERY = """
query Runs($entity: String!, $project: String!, $filters: JSONString, $after: String) {
  project(name: $project, entityName: $entity) {
    runs(filters: $filters, first: 50, after: $after) {
      edges { node { name displayName state createdAt summaryMetrics } }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""


def _auth_header():
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise SystemExit("WANDB_API_KEY not set: source the wandb.env before running (private entity).")
    return "Basic " + base64.b64encode(f"api:{key}".encode()).decode()


def _graphql(query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        WANDB_GRAPHQL_URL,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, "Authorization": _auth_header()},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())
    if payload.get("errors"):
        raise RuntimeError(f"W&B GraphQL errors: {payload['errors']}")
    return payload["data"]


def iter_runs(regex):
    after = None
    while True:
        data = _graphql(
            RUNS_QUERY,
            {
                "entity": WANDB_ENTITY,
                "project": WANDB_PROJECT,
                "filters": json.dumps({"display_name": {"$regex": regex}}),
                "after": after,
            },
        )
        runs = data["project"]["runs"]
        for e in runs["edges"]:
            yield e["node"]
        if not runs["pageInfo"]["hasNextPage"]:
            return
        after = runs["pageInfo"]["endCursor"]


def main():
    prereg = json.loads((OUT / "preregistration.json").read_text())
    anchor_bpb = prereg["anchor"]["realized_bpb"]
    pred = prereg["surrogate_predicted_bpb"]

    realized = {}
    for node in iter_runs(DISPLAY_REGEX):
        name = node["displayName"].rsplit("/", 1)[-1]
        if name not in RUNS:
            continue
        summary = json.loads(node["summaryMetrics"] or "{}")
        realized[RUNS[name]] = {
            "run_name": name,
            "state": node["state"],
            "wandb_id": node["name"],
            "bpb": summary.get(PRIMARY),
            "step": summary.get("_step"),
        }

    print("=== H3 readout ===")
    print(f"{'run':20s} {'state':10s} {'step':>6s} {'realized_bpb':>13s} {'pred_bpb':>9s}")
    for key in ("PROPOSAL", "OLMIX_REUSE", "TOKEN_PROPORTIONAL"):
        r = realized.get(key, {})
        b = r.get("bpb")
        bs = f"{b:.4f}" if b is not None else "PENDING"
        print(
            f"{key:20s} {r.get('state','MISSING'):10s} {r.get('step','')!s:>6s} "
            f"{bs:>13s} {pred[key]['ensemble_mean']:>9.4f}"
        )
    print(
        f"{'ANCHOR (historical)':20s} {'-':10s} {'-':>6s} {anchor_bpb:>13.4f} "
        f"{pred['ANCHOR']['ensemble_mean']:>9.4f}"
    )

    p = realized.get("PROPOSAL", {}).get("bpb")
    o = realized.get("OLMIX_REUSE", {}).get("bpb")
    t = realized.get("TOKEN_PROPORTIONAL", {}).get("bpb")
    print("\n=== pre-registered verdict ===")
    if None in (p, o, t):
        print("INCOMPLETE — not all 3 runs have a final bpb yet.")
        return
    beats_olmix = p < o
    beats_tokprop = p < t
    success = beats_olmix and beats_tokprop
    print(f"PROPOSAL < OLMIX_REUSE:        {p:.4f} < {o:.4f}  -> {beats_olmix}")
    print(f"PROPOSAL < TOKEN_PROPORTIONAL: {p:.4f} < {t:.4f}  -> {beats_tokprop}")
    print(f"SUCCESS (both): {success}")
    print(
        f"regret vs ANCHOR: PROPOSAL {p:.4f} - anchor {anchor_bpb:.4f} = {p - anchor_bpb:+.4f} "
        f"({'beats' if p < anchor_bpb else 'worse than'} anchor)"
    )
    print("\n=== surrogate calibration (predicted vs realized) ===")
    for key in ("PROPOSAL", "OLMIX_REUSE", "TOKEN_PROPORTIONAL"):
        rb = realized[key]["bpb"]
        print(
            f"{key:20s} predicted {pred[key]['ensemble_mean']:.4f}  realized {rb:.4f}  "
            f"err {pred[key]['ensemble_mean'] - rb:+.4f}"
        )


if __name__ == "__main__":
    main()
