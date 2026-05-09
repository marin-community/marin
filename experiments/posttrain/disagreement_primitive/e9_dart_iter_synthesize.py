"""DART iterative validation — Phase 1 synthesis.

Reads the 3 compiler output files (gpt/gemini/claude diagnoses), applies the
§1.7 majority-vote rule, and writes per-statement v{N+1} rubric and spec text
plus an updated history.json.

Round-1 mode (default): reads the existing Run-1/2/3 jsonl files; writes
dart_iteration/{sid}/rubric_v2.json + spec_v2.txt + history.json (1 entry).

Round-N>1 mode: reads dart_iteration/{sid}/round_{N}_compile/{compiler}.json
(produced by e9_dart_iter_compile.py, which incorporates cumulative history),
applies majority vote, writes rubric_v{N+1}.json + spec_v{N+1}.txt and appends
to history.json.

Usage:
    python e9_dart_iter_synthesize.py --round 1
    python e9_dart_iter_synthesize.py --round 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import SPEC_PATH

DIR = Path("experiments/posttrain/disagreement_primitive")
ITER_DIR = DIR / "dart_iteration"
RUBRICS_V1_PATH = DIR / "e8_rubrics_v1.jsonl"
ALIGNED_PAIRS_3WAY = DIR / "dart_aligned_pairs_3way.jsonl"

T1 = 0.5  # bucket threshold
EPSILON_IMPROVING = 0.05  # min delta-alpha to qualify as IMPROVING


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def load_round1_compiler_outputs() -> tuple[dict, dict, dict]:
    gpt = {r["statement_id"]: r for r in load_jsonl(DIR / "dart_diagnoses.jsonl") if "error" not in r}
    gem = {r["statement_id"]: r for r in load_jsonl(DIR / "dart_diagnoses_gemini.jsonl") if "error" not in r}
    cla = {r["statement_id"]: r for r in load_jsonl(DIR / "dart_diagnoses_claude.jsonl") if "error" not in r}
    return gpt, gem, cla


def load_pair_directions(sid: str) -> dict:
    """Load per-anchor direction classifications from dart_aligned_pairs_3way.jsonl."""
    for r in load_jsonl(ALIGNED_PAIRS_3WAY):
        if r["statement_id"] != sid:
            continue
        # build anchor-level direction maps per pair
        out = {}  # (frozenset({a, b}), kind, anchor_or_oldphrase) -> direction
        for cp_name, alignment in r["pair_alignments"].items():
            ab = frozenset(cp_name.split("_"))
            for ep in alignment.get("rubric_pairs", []):
                anchor = str(ep.get("anchor", ""))
                out[(ab, "rubric", anchor)] = ep.get("ensemble", "?")
            for ep in alignment.get("spec_pairs", []):
                phrase = (ep.get("old_phrase") or "")[:200]
                out[(ab, "spec", phrase)] = ep.get("ensemble", "?")
        return out
    return {}


def load_v1_rubric(sid: str) -> dict:
    for r in load_jsonl(RUBRICS_V1_PATH):
        if r["statement_id"] == sid:
            return r["rubric"]
    raise KeyError(f"v1 rubric for {sid} not found")


def load_spec_text(sid: str) -> str:
    for r in load_jsonl(SPEC_PATH):
        if r["id"] == sid:
            return r["text"]
    raise KeyError(f"spec text for {sid} not found")


def majority_rubric_edits_for_statement(
    sid: str, gpt: dict, gem: dict, cla: dict, pair_dir: dict
) -> tuple[list[dict], list[dict]]:
    """Return (adopted_edits, rejected_edits_with_reason).

    Adopted: anchors where ≥2 compilers proposed AND ≥1 pair is same_direction.
    Rejected: singleton or all-disputed or any opposite_direction.
    """
    # Build anchor → {compiler: edit_dict}
    anchor_to_edits: dict[str, dict[str, dict]] = defaultdict(dict)
    for nm, src in [("gpt", gpt), ("gem", gem), ("cla", cla)]:
        if sid not in src:
            continue
        for e in src[sid].get("rubric_edits") or []:
            anchor = str(e.get("anchor", ""))
            if anchor:
                anchor_to_edits[anchor][nm] = e

    adopted = []
    rejected = []
    for anchor, by_cmp in anchor_to_edits.items():
        if len(by_cmp) < 2:
            cmp_name = next(iter(by_cmp))
            rejected.append({"anchor": anchor, "compiler": cmp_name, "reason": "singleton"})
            continue

        # Check pair directions
        cmps = sorted(by_cmp.keys())
        dirs = {}
        for i, a in enumerate(cmps):
            for b in cmps[i + 1 :]:
                d = pair_dir.get((frozenset((a, b)), "rubric", anchor), "?")
                dirs[(a, b)] = d

        if any(d == "opposite_direction" for d in dirs.values()):
            rejected.append({
                "anchor": anchor,
                "compilers": cmps,
                "reason": "opposite_direction",
                "pair_directions": {f"{a}-{b}": d for (a, b), d in dirs.items()},
            })
            continue

        same_pairs = [(a, b) for (a, b), d in dirs.items() if d == "same_direction"]
        if not same_pairs:
            rejected.append({
                "anchor": anchor,
                "compilers": cmps,
                "reason": "all_disputed",
                "pair_directions": {f"{a}-{b}": d for (a, b), d in dirs.items()},
            })
            continue

        # Pick the edit text from the compiler that appears in the most same_direction pairs.
        # Prefer Gemini > Claude > GPT on ties (Gemini and Claude are the consensus pole;
        # GPT was the outlier in Run 3).
        compiler_score: Counter[str] = Counter()
        for a, b in same_pairs:
            compiler_score[a] += 1
            compiler_score[b] += 1
        priority = {"gem": 3, "cla": 2, "gpt": 1}
        best_cmp = max(compiler_score, key=lambda c: (compiler_score[c], priority[c]))
        edit = by_cmp[best_cmp]
        adopted.append({
            "anchor": anchor,
            "old": edit.get("old_criterion") or edit.get("old", ""),
            "new": edit.get("new_criterion") or edit.get("new", ""),
            "rationale": edit.get("rationale", ""),
            "confidence": edit.get("confidence"),
            "source_compiler": best_cmp,
            "supporting_compilers": cmps,
            "pair_directions": {f"{a}-{b}": d for (a, b), d in dirs.items()},
        })

    return adopted, rejected


def majority_spec_edits_for_statement(sid: str) -> tuple[list[dict], list[dict]]:
    """Adopt spec edits using dart_aligned_pairs_3way.jsonl as the authoritative source
    of cross-compiler edit pairings (already aligned via 60% old_phrase overlap).

    Rule: for each statement, iterate spec_pairs across all 3 compiler-pair alignments.
    A spec edit cluster is identified by its primary old_phrase (first 80 chars).
    Adopt if at least one same_direction pair exists AND no opposite_direction pair
    exists for that cluster.
    """
    # Load this statement's row from aligned_pairs
    row = None
    for r in load_jsonl(ALIGNED_PAIRS_3WAY):
        if r["statement_id"] == sid:
            row = r
            break
    if row is None:
        return [], []

    # Collect spec_pair info: cluster by old_phrase prefix
    # Each spec_pair has fields like 'gpt_edit', 'gem_edit', 'classification_*', 'ensemble'
    # The keys gpt_edit / gem_edit / cla_edit are statically named even though the pair is
    # only 2-of-3 — the inactive compiler's *_edit field will be missing.
    cluster_to_evidence: dict[str, dict] = defaultdict(lambda: {
        "edits_by_compiler": {},
        "pair_classifications": [],
    })

    for cp_name, alignment in row["pair_alignments"].items():
        for ep in alignment.get("spec_pairs", []):
            ens = ep.get("ensemble", "?")
            # Pull whichever *_edit dicts are present
            present_edits = {
                k.replace("_edit", ""): v for k, v in ep.items()
                if k.endswith("_edit") and isinstance(v, dict) and v.get("old_phrase")
            }
            if not present_edits:
                continue
            # Cluster key: first present old_phrase (first 80 chars)
            first_op = next(iter(present_edits.values())).get("old_phrase", "")[:80]
            if not first_op:
                continue
            ev = cluster_to_evidence[first_op]
            for cmp_name, edit_dict in present_edits.items():
                ev["edits_by_compiler"][cmp_name] = edit_dict
            ev["pair_classifications"].append({"pair": cp_name, "ensemble": ens})

    adopted = []
    rejected = []
    for cluster_key, ev in cluster_to_evidence.items():
        ensembles = [pc["ensemble"] for pc in ev["pair_classifications"]]
        cmps = sorted(ev["edits_by_compiler"].keys())

        if len(cmps) < 2:
            rejected.append({
                "old_phrase_prefix": cluster_key,
                "compilers": cmps,
                "reason": "only_one_compiler_in_pairings",
                "ensembles": ensembles,
            })
            continue

        if any(e == "opposite_direction" for e in ensembles):
            rejected.append({
                "old_phrase_prefix": cluster_key,
                "compilers": cmps,
                "reason": "opposite_direction",
                "ensembles": ensembles,
            })
            continue

        if not any(e == "same_direction" for e in ensembles):
            rejected.append({
                "old_phrase_prefix": cluster_key,
                "compilers": cmps,
                "reason": "all_disputed_or_different_scope",
                "ensembles": ensembles,
            })
            continue

        priority = {"gem": 3, "cla": 2, "gpt": 1}
        best_cmp = max(cmps, key=lambda c: priority.get(c, 0))
        edit = ev["edits_by_compiler"][best_cmp]
        adopted.append({
            "old_phrase": edit.get("old_phrase", ""),
            "new_phrase": edit.get("new_phrase", ""),
            "rationale": edit.get("rationale", ""),
            "confidence": edit.get("confidence"),
            "source_compiler": best_cmp,
            "supporting_compilers": cmps,
            "pair_ensembles": ensembles,
        })

    return adopted, rejected


def majority_diagnosis_for_statement(sid: str, gpt: dict, gem: dict, cla: dict) -> tuple[str, str, dict]:
    """Return (majority_diagnosis, diag_tier, votes_breakdown)."""
    diags = []
    votes = {}
    for nm, src in [("gpt", gpt), ("gem", gem), ("cla", cla)]:
        if sid in src:
            d = src[sid].get("diagnosis", "?")
            diags.append(d)
            votes[nm] = d
    c = Counter(diags)
    most_common, count = c.most_common(1)[0]
    if count == 3:
        tier = "consensus"
    elif count == 2:
        tier = "plurality"
    else:
        tier = "split"
    return most_common, tier, votes


def apply_rubric_edits(rubric: dict, adopted: list[dict]) -> dict:
    """Return new rubric dict with adopted criterion replacements applied."""
    new = json.loads(json.dumps(rubric))  # deep copy
    for e in adopted:
        anchor = e["anchor"]
        new_text = e["new"]
        if anchor in new["anchors"]:
            new["anchors"][anchor]["criterion"] = new_text
    return new


def apply_spec_edits(spec_text: str, adopted: list[dict]) -> str:
    """Return new spec text with adopted phrase replacements applied.

    Replaces each old_phrase verbatim. If old_phrase is not a substring, leaves it
    unchanged (we'll record this as a validation warning in history.json).
    """
    new = spec_text
    for e in adopted:
        old = e["old_phrase"]
        nu = e["new_phrase"]
        if old and old in new:
            new = new.replace(old, nu)
    return new


def synthesize_round1(verbose: bool = True) -> dict:
    """Apply majority vote to Run 1/2/3 outputs to produce v2 per statement."""
    gpt, gem, cla = load_round1_compiler_outputs()
    common = sorted(set(gpt) & set(gem) & set(cla))
    summary = {"round": 1, "statements_processed": 0, "per_statement": {}}

    for sid in common:
        sid_dir = ITER_DIR / sid
        sid_dir.mkdir(parents=True, exist_ok=True)

        pair_dir = load_pair_directions(sid)
        rubric_v1 = load_v1_rubric(sid)
        spec_v1 = load_spec_text(sid)

        adopted_rubric, rejected_rubric = majority_rubric_edits_for_statement(sid, gpt, gem, cla, pair_dir)
        adopted_spec, rejected_spec = majority_spec_edits_for_statement(sid)
        maj_diag, diag_tier, votes = majority_diagnosis_for_statement(sid, gpt, gem, cla)

        # Apply edits
        rubric_v2 = apply_rubric_edits(rubric_v1, adopted_rubric)
        spec_v2 = apply_spec_edits(spec_v1, adopted_spec)

        # Track which spec edits actually applied (substring found)
        applied_count = sum(1 for e in adopted_spec if e["old_phrase"] in spec_v1)
        unmatched = [e for e in adopted_spec if e["old_phrase"] not in spec_v1]

        # Write rubric_v2.json
        (sid_dir / "rubric_v2.json").write_text(json.dumps(rubric_v2, indent=2))
        # Write spec_v2.txt
        (sid_dir / "spec_v2.txt").write_text(spec_v2)
        # Write spec_v1 baseline alongside for diff
        (sid_dir / "spec_v1.txt").write_text(spec_v1)
        # Save the original v1 rubric for reference
        (sid_dir / "rubric_v1.json").write_text(json.dumps(rubric_v1, indent=2))

        # Initial history entry — α_before/after will be filled in by analyze step
        history_entry = {
            "round": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rubric_state_at_start": "v1",
            "spec_state_at_start": "v1",
            "round_diagnosis_majority": maj_diag,
            "round_diagnosis_tier": diag_tier,
            "diagnosis_votes": votes,
            "rubric_edits_adopted": adopted_rubric,
            "rubric_edits_rejected": rejected_rubric,
            "spec_edits_adopted": adopted_spec,
            "spec_edits_rejected": rejected_spec,
            "spec_edits_applied_count": applied_count,
            "spec_edits_unmatched_count": len(unmatched),
            "alpha_before_round": None,  # filled by analyze step
            "alpha_after_round": None,
            "delta_alpha": None,
            "delta_pwv_top10_pct_drop": None,
            "verdict": "pending_judging",
        }
        history_path = sid_dir / "history.json"
        history_path.write_text(json.dumps([history_entry], indent=2))

        summary["per_statement"][sid] = {
            "rubric_edits_adopted": len(adopted_rubric),
            "rubric_edits_rejected": len(rejected_rubric),
            "spec_edits_adopted": len(adopted_spec),
            "spec_edits_rejected": len(rejected_spec),
            "spec_edits_applied": applied_count,
            "majority_diagnosis": maj_diag,
            "diag_tier": diag_tier,
        }
        summary["statements_processed"] += 1

        if verbose:
            print(
                f"  {sid:35s} maj={maj_diag:18s} tier={diag_tier:10s} "
                f"rubric: {len(adopted_rubric)}adopt/{len(rejected_rubric)}reject  "
                f"spec: {len(adopted_spec)}adopt({applied_count}applied)/{len(rejected_spec)}reject"
            )

    summary_path = ITER_DIR / f"round_{summary['round']}_synthesis_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"\nWrote {summary_path}")
        print(f"Total statements: {summary['statements_processed']}")
        total_rubric = sum(s["rubric_edits_adopted"] for s in summary["per_statement"].values())
        total_spec = sum(s["spec_edits_adopted"] for s in summary["per_statement"].values())
        total_applied = sum(s["spec_edits_applied"] for s in summary["per_statement"].values())
        print(f"Total rubric edits adopted: {total_rubric}")
        print(f"Total spec edits adopted: {total_spec} ({total_applied} verbatim-applied to spec text)")
    return summary


def synthesize_round_n(round_n: int, verbose: bool = True) -> dict:
    """Apply majority vote to Round-N>=2 compiler outputs (from e9_dart_iter_round_n_compile.py).

    Reads dart_iteration/dart_diagnoses{,_gemini,_claude}_round_{N}.jsonl.
    Skips the aligned-pairs disagreement classifier — uses a simpler N-of-3
    rubric anchor concurrence rule: if ≥2 compilers proposed an edit for the
    same anchor, adopt the edit text from the highest-priority compiler (gem >
    cla > gpt). Same for spec edits, clustered by old_phrase[:80].
    """
    gpt = {r["statement_id"]: r for r in load_jsonl(ITER_DIR / f"dart_diagnoses_round_{round_n}.jsonl")
           if "error" not in r}
    gem = {r["statement_id"]: r for r in load_jsonl(ITER_DIR / f"dart_diagnoses_gemini_round_{round_n}.jsonl")
           if "error" not in r}
    cla = {r["statement_id"]: r for r in load_jsonl(ITER_DIR / f"dart_diagnoses_claude_round_{round_n}.jsonl")
           if "error" not in r}
    common = sorted(set(gpt) & set(gem) & set(cla))
    print(f"Round {round_n}: {len(common)} statements with all 3 compiler outputs")

    summary = {"round": round_n, "statements_processed": 0, "per_statement": {}}

    for sid in common:
        sid_dir = ITER_DIR / sid

        # Load current state (=v_N)
        rubric_v_n = json.loads((sid_dir / f"rubric_v{round_n}.json").read_text())
        spec_v_n = (sid_dir / f"spec_v{round_n}.txt").read_text()

        # Cluster rubric edits by anchor across all 3 compilers
        anchor_to_edits: dict[str, dict[str, dict]] = defaultdict(dict)
        for nm, src in [("gpt", gpt), ("gem", gem), ("cla", cla)]:
            if sid not in src:
                continue
            for e in src[sid].get("rubric_edits") or []:
                anchor = str(e.get("anchor", ""))
                if anchor:
                    anchor_to_edits[anchor][nm] = e

        adopted_rubric = []
        rejected_rubric = []
        for anchor, by_cmp in anchor_to_edits.items():
            if len(by_cmp) < 2:
                rejected_rubric.append({"anchor": anchor, "compiler": next(iter(by_cmp)), "reason": "singleton"})
                continue
            priority = {"gem": 3, "cla": 2, "gpt": 1}
            best_cmp = max(by_cmp, key=lambda c: priority[c])
            edit = by_cmp[best_cmp]
            adopted_rubric.append({
                "anchor": anchor,
                "old": edit.get("old_criterion") or edit.get("old", ""),
                "new": edit.get("new_criterion") or edit.get("new", ""),
                "rationale": edit.get("rationale", ""),
                "source_compiler": best_cmp,
                "supporting_compilers": sorted(by_cmp.keys()),
            })

        # Cluster spec edits by old_phrase[:80]
        phrase_to_edits: dict[str, dict[str, dict]] = defaultdict(dict)
        for nm, src in [("gpt", gpt), ("gem", gem), ("cla", cla)]:
            if sid not in src:
                continue
            for e in src[sid].get("spec_edits_for_author_review") or []:
                phrase = (e.get("old_phrase") or "")[:80]
                if phrase:
                    phrase_to_edits[phrase][nm] = e

        adopted_spec = []
        rejected_spec = []
        for phrase, by_cmp in phrase_to_edits.items():
            if len(by_cmp) < 2:
                rejected_spec.append({"old_phrase_prefix": phrase, "compiler": next(iter(by_cmp)), "reason": "singleton"})
                continue
            priority = {"gem": 3, "cla": 2, "gpt": 1}
            best_cmp = max(by_cmp, key=lambda c: priority[c])
            edit = by_cmp[best_cmp]
            adopted_spec.append({
                "old_phrase": edit.get("old_phrase", ""),
                "new_phrase": edit.get("new_phrase", ""),
                "rationale": edit.get("rationale", ""),
                "source_compiler": best_cmp,
                "supporting_compilers": sorted(by_cmp.keys()),
            })

        # Apply
        rubric_v_next = apply_rubric_edits(rubric_v_n, adopted_rubric)
        spec_v_next = apply_spec_edits(spec_v_n, adopted_spec)

        # Track verbatim spec application
        applied_count = sum(1 for e in adopted_spec if e["old_phrase"] in spec_v_n)

        # Majority diagnosis at round N (votes only on these compiler outputs)
        diags = [gpt[sid].get("diagnosis"), gem[sid].get("diagnosis"), cla[sid].get("diagnosis")]
        c = Counter(diags)
        most_common, count = c.most_common(1)[0]
        diag_tier = "consensus" if count == 3 else ("plurality" if count == 2 else "split")

        # Write v_{N+1} files
        next_n = round_n + 1
        (sid_dir / f"rubric_v{next_n}.json").write_text(json.dumps(rubric_v_next, indent=2))
        (sid_dir / f"spec_v{next_n}.txt").write_text(spec_v_next)

        # Append round N entry to history.json
        history_path = sid_dir / "history.json"
        history = json.loads(history_path.read_text())
        new_entry = {
            "round": round_n,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rubric_state_at_start": f"v{round_n}",
            "spec_state_at_start": f"v{round_n}",
            "round_diagnosis_majority": most_common,
            "round_diagnosis_tier": diag_tier,
            "diagnosis_votes": {"gpt": diags[0], "gem": diags[1], "cla": diags[2]},
            "rubric_edits_adopted": adopted_rubric,
            "rubric_edits_rejected": rejected_rubric,
            "spec_edits_adopted": adopted_spec,
            "spec_edits_rejected": rejected_spec,
            "spec_edits_applied_count": applied_count,
            "alpha_before_round": history[-1].get("alpha_after_round"),  # carry forward
            "alpha_after_round": None,
            "delta_alpha": None,
            "delta_pwv_top10_pct_drop": None,
            "verdict": "pending_judging",
        }
        # Pad with placeholder entries for any rounds the statement skipped (e.g. CONVERGED at round 1)
        while len(history) < round_n - 1:
            history.append({"round": len(history) + 1, "verdict": "skipped"})
        history.append(new_entry)
        history_path.write_text(json.dumps(history, indent=2))

        summary["per_statement"][sid] = {
            "rubric_edits_adopted": len(adopted_rubric),
            "spec_edits_adopted": len(adopted_spec),
            "spec_edits_applied": applied_count,
            "majority_diagnosis": most_common,
            "diag_tier": diag_tier,
        }
        summary["statements_processed"] += 1
        if verbose:
            print(f"  {sid:35s} maj={most_common:18s} tier={diag_tier:10s}  "
                  f"rubric: {len(adopted_rubric)}adopt  spec: {len(adopted_spec)}adopt({applied_count}applied)")

    summary_path = ITER_DIR / f"round_{round_n}_synthesis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"\nWrote {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()

    if args.round == 1:
        synthesize_round1()
    else:
        synthesize_round_n(args.round)


if __name__ == "__main__":
    main()
