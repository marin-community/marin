"""DART Run 9 Phase 2 — apply §1.9.4 hierarchical rule to compiler outputs.

Reads diagnoses_{gpt,gem,cla}.jsonl from dart_run9/, applies L1→L2→L3, writes:
  - rubric_v9_{sid}.json (only for rubric_drift majorities)
  - spec_v9_proposals_{sid}.jsonl (spec_ambiguity escalations, never auto-deploy)
  - examples_v9_{sid}.jsonl (response_interpretation_disagreement)
  - run9_synthesis_summary.json
  - run9_escalation_log.json (rejected proposals)
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DIR = Path("experiments/posttrain/disagreement_primitive")
OUT_DIR = DIR / "dart_run9"
RUBRICS_V1_PATH = DIR / "e8_rubrics_v1.jsonl"
SPEC_PATH = Path("experiments/posttrain/specs/openai_model_spec.jsonl")


def load_jsonl(path):
    return [json.loads(l) for l in path.open() if l.strip()]


def load_diagnoses():
    out = {"gpt": {}, "gem": {}, "cla": {}}
    for nm, fn in [("gpt", "diagnoses_gpt.jsonl"), ("gem", "diagnoses_gem.jsonl"),
                   ("cla", "diagnoses_cla.jsonl")]:
        for r in load_jsonl(OUT_DIR / fn):
            out[nm][r["statement_id"]] = r
    return out


def majority_diag(diags_per_sid):
    """Diagnose vote per §1.9.4 Level 1."""
    valid = {nm: d.get("diagnosis") for nm, d in diags_per_sid.items()
             if d.get("diagnosis") and d["diagnosis"] != "?"}
    if not valid:
        return None, "split", {}
    c = Counter(valid.values())
    most_common, count = c.most_common(1)[0]
    n = len(valid)
    if count == n and n >= 2:  # all agree (treat 2 as consensus if 1 refused)
        tier = "consensus"
    elif count >= max(2, (n + 1) // 2):
        tier = "plurality"
    else:
        tier = "split"
    return most_common, tier, valid


def admissible_edit_types(operative_diag):
    """Per §1.9.4 Level 2 table."""
    table = {
        "rubric_drift": ["rubric_edits"],
        "spec_ambiguity": ["spec_edits"],
        "both": ["rubric_edits", "example_additions"],
        "response_interpretation_disagreement": ["example_additions"],
        "irreducible": [],
    }
    return table.get(operative_diag, [])


def cluster_rubric_edits(diagnoses):
    """L3 per-instance vote on rubric_edits. Returns (adopted, rejected)."""
    by_anchor = defaultdict(dict)
    for nm, d in diagnoses.items():
        for e in d.get("rubric_edits") or []:
            anc = str(e.get("anchor", ""))
            if anc:
                by_anchor[anc][nm] = e
    adopted, rejected = [], []
    priority = {"gem": 3, "cla": 2, "gpt": 1}
    for anc, by_cmp in by_anchor.items():
        if len(by_cmp) < 2:
            cmp_name = next(iter(by_cmp))
            rejected.append({"anchor": anc, "compiler": cmp_name, "reason": "singleton"})
            continue
        best = max(by_cmp.keys(), key=lambda c: priority.get(c, 0))
        edit = by_cmp[best]
        adopted.append({
            "anchor": anc,
            "old": edit.get("old_criterion") or edit.get("old", ""),
            "new": edit.get("new_criterion") or edit.get("new", ""),
            "rationale": edit.get("rationale", ""),
            "source_compiler": best,
            "supporting_compilers": sorted(by_cmp.keys()),
            "confidence": edit.get("confidence"),
        })
    return adopted, rejected


def cluster_spec_edits(diagnoses):
    by_phrase = defaultdict(dict)
    for nm, d in diagnoses.items():
        for e in d.get("spec_edits_for_author_review") or []:
            phrase = (e.get("old_phrase") or "")[:80]
            if phrase:
                by_phrase[phrase][nm] = e
    adopted, rejected = [], []
    priority = {"gem": 3, "cla": 2, "gpt": 1}
    for phrase, by_cmp in by_phrase.items():
        if len(by_cmp) < 2:
            cmp_name = next(iter(by_cmp))
            rejected.append({"old_phrase_prefix": phrase, "compiler": cmp_name, "reason": "singleton"})
            continue
        best = max(by_cmp.keys(), key=lambda c: priority.get(c, 0))
        edit = by_cmp[best]
        adopted.append({
            "old_phrase": edit.get("old_phrase", ""),
            "new_phrase": edit.get("new_phrase", ""),
            "rationale": edit.get("rationale", ""),
            "source_compiler": best,
            "supporting_compilers": sorted(by_cmp.keys()),
            "confidence": edit.get("confidence"),
        })
    return adopted, rejected


def cluster_example_additions(diagnoses):
    """Cluster examples by user_query overlap (60%)."""
    all_props = []
    for nm, d in diagnoses.items():
        for e in d.get("spec_example_additions") or []:
            uq = (e.get("user_query") or "").strip()
            if uq:
                all_props.append((nm, uq, e))
    if not all_props:
        return [], []

    def overlap(a, b):
        n = min(len(a), len(b))
        if n == 0: return 0
        same = sum(1 for i in range(n) if a[i] == b[i])
        return same / max(len(a), len(b))

    clusters = []
    for prop in all_props:
        nm, uq, edit = prop
        placed = False
        for cluster in clusters:
            if overlap(uq[:200].lower(), cluster[0][1][:200].lower()) >= 0.6:
                cluster.append(prop)
                placed = True
                break
        if not placed:
            clusters.append([prop])

    adopted, rejected = [], []
    priority = {"gem": 3, "cla": 2, "gpt": 1}
    for cluster in clusters:
        cmps = {nm for (nm, _, _) in cluster}
        if len(cmps) < 2:
            for nm, uq, edit in cluster:
                rejected.append({"compiler": nm, "user_query_prefix": uq[:80], "reason": "singleton"})
            continue
        best_nm = max(cmps, key=lambda c: priority.get(c, 0))
        edit = next(e for (n, _, e) in cluster if n == best_nm)
        adopted.append({
            "user_query": edit.get("user_query", ""),
            "good_response": edit.get("good_response", ""),
            "bad_response": edit.get("bad_response", ""),
            "description": edit.get("description", ""),
            "target_anchor": edit.get("target_anchor"),
            "rationale": edit.get("rationale", ""),
            "source_compiler": best_nm,
            "supporting_compilers": sorted(cmps),
            "confidence": edit.get("confidence"),
        })
    return adopted, rejected


def main():
    diagnoses = load_diagnoses()
    sids = sorted(set(diagnoses["gpt"]) | set(diagnoses["gem"]) | set(diagnoses["cla"]))

    summary = {"per_statement": {}, "operative_distribution": Counter(),
               "admissible_distribution": Counter()}
    escalation = []

    rubrics_v1 = {r["statement_id"]: r["rubric"]
                  for r in load_jsonl(RUBRICS_V1_PATH) if "error" not in r}
    spec = {r["id"]: r for r in load_jsonl(SPEC_PATH)}

    for sid in sids:
        diags_per_sid = {nm: diagnoses[nm].get(sid, {}) for nm in ("gpt", "gem", "cla")}
        op_diag, tier, votes = majority_diag(diags_per_sid)
        admissible = admissible_edit_types(op_diag) if op_diag else []
        summary["operative_distribution"][op_diag or "split"] += 1

        # L3: per-instance majority within admissible types
        rubric_adopted, rubric_rejected = ([], [])
        spec_adopted, spec_rejected = ([], [])
        examples_adopted, examples_rejected = ([], [])

        if "rubric_edits" in admissible:
            rubric_adopted, rubric_rejected = cluster_rubric_edits(diags_per_sid)
        else:
            # Reject all rubric proposals (out of admissible)
            for nm, d in diags_per_sid.items():
                for e in d.get("rubric_edits") or []:
                    rubric_rejected.append({"compiler": nm, "anchor": e.get("anchor"),
                                            "reason": "edit_type_not_admissible",
                                            "operative_diagnosis": op_diag})
        if "spec_edits" in admissible:
            spec_adopted, spec_rejected = cluster_spec_edits(diags_per_sid)
        else:
            for nm, d in diags_per_sid.items():
                for e in d.get("spec_edits_for_author_review") or []:
                    spec_rejected.append({"compiler": nm,
                                          "old_phrase_prefix": (e.get("old_phrase") or "")[:80],
                                          "reason": "edit_type_not_admissible",
                                          "operative_diagnosis": op_diag})
        if "example_additions" in admissible:
            examples_adopted, examples_rejected = cluster_example_additions(diags_per_sid)
        else:
            for nm, d in diags_per_sid.items():
                for e in d.get("spec_example_additions") or []:
                    examples_rejected.append({"compiler": nm,
                                              "user_query_prefix": (e.get("user_query") or "")[:80],
                                              "reason": "edit_type_not_admissible",
                                              "operative_diagnosis": op_diag})

        summary["per_statement"][sid] = {
            "operative_diagnosis": op_diag,
            "diagnosis_tier": tier,
            "votes": votes,
            "admissible_edit_types": admissible,
            "n_rubric_adopted": len(rubric_adopted),
            "n_spec_adopted": len(spec_adopted),
            "n_examples_adopted": len(examples_adopted),
        }
        for et in admissible:
            summary["admissible_distribution"][et] += 1

        # Write per-statement artifacts
        sid_dir = OUT_DIR / sid
        sid_dir.mkdir(parents=True, exist_ok=True)
        if rubric_adopted:
            # Apply to v1 rubric
            v9 = json.loads(json.dumps(rubrics_v1.get(sid, {})))
            for e in rubric_adopted:
                if e["anchor"] in v9.get("anchors", {}):
                    v9["anchors"][e["anchor"]]["criterion"] = e["new"]
            (sid_dir / "rubric_v9.json").write_text(json.dumps(v9, indent=2))
            (sid_dir / "rubric_edits_v9.jsonl").write_text(
                "\n".join(json.dumps(e) for e in rubric_adopted))
        if spec_adopted:
            (sid_dir / "spec_proposals_v9.jsonl").write_text(
                "\n".join(json.dumps(e) for e in spec_adopted))
        if examples_adopted:
            (sid_dir / "examples_v9.jsonl").write_text(
                "\n".join(json.dumps(e) for e in examples_adopted))
            # Also produce a modified spec with examples appended
            new_spec = json.loads(json.dumps(spec[sid]))
            new_spec.setdefault("metadata", {}).setdefault("examples", [])
            for e in examples_adopted:
                new_spec["metadata"]["examples"].append({
                    "user_query": e["user_query"],
                    "good_response": e["good_response"],
                    "bad_response": e["bad_response"],
                    "description": e.get("description", ""),
                })
            (sid_dir / "spec_with_examples_v9.json").write_text(json.dumps(new_spec, indent=2))

        escalation.append({
            "statement_id": sid,
            "operative_diagnosis": op_diag,
            "tier": tier,
            "votes": votes,
            "rejected_rubric": rubric_rejected,
            "rejected_spec": spec_rejected,
            "rejected_examples": examples_rejected,
        })

    # Persist
    summary["operative_distribution"] = dict(summary["operative_distribution"])
    summary["admissible_distribution"] = dict(summary["admissible_distribution"])
    (OUT_DIR / "run9_synthesis_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "run9_escalation_log.json").write_text(json.dumps(escalation, indent=2))

    # Print
    print("=== Run 9 Phase 2 synthesis ===")
    print(f"Operative diagnosis distribution: {summary['operative_distribution']}")
    print(f"Admissible edit types (statement-instances): {summary['admissible_distribution']}")
    print()
    print(f"{'sid':45s} {'operative':40s} tier      adopted")
    print("-"*125)
    for sid in sids:
        s = summary["per_statement"][sid]
        adopted_str = f"R={s['n_rubric_adopted']} S={s['n_spec_adopted']} E={s['n_examples_adopted']}"
        print(f"{sid:45s} {str(s['operative_diagnosis']):40s} {s['diagnosis_tier']:10s} {adopted_str}")


if __name__ == "__main__":
    main()
