# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E9 apply stage: build rolling spec_v<N>.jsonl from accepted candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import SPEC_PATH, write_jsonl
from e9_repair_common import REPAIR_DIR, load_jsonl, load_spec


def load_verdicts(round_id: int) -> list[dict[str, Any]]:
    path = REPAIR_DIR / f"round_{round_id}" / "verdicts.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing verdicts: {path}")
    return load_jsonl(path)


def pick_best_per_statement(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passed = [v for v in verdicts if ((v.get("gate") or {}).get("passed") is True)]
    by_statement: dict[str, list[dict[str, Any]]] = {}
    for verdict in passed:
        by_statement.setdefault(verdict["statement_id"], []).append(verdict)
    selected = []
    for _statement_id, rows in sorted(by_statement.items()):

        def score(row: dict[str, Any]) -> tuple[float, float, float]:
            var_a = row.get("var_A") or {}
            cross = row.get("cross") or {}
            return (
                float(var_a.get("delta_kappa_held_out_var_A") or -999),
                float(cross.get("delta_kappa_phase_4") or -999),
                float(cross.get("delta_kappa_full_spec") or -999),
            )

        selected.append(max(rows, key=score))
    return selected


def candidate_statement_path(verdict: dict[str, Any]) -> Path:
    return (
        REPAIR_DIR / f"round_{verdict['round']}" / verdict["statement_id"] / verdict["candidate_id"] / "statement.jsonl"
    )


def load_candidate_statement(verdict: dict[str, Any]) -> dict[str, Any]:
    path = candidate_statement_path(verdict)
    rows = load_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, got {len(rows)}")
    return rows[0]


def base_spec_for_round(round_id: int) -> Path:
    if round_id <= 1:
        return SPEC_PATH
    prior = Path(f"experiments/posttrain/specs/openai_model_spec_v{round_id - 1}.jsonl")
    if prior.exists():
        return prior
    return SPEC_PATH


def build_spec(round_id: int, selected: list[dict[str, Any]], force: bool) -> Path:
    base_path = base_spec_for_round(round_id)
    out_path = Path(f"experiments/posttrain/specs/openai_model_spec_v{round_id}.jsonl")
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} exists; pass --force to overwrite")
    replacements = {v["statement_id"]: load_candidate_statement(v) for v in selected}
    rows = []
    replaced = set()
    for row in load_spec(base_path):
        sid = row["id"]
        if sid in replacements:
            rows.append(replacements[sid])
            replaced.add(sid)
        else:
            rows.append(row)
    missing = sorted(set(replacements) - replaced)
    if missing:
        raise ValueError(f"selected statements not found in base spec {base_path}: {missing}")
    write_jsonl(rows, out_path)
    return out_path


def write_apply_log(
    round_id: int, selected: list[dict[str, Any]], all_verdicts: list[dict[str, Any]], spec_path: Path
) -> Path:
    out_path = REPAIR_DIR / f"round_{round_id}" / "apply_log.jsonl"
    rows = []
    selected_keys = {(v["statement_id"], v["candidate_id"]) for v in selected}
    for verdict in all_verdicts:
        key = (verdict["statement_id"], verdict["candidate_id"])
        rows.append(
            {
                "round": round_id,
                "statement_id": verdict["statement_id"],
                "candidate_id": verdict["candidate_id"],
                "applied": key in selected_keys,
                "gate_passed": (verdict.get("gate") or {}).get("passed") is True,
                "gate_reasons": (verdict.get("gate") or {}).get("reasons") or [],
                "delta_kappa_held_out_var_A": (verdict.get("var_A") or {}).get("delta_kappa_held_out_var_A"),
                "delta_kappa_compiler_input_var_A": (verdict.get("var_A") or {}).get("delta_kappa_compiler_input_var_A"),
                "delta_kappa_phase_4": (
                    (verdict.get("cross") or {}).get("delta_kappa_phase_4") if verdict.get("cross") else None
                ),
                "delta_kappa_full_spec": (
                    (verdict.get("cross") or {}).get("delta_kappa_full_spec") if verdict.get("cross") else None
                ),
                "spec_version_path": str(spec_path) if key in selected_keys else None,
            }
        )
    write_jsonl(rows, out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    verdicts = load_verdicts(args.round)
    selected = pick_best_per_statement(verdicts)
    print(f"round={args.round} verdicts={len(verdicts)} selected={len(selected)}")
    for row in selected:
        delta = (row.get("var_A") or {}).get("delta_kappa_held_out_var_A")
        print(f"  apply {row['statement_id']} / {row['candidate_id']} delta_var_A={delta}")
    spec_path = build_spec(args.round, selected, args.force)
    apply_log = write_apply_log(args.round, selected, verdicts, spec_path)
    print(f"wrote {spec_path}")
    print(f"wrote {apply_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
