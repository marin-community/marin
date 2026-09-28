"""Merge the four 1e21 accuracy reports and native OlmoBaseEval Easy BPB summaries (Proportional, UniMax-8, Olmix, MARINER).

Olmix's native entry is built from its audited Table-9 result the same way MARINER's was (2026-09-22), and the MARINER
entry is checked against its own audited result so both follow one construction.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPO = PROJECT.parents[2]
AUDITED = REPO / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/frozen_scaling_update_20260913"
PREVIOUS = PROJECT / "mariner_t9_1e21/accuracy_vs_bpb"


def native_entry(path: Path) -> dict:
    result = json.loads(path.read_text())
    summary = {f"olmo_base_easy/table9/{name}/bpb": value for name, value in result["table9_components"].items()}
    summary["olmo_base_easy/table9_macro_bpb"] = result["table9_macro_bpb"]
    return {"id": result["name"], "state": "audited", "config": {"checkpoint_path": result["checkpoint_path"]}, "summary": summary}


def main() -> None:
    native = json.loads((PREVIOUS / "native_summaries_with_mariner.json").read_text())
    rebuilt = native_entry(AUDITED / "mariner/lwspu_t9_snc_cap08_1e21_seed662005-e8e9d7_table9.json")
    assert rebuilt["summary"] == native["mariner"]["summary"], "MARINER's native entry no longer matches its audited result"
    native["olmix"] = native_entry(AUDITED / "matched_olmix/olmixq_t9_kl0p005_cap04_1e21_seed662005-3f95f2_table9.json")
    coverage = json.loads((PREVIOUS / "coverage_merged.json").read_text())
    coverage += json.loads((PROJECT / "olmix_t9_1e21_east1/coverage/coverage.json").read_text())
    assert len(coverage) == 4 and len({r["checkpoint"]["name"] for r in coverage}) == 4
    (HERE / "native_summaries.json").write_text(json.dumps(native, indent=1) + "\n")
    (HERE / "coverage_merged.json").write_text(json.dumps(coverage, indent=1) + "\n")
    print("merged:", sorted(native), [r["checkpoint"]["name"] for r in coverage])


if __name__ == "__main__":
    main()
