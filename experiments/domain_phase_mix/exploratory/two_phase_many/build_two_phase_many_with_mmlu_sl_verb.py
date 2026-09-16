# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a merged original-swarm CSV including MMLU-SL-Verb rerun metrics."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_CSV = SCRIPT_DIR / "two_phase_many.csv"
OUTPUT_CSV = SCRIPT_DIR / "two_phase_many_with_mmlu_sl_verb.csv"
SL_VERB_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_qsplit240_mmlu_sl_verb_rerun/collect_results-ef2602/results.csv"
)

SL_VERB_COLUMNS = [
    "run_name",
    "lm_eval/mmlu_sl_verb_5shot/acc",
    "lm_eval/mmlu_sl_verb_5shot/acc_norm",
    "lm_eval/mmlu_sl_verb_5shot/bpb",
    "lm_eval/mmlu_sl_verb_5shot/choice_logprob",
    "lm_eval/mmlu_sl_verb_5shot/choice_logprob_norm",
    "lm_eval/mmlu_sl_verb_5shot/choice_prob_norm",
    "lm_eval/mmlu_sl_verb_5shot/logprob",
]


def _read_gcs_csv(uri: str) -> pd.DataFrame:
    data = subprocess.check_output(["gsutil", "cat", uri], text=True)
    return pd.read_csv(io.StringIO(data))


def main() -> None:
    base = pd.read_csv(SOURCE_CSV)
    sl_verb = _read_gcs_csv(SL_VERB_RESULTS_URI)[SL_VERB_COLUMNS].copy()

    merged = base.merge(sl_verb, on="run_name", how="left", validate="one_to_one")
    merged.to_csv(OUTPUT_CSV, index=False)

    missing = merged["lm_eval/mmlu_sl_verb_5shot/bpb"].isna()
    missing_runs = merged.loc[missing, "run_name"].tolist()

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Rows: {len(merged)}")
    print(f"SL-Verb non-null rows: {int((~missing).sum())}")
    print(f"SL-Verb missing rows: {missing_runs}")


if __name__ == "__main__":
    main()
