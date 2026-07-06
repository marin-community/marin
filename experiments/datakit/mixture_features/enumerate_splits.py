# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow"]
# ///
"""H2b feasibility enumeration (spec: `enumerate_splits`).

Reads `runs.parquet` (built by `swarm_runs.py`) and writes `feasibility.parquet`: for every
domain x scale x phase reducer x threshold pair, the train/test membership counts of the
held-out-dose split, without reference to outcomes. Also prints the pre-registration table at
the spec defaults (train_max_dose 0.02, test_min_dose 0.30, MAX reducer, n_train >= 60,
n_test >= 20) and a per-domain dose-quantile recalibration table.

Note: the qsplit240 swarm was sampled with SamplingStrategy.DIRICHLET only (see
two_phase_dolma3_dolmino_top_level.py SAMPLING_PARAMS on the swarm branch) — the vertex-biased
strategy was never enabled, so no vertex runs exist and high-dose runs are scarce by
construction. This table reports the numbers faithfully.
"""

import argparse
import logging
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "scratch" / "mixture_features"

TRAIN_MAX_DOSES = (0.01, 0.02, 0.05)
TEST_MIN_DOSES = (0.2, 0.3)
SPEC_TRAIN_MAX_DOSE = 0.02
SPEC_TEST_MIN_DOSE = 0.30
SPEC_MIN_TRAIN = 60
SPEC_MIN_TEST = 20
RELAXED_MIN_TEST = 15
RECALIBRATION_TEST_MIN_DOSE_GRID = np.round(np.arange(0.02, 0.42, 0.02), 3)


class PhaseReducer(StrEnum):
    MAX = "max"
    TOKEN_WEIGHTED = "token_weighted"


def domain_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c[len("phase_0_") :] for c in df.columns if c.startswith("phase_0_") and c != "phase_0_tokens")


def reduced_doses(group: pd.DataFrame, domains: list[str], reducer: PhaseReducer) -> np.ndarray:
    """Per-(run, domain) dose under the given phase reducer; shape (n_runs, n_domains)."""
    w0 = group[[f"phase_0_{d}" for d in domains]].to_numpy(dtype=float)
    w1 = group[[f"phase_1_{d}" for d in domains]].to_numpy(dtype=float)
    if reducer is PhaseReducer.MAX:
        return np.maximum(w0, w1)
    t0 = group["phase_0_tokens"].to_numpy(dtype=float)
    t1 = group["phase_1_tokens"].to_numpy(dtype=float)
    return (w0 * t0[:, None] + w1 * t1[:, None]) / (t0 + t1)[:, None]


def enumerate_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Feasibility table over domain x scale x reducer x (train_max_dose, test_min_dose)."""
    domains = domain_columns(df)
    rows = []
    for scale, group in df.groupby("scale", sort=True):
        for reducer in PhaseReducer:
            doses = reduced_doses(group, domains, reducer)
            for j, domain in enumerate(domains):
                dose = doses[:, j]
                is_vertex_k = (group["is_vertex"] & (group["vertex_domain"] == domain)).to_numpy()
                for train_max_dose in TRAIN_MAX_DOSES:
                    for test_min_dose in TEST_MIN_DOSES:
                        rows.append(
                            {
                                "domain": domain,
                                "scale": scale,
                                "phase_reducer": str(reducer),
                                "train_max_dose": train_max_dose,
                                "test_min_dose": test_min_dose,
                                "n_train": int((dose <= train_max_dose).sum()),
                                "n_test": int(((dose >= test_min_dose) | is_vertex_k).sum()),
                                "n_vertex": int(is_vertex_k.sum()),
                            }
                        )
    return pd.DataFrame(rows)


def preregistration_table(feasibility: pd.DataFrame) -> pd.DataFrame:
    """Domains meeting the spec defaults, per scale."""
    at_defaults = feasibility[
        (feasibility["phase_reducer"] == str(PhaseReducer.MAX))
        & (feasibility["train_max_dose"] == SPEC_TRAIN_MAX_DOSE)
        & (feasibility["test_min_dose"] == SPEC_TEST_MIN_DOSE)
    ].copy()
    at_defaults["eligible"] = (at_defaults["n_train"] >= SPEC_MIN_TRAIN) & (at_defaults["n_test"] >= SPEC_MIN_TEST)
    return at_defaults.sort_values(["scale", "domain"]).reset_index(drop=True)


def recalibration_table(df: pd.DataFrame) -> pd.DataFrame:
    """Best achievable thresholds per domain x scale (MAX reducer): dose quantiles and the
    largest test_min_dose still yielding n_test >= 15 / >= 20."""
    domains = domain_columns(df)
    rows = []
    for scale, group in df.groupby("scale", sort=True):
        doses = reduced_doses(group, domains, PhaseReducer.MAX)
        for j, domain in enumerate(domains):
            dose = np.sort(doses[:, j])[::-1]  # descending
            rows.append(
                {
                    "domain": domain,
                    "scale": scale,
                    "dose_p90": float(np.quantile(dose, 0.9)),
                    "dose_p95": float(np.quantile(dose, 0.95)),
                    "dose_max": float(dose[0]),
                    # the k-th largest dose = max test_min_dose with n_test >= k (no vertex runs)
                    "tmin_for_ntest_15": float(dose[RELAXED_MIN_TEST - 1]),
                    "tmin_for_ntest_20": float(dose[SPEC_MIN_TEST - 1]),
                    "n_train_at_0p02": int((doses[:, j] <= SPEC_TRAIN_MAX_DOSE).sum()),
                }
            )
    return pd.DataFrame(rows)


def headline_scan(df: pd.DataFrame) -> pd.DataFrame:
    """For MAX reducer at train_max_dose 0.02: n eligible domains as test_min_dose varies."""
    domains = domain_columns(df)
    rows = []
    for scale, group in df.groupby("scale", sort=True):
        doses = reduced_doses(group, domains, PhaseReducer.MAX)
        n_train = (doses <= SPEC_TRAIN_MAX_DOSE).sum(axis=0)
        for tmin in RECALIBRATION_TEST_MIN_DOSE_GRID:
            n_test = (doses >= tmin).sum(axis=0)
            rows.append(
                {
                    "scale": scale,
                    "test_min_dose": float(tmin),
                    "n_domains_ntest_ge_15": int(((n_test >= RELAXED_MIN_TEST) & (n_train >= SPEC_MIN_TRAIN)).sum()),
                    "n_domains_ntest_ge_20": int(((n_test >= SPEC_MIN_TEST) & (n_train >= SPEC_MIN_TRAIN)).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    df = pd.read_parquet(args.data_dir / "runs.parquet")
    df = df[df["eval/uncheatable_eval/bpb"].notna()]

    feasibility = enumerate_splits(df)
    out_path = args.data_dir / "feasibility.parquet"
    feasibility.to_parquet(out_path, index=False)
    logger.info("wrote %s (%d rows)", out_path, len(feasibility))

    pd.set_option("display.max_rows", 200, "display.width", 200)
    prereg = preregistration_table(feasibility)
    print(
        f"\n=== Pre-registration at spec defaults (reducer=max, train_max_dose={SPEC_TRAIN_MAX_DOSE}, "
        f"test_min_dose={SPEC_TEST_MIN_DOSE}; eligible = n_train>={SPEC_MIN_TRAIN} & n_test>={SPEC_MIN_TEST}) ==="
    )
    print(prereg[["scale", "domain", "n_train", "n_test", "n_vertex", "eligible"]].to_string(index=False))
    for scale, group in prereg.groupby("scale"):
        eligible = group[group["eligible"]]["domain"].tolist()
        print(f"\nscale {scale}: {len(eligible)}/{len(group)} domains eligible at defaults: {eligible}")

    print("\n=== Recalibration: per-domain dose quantiles and best achievable test_min_dose (reducer=max) ===")
    print(recalibration_table(df).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n=== Headline scan (reducer=max, train_max_dose={SPEC_TRAIN_MAX_DOSE}) ===")
    print(headline_scan(df).to_string(index=False))


if __name__ == "__main__":
    main()
