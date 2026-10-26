# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Write the tuned RegMix full-swarm out-of-fold predictions in the layout of ``regmix_official_oof_20260913``.

``learning_curve_regmix_oof_20260924`` fitted the inner-fold-tuned trees (registry entry ``lightgbm_regmix``, the tuned
RegMix of Table 5) on the frozen learning-curve folds and stored them as study records. This module reads the
full-swarm records (k = 279, ten fold assignments, both objectives) and writes ``<target>_draw<d>.npz`` files with
``rows`` and ``oof`` so that ``table9_reliability_fit_metrics_20260905.py --regmix-oof-dir`` scores the tuned trees
beside MARINER and Olmix in the R^2-versus-SNR figure, in place of the released recipe.

usage: uv run --offline --no-sync python regmix_tuned_oof_records_20260930.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_mariner_fits_20260908 as fits,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_metrics_20260905 as metrics_module,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import table9_snr_table_20260905 as snr  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "regmix_tuned_oof_20260930"
MODEL = "regmix"
FULL_K = 279
DRAWS = 10
TARGETS = ("uncheatable", "table9")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_module.select_study("mariner")
    panel = benchmark.load_panel(snr.PANEL)
    expected_rows = np.delete(np.arange(panel.rows), fits.calibration_row(panel))
    manifest = {"model": MODEL, "k": FULL_K, "draws": DRAWS, "record_dir": str(fits.RECORD_DIR), "records": {}}
    for target in TARGETS:
        components = len(panel.group(target).components)
        for draw in range(DRAWS):
            record, status = metrics_module.load_record_set(
                target, MODEL, FULL_K, draw, components, fits.RECORD_DIR, strict=False
            )
            if record is None:
                raise FileNotFoundError(f"{target} draw {draw}: {status}")
            if not np.array_equal(record.rows, expected_rows):
                raise ValueError(f"{target} draw {draw}: the record must score every non-anchor mixture exactly once")
            np.savez(args.output_dir / f"{target}_draw{draw}.npz", rows=record.rows, oof=record.oof)
            manifest["records"][f"{target}_draw{draw}"] = {"status": status, "shape": list(record.oof.shape)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
