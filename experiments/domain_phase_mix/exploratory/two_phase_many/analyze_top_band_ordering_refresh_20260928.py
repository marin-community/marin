# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Top-band ordering of the Delphi bank with the paper's MARINER (kappa-floor link, flat profile) added.

Runs `analyze_delphi_top_band_ordering_20260906` over the frozen prediction packages with the current MARINER
entry (`weibull_softplus_unscaled@kappa_floor_link_flat15`, scored on 2026-09-07) beside the September 6 methods, and
writes the same outputs to `reference_outputs/delphi_top_band_ordering_20260928/`. Nothing is fitted.

usage: uv run --offline --no-sync python analyze_top_band_ordering_refresh_20260928.py
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

import analyze_delphi_top_band_ordering_20260906 as top_band  # noqa: E402

OUTPUT = SCRIPT_DIR / "reference_outputs" / "delphi_top_band_ordering_20260928"
PACKAGES = {
    "delphi_kappa_floor_flat15_selection_20260907": {
        "weibull_softplus_unscaled@kappa_floor_link_flat15": "MARINER",
    },
    # The September 6 labels are fixed: the script's ensembles and two-stage policy refer to them by name.
    "delphi_offline_selection_20260906": {
        "weibull_softplus_unscaled": "WSPU",
        "dsp_total_exposure": "DSP",
        "olmix_loglinear_taskwise": "OLMix",
    },
    "delphi_link_variants_selection_20260906": {
        "weibull_softplus_unscaled@log_deficit_bounded_link": "WSPU, bounded link",
        "weibull_softplus_unscaled@log_deficit_bounded_link_total_hub": "WSPU, link + hub",
    },
}
PLOTTED = (
    "MARINER",
    "WSPU",
    "WSPU, bounded link",
    "DSP",
    "OLMix",
    "bank kernel LOSO, TV 0.05",
    "bank kernel LOSO, TV 0.1",
)


def main() -> None:
    top_band.PACKAGES = PACKAGES
    top_band.PLOTTED = PLOTTED
    sys.argv = [sys.argv[0], "--output-dir", str(OUTPUT)]
    top_band.main()


if __name__ == "__main__":
    main()
