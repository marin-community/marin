# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the 60M/1.2B proportional domain-deletion and log-tilt panel.

This is the 60M mirror of ``launch_proportional_controllability_300m``: 39
domain deletions plus 39 paired central log-tilts around proportional.  It
reuses the 300M launcher implementation after replacing only the scale-specific
constants, so the intervention formulas stay identical across scales.
"""

from __future__ import annotations

from pathlib import Path

import experiments.domain_phase_mix.launch_proportional_controllability_300m as pctrl
from experiments.domain_phase_mix.scaling_study_recipes import ScalingStudyScale

# Keep this short: Levanter/W&B truncates long checkpoint step names, and the
# full "proportional_controllability" prefix collapses many high/low tilt names.
BASE_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_pctrl60"
COHORT = "proportional_controllability_60m"
FAMILY = "proportional_controllability_60m_1p2b"
RUN_ID_BASE = 810_000
SCALE = ScalingStudyScale.REGMIX_60M_1P2B
DISPLAY_LABEL = "60M/1.2B"
DEFAULT_MAX_CONCURRENT = 32
DEFAULT_EVAL_DATASETS_CACHE_PATH = "gs://marin-us-east5/raw/eval-datasets/proportional-controllability-60m"
DEFAULT_LOCAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "proportional_controllability_60m_20260620"
)
_ORIGINAL_BUILD_INTERVENTIONS = pctrl.build_interventions


def _compact_run_name(spec: pctrl.ControllabilityInterventionSpec) -> str:
    domain_to_index = {domain_name: index for index, domain_name in enumerate(pctrl.DOMAIN_NAMES)}
    if spec.intervention_type == pctrl.InterventionType.DOMAIN_DELETION.value:
        if spec.target_domain is None:
            raise ValueError(f"Deletion intervention {spec.intervention_id} is missing target_domain")
        return f"p60_del_{domain_to_index[spec.target_domain]:02d}"
    if spec.intervention_type == pctrl.InterventionType.LOG_TILT.value:
        if spec.target_domain is None or spec.tilt_sign is None:
            raise ValueError(f"Log-tilt intervention {spec.intervention_id} has incomplete metadata")
        sign = {"plus": "p", "minus": "m"}[spec.tilt_sign]
        return f"p60_tilt_{domain_to_index[spec.target_domain]:02d}_{sign}"
    raise ValueError(f"Unsupported intervention type {spec.intervention_type!r}")


def _build_compact_interventions() -> list[pctrl.ControllabilityInterventionSpec]:
    specs = [pctrl.replace(spec, run_name=_compact_run_name(spec)) for spec in _ORIGINAL_BUILD_INTERVENTIONS()]
    pctrl.validate_interventions(specs)
    return specs


def configure_60m_panel() -> None:
    """Patch the shared proportional-controllability launcher to 60M constants."""
    pctrl.BASE_NAME_PREFIX = BASE_NAME_PREFIX
    pctrl.COHORT = COHORT
    pctrl.FAMILY = FAMILY
    pctrl.RUN_ID_BASE = RUN_ID_BASE
    pctrl.SCALE = SCALE
    pctrl.DISPLAY_LABEL = DISPLAY_LABEL
    pctrl.DEFAULT_MAX_CONCURRENT = DEFAULT_MAX_CONCURRENT
    pctrl.DEFAULT_EVAL_DATASETS_CACHE_PATH = DEFAULT_EVAL_DATASETS_CACHE_PATH
    pctrl.DEFAULT_LOCAL_ARTIFACT_DIR = DEFAULT_LOCAL_ARTIFACT_DIR
    pctrl.build_interventions = _build_compact_interventions


def build_interventions() -> list[pctrl.ControllabilityInterventionSpec]:
    """Build the 117-row 60M intervention manifest."""
    configure_60m_panel()
    return pctrl.build_interventions()


def build_run_specs() -> list[pctrl.ControllabilityRunSpec]:
    """Build the 117-row 60M training manifest."""
    configure_60m_panel()
    return pctrl.build_run_specs()


def main() -> None:
    configure_60m_panel()
    pctrl.main()


if __name__ == "__main__":
    main()
