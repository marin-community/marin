# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import score_delphi_wspu_coupling_20260906 as scoring


def test_primary_source_contrast_is_kappa_one_minus_zero_and_preserves_ties():
    frame = pd.DataFrame(
        {
            "target": ["table9"] * 6,
            "method": [scoring.COUPLING_METHODS[0]] * 3 + [scoring.COUPLING_METHODS[-1]] * 3,
            "stratum": ["source_block:0", "source_block:1", "source_block:2"] * 2,
            "regret_at_1": [0.2, 0.2, 0.1, 0.1, 0.2, 0.3],
        }
    )
    result = scoring.source_contrasts(frame, ["target"], ("regret_at_1",)).iloc[0]
    assert result.candidate == scoring.COUPLING_METHODS[-1]
    assert result.reference == scoring.COUPLING_METHODS[0]
    np.testing.assert_allclose(result.mean_delta, 1 / 30, atol=1e-15, rtol=0)
    np.testing.assert_allclose(result.fraction_candidate_lower, 1 / 3, atol=1e-15, rtol=0)
    assert result.ci_low <= result.mean_delta <= result.ci_high
