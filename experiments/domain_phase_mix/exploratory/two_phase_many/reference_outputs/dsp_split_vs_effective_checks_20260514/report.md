# Split DSP vs Effective-Exposure DSP Checks

This report compares the promoted effective-exposure DSP against the split benefit/saturation/penalty DSP.
All perturbation checks use the proportional domain-bump experiment and cached DSP fits; no models are refit here.

## Fit And Raw-Optimum Diagnostics

| scale_label   | variant                                   |   total_param_count |   cv_rmse |   oof_spearman |   raw_nearest_observed_tv |   phase0_max_weight |   phase1_max_weight |
|:--------------|:------------------------------------------|--------------------:|----------:|---------------:|--------------------------:|--------------------:|--------------------:|
| 60M/1.2B      | dsp_effective_exposure_penalty_nnls       |                 158 |  0.007280 |       0.903576 |                  0.809246 |            0.793439 |            0.820979 |
| 60M/1.2B      | dsp_phase_benefit_saturation_penalty_nnls |                 160 |  0.006766 |       0.916493 |                  0.756612 |            1.000000 |            0.249347 |
| 100M/6B       | dsp_effective_exposure_penalty_nnls       |                 158 |  0.007106 |       0.919645 |                  0.500000 |            0.271780 |            0.932116 |
| 100M/6B       | dsp_phase_benefit_saturation_penalty_nnls |                 160 |  0.006635 |       0.926152 |                  0.499956 |            1.000000 |            0.723478 |

## Within-Scale Domain-Bump Agreement

Finite predictions compare `DSP(w_bump) - DSP(w_prop)` against the observed 0.05-domain bump.
Local predictions compare the directional derivative at proportional against that same finite intervention.

| variant                                   | target_scale   |   finite_rmse |   finite_pearson |   finite_spearman |   finite_sign_agreement |   local_rmse |   local_pearson |   local_spearman |   local_sign_agreement |
|:------------------------------------------|:---------------|--------------:|-----------------:|------------------:|------------------------:|-------------:|----------------:|-----------------:|-----------------------:|
| dsp_effective_exposure_penalty_nnls       | 60m_1p2b       |      0.003247 |         0.899660 |          0.727126 |                0.948718 |     0.023807 |        0.695586 |         0.685628 |               0.769231 |
| dsp_phase_benefit_saturation_penalty_nnls | 60m_1p2b       |      0.003170 |         0.913475 |          0.745749 |                0.923077 |     0.018229 |        0.546486 |         0.458704 |               0.820513 |
| dsp_effective_exposure_penalty_nnls       | 300m_6b        |      0.003007 |         0.906863 |          0.643522 |                0.923077 |     0.016547 |        0.677940 |         0.493320 |               0.743590 |
| dsp_phase_benefit_saturation_penalty_nnls | 300m_6b        |      0.003603 |         0.895747 |          0.582186 |                0.846154 |     0.143211 |        0.012718 |         0.475101 |               0.794872 |

## Cross-Scale Transfer

Rows below evaluate a model fit at one scale against observed perturbation effects at the other scale.

| variant                                   | model_fit_scale   | target_scale   |   finite_rmse |   finite_pearson |   finite_spearman |   finite_sign_agreement |
|:------------------------------------------|:------------------|:---------------|--------------:|-----------------:|------------------:|------------------------:|
| dsp_effective_exposure_penalty_nnls       | 60m_1p2b          | 300m_6b        |      0.004566 |         0.784369 |          0.539271 |                0.897436 |
| dsp_phase_benefit_saturation_penalty_nnls | 60m_1p2b          | 300m_6b        |      0.004496 |         0.813062 |          0.542915 |                0.871795 |
| dsp_effective_exposure_penalty_nnls       | 300m_6b           | 60m_1p2b       |      0.003451 |         0.897139 |          0.670243 |                0.974359 |
| dsp_phase_benefit_saturation_penalty_nnls | 300m_6b           | 60m_1p2b       |      0.004051 |         0.832873 |          0.554656 |                0.897436 |

## Scale-Interaction Prediction

Interaction is `effect_100_bpb - effect_60_bpb`; negative means a perturbation helps more at 100M/6B.

| variant                                   |   finite_interaction_rmse |   finite_interaction_pearson |   finite_interaction_spearman |   local_interaction_rmse |   local_interaction_pearson |   local_interaction_spearman |
|:------------------------------------------|--------------------------:|-----------------------------:|------------------------------:|-------------------------:|----------------------------:|-----------------------------:|
| dsp_effective_exposure_penalty_nnls       |                  0.002734 |                     0.624192 |                      0.350607 |                 0.027527 |                    0.777744 |                     0.435020 |
| dsp_phase_benefit_saturation_penalty_nnls |                  0.003347 |                     0.650222 |                      0.287449 |                 0.134560 |                    0.645758 |                     0.231579 |

## Existing Three-Vector Alignment At 100M

This uses the previously generated vectors: measured perturbation gradient, DSP-predicted perturbation gradient, and direction from proportional to the raw DSP optimum.

| model_variant                             | comparison                                     |   pearson |   spearman |   cosine |   sign_agreement |
|:------------------------------------------|:-----------------------------------------------|----------:|-----------:|---------:|-----------------:|
| dsp_phase_benefit_saturation_penalty_nnls | measured_gradient_vs_finite_predicted_gradient |  0.895747 |   0.582186 | 0.895747 |         0.820513 |
| dsp_phase_benefit_saturation_penalty_nnls | measured_gradient_vs_local_predicted_gradient  |  0.012713 |   0.499798 | 0.012713 |         0.769231 |
| dsp_phase_benefit_saturation_penalty_nnls | measured_gradient_vs_optimum_direction         |  0.529306 |   0.122672 | 0.529306 |         0.743590 |
| dsp_phase_benefit_saturation_penalty_nnls | predicted_local_gradient_vs_optimum_direction  |  0.032342 |   0.522267 | 0.032342 |         0.923077 |
| dsp_phase_benefit_saturation_penalty_nnls | predicted_finite_gradient_vs_optimum_direction |  0.452442 |   0.044130 | 0.452442 |         0.717949 |
| dsp_effective_exposure_penalty_nnls       | measured_gradient_vs_finite_predicted_gradient |  0.906863 |   0.643522 | 0.906863 |         0.820513 |
| dsp_effective_exposure_penalty_nnls       | measured_gradient_vs_local_predicted_gradient  |  0.686412 |   0.515182 | 0.686412 |         0.794872 |
| dsp_effective_exposure_penalty_nnls       | measured_gradient_vs_optimum_direction         |  0.630356 |   0.282389 | 0.630356 |         0.666667 |
| dsp_effective_exposure_penalty_nnls       | predicted_local_gradient_vs_optimum_direction  |  0.456837 |   0.539271 | 0.456837 |         0.769231 |
| dsp_effective_exposure_penalty_nnls       | predicted_finite_gradient_vs_optimum_direction |  0.482979 |   0.111741 | 0.482979 |         0.589744 |

## Interpretation

- Split benefit/saturation/penalty DSP improves OOF fit on both scales, but its raw optimum is still extrapolative.
- Effective-exposure DSP has slightly weaker OOF fit but cleaner local-gradient behavior near proportional.
- Split DSP is competitive for finite 0.05 perturbation prediction; it is less convincing as a true local-gradient oracle.
- For validation candidates, treat split DSP as a strong fit/finite-bump surrogate and effective-exposure DSP as the safer local-geometry comparator.
