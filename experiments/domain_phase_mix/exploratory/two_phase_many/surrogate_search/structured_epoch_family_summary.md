# Structured epoch-law surrogate results

## Many-domain
| model                         |   reported_n_params |   linear_coefficients |   global_shape_parameters |   total_with_shapes |   train_rmse |    cv_rmse |   cv_regret_at_1 |   cv_foldmean_regret_at_1 |
|:------------------------------|--------------------:|----------------------:|--------------------------:|--------------------:|-------------:|-----------:|-----------------:|--------------------------:|
| CCGlobalPremium-RetainedTotal |                  29 |                    28 |                         6 |                  35 |   0.00794963 | 0.0088934  |                0 |                0.00976806 |
| CCGlobalPremium-Threshold     |                  29 |                    28 |                         6 |                  35 |   0.00865014 | 0.00973766 |                0 |                0.0104599  |

## Starcoder
| model                  |   reported_n_params |   linear_coefficients |   global_shape_parameters |   total_with_shapes |   train_rmse |   cv_rmse |   cv_regret_at_1 |   cv_foldmean_regret_at_1 |
|:-----------------------|--------------------:|----------------------:|--------------------------:|--------------------:|-------------:|----------:|-----------------:|--------------------------:|
| ThresholdTotal         |                   3 |                     3 |                         6 |                  10 |    0.0319316 | 0.03394   |       0.00512892 |                0.00445764 |
| TotalLog               |                   3 |                     3 |                         5 |                   9 |    0.031932  | 0.0339403 |       0.00512892 |                0.00445764 |
| RetainedTotal          |                   3 |                     3 |                         5 |                   9 |    0.0322017 | 0.0341261 |       0.00512892 |                0.00445764 |
| ThresholdRetainedTotal |                   3 |                     3 |                         6 |                  10 |    0.0322019 | 0.0341273 |       0.00512892 |                0.00445764 |

Reported parameter count matches the legacy tables only for the many-domain rows.
For Starcoder, the legacy scripts counted only fitted linear coefficients and excluded the intercept.
