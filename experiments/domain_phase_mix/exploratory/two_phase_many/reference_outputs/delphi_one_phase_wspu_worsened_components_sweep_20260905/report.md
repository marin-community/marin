# WSPU epoch-cap sweep for the three worsened Uncheatable components

Objective: byte-weighted aggregate of bbc_news (0.314), ao3_english (0.449), wikipedia_english (0.238). Same reconstructed component fits and exact
runtime-grid optimizer as the full sweep. No training launched.

- The runtime-grid optimum stops moving at cap 7; caps 7 to 20
  share one mixture.
- Predicted three-component aggregate: 1.1047 at cap 2, 1.0921 at the plateau,
  1.1399 predicted at proportional, 1.1594 predicted at the full
  Uncheatable optimum.
- Predicted full Uncheatable aggregate at the plateau mixture: 1.0771
  (proportional predicted 1.0369).
- Outer-fold prediction SD at the plateau: 0.0176; fold optima sit TV
  0.115 (median) from the full-selection optimum.
- Plateau mixture: 31 supported buckets, largest dolma3_cc/literature_high
  at 0.246, max materialized epochs 6.01,
  TV 0.497 to proportional, 0.584
  to the full Uncheatable optimum.

## Cap sweep

|   epoch_cap |   predicted_worsened_aggregate_bpb |   outer_fold_prediction_sd |   predicted_bbc_news_bpb |   predicted_ao3_english_bpb |   predicted_wikipedia_english_bpb |   predicted_full_uncheatable_bpb |   tv_to_previous_cap |   tv_to_proportional |   fold_optimum_tv_median |   support_buckets |   largest_weight |
|------------:|-----------------------------------:|---------------------------:|-------------------------:|----------------------------:|----------------------------------:|---------------------------------:|---------------------:|---------------------:|-------------------------:|------------------:|-----------------:|
|      2.0000 |                             1.1047 |                     0.0142 |                   1.0142 |                      1.1809 |                            1.0803 |                           1.0748 |             nan      |               0.3549 |                   0.0894 |           31.0000 |           0.2080 |
|      3.0000 |                             1.0971 |                     0.0142 |                   1.0057 |                      1.1691 |                            1.0819 |                           1.0750 |               0.1230 |               0.4296 |                   0.1177 |           31.0000 |           0.1724 |
|      4.0000 |                             1.0940 |                     0.0153 |                   1.0029 |                      1.1625 |                            1.0848 |                           1.0765 |               0.0674 |               0.4623 |                   0.1177 |           31.0000 |           0.1631 |
|      5.0000 |                             1.0925 |                     0.0164 |                   1.0045 |                      1.1572 |                            1.0867 |                           1.0765 |               0.0410 |               0.4789 |                   0.1079 |           31.0000 |           0.2041 |
|      6.0000 |                             1.0921 |                     0.0175 |                   1.0068 |                      1.1534 |                            1.0891 |                           1.0771 |               0.0405 |               0.4962 |                   0.1011 |           31.0000 |           0.2446 |
|      7.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0010 |               0.4972 |                   0.1147 |           31.0000 |           0.2456 |
|      8.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0000 |               0.4972 |                   0.1304 |           31.0000 |           0.2456 |
|      9.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0000 |               0.4972 |                   0.1382 |           31.0000 |           0.2456 |
|     10.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0000 |               0.4972 |                   0.1382 |           31.0000 |           0.2456 |
|     11.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0000 |               0.4972 |                   0.1382 |           31.0000 |           0.2456 |
|     12.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0000 |               0.4972 |                   0.1382 |           31.0000 |           0.2456 |
|     13.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0000 |               0.4972 |                   0.1382 |           31.0000 |           0.2456 |
|     14.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0000 |               0.4972 |                   0.1382 |           31.0000 |           0.2456 |
|     15.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0000 |               0.4972 |                   0.1382 |           31.0000 |           0.2456 |
|     16.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0000 |               0.4972 |                   0.1382 |           31.0000 |           0.2456 |
|     17.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0000 |               0.4972 |                   0.1382 |           31.0000 |           0.2456 |
|     18.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0000 |               0.4972 |                   0.1382 |           31.0000 |           0.2456 |
|     19.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0000 |               0.4972 |                   0.1382 |           31.0000 |           0.2456 |
|     20.0000 |                             1.0921 |                     0.0176 |                   1.0069 |                      1.1533 |                            1.0891 |                           1.0771 |               0.0000 |               0.4972 |                   0.1382 |           31.0000 |           0.2456 |

## Predicted versus observed at the reference mixtures

| component              |   ('predicted', 'full_uncheatable_optimum_cap06') |   ('predicted', 'proportional') |   ('observed', 'full_uncheatable_optimum_cap06') |   ('observed', 'proportional') |
|:-----------------------|--------------------------------------------------:|--------------------------------:|-------------------------------------------------:|-------------------------------:|
| ao3_english            |                                            1.2447 |                          1.2316 |                                           1.2519 |                         1.2310 |
| arxiv_computer_science |                                            0.9302 |                          1.0017 |                                           0.9652 |                         1.0022 |
| arxiv_physics          |                                            0.9723 |                          1.0922 |                                           1.0186 |                         1.0940 |
| bbc_news               |                                            1.0699 |                          1.0415 |                                           1.0881 |                         1.0423 |
| full_aggregate         |                                            0.9471 |                          1.0369 |                                           0.9834 |                         1.0379 |
| github_cpp             |                                            0.6735 |                          0.9036 |                                           0.7476 |                         0.9069 |
| github_python          |                                            0.6691 |                          0.8799 |                                           0.7273 |                         0.8786 |
| wikipedia_english      |                                            1.1166 |                          1.0968 |                                           1.1156 |                         1.1008 |
| worsened_aggregate     |                                            1.1594 |                          1.1399 |                                           1.1681 |                         1.1408 |

## Plateau mixture (cap 7)

| domain                                     |   weight |   proportional_weight |   full_uncheatable_optimum_weight |   materialized_epochs |
|:-------------------------------------------|---------:|----------------------:|----------------------------------:|----------------------:|
| dolma3_cc/literature_high                  |   0.2456 |                0.0370 |                            0.1089 |                6.0134 |
| dolmino_synth_qa                           |   0.1440 |                0.0755 |                            0.0728 |                1.7283 |
| dolma3_cc/entertainment_high               |   0.1245 |                0.0606 |                            0.0171 |                1.8600 |
| dolmino_common_crawl_hq                    |   0.1055 |                0.1886 |                            0.0381 |                0.5063 |
| dolma3_cc/crime_and_law_high               |   0.0908 |                0.0220 |                            0.0132 |                3.7381 |
| dolma3_cc/history_and_geography_high       |   0.0474 |                0.0193 |                            0.0190 |                2.2173 |
| dolma3_cc/crime_and_law_low                |   0.0312 |                0.0099 |                            0.0088 |                2.8634 |
| dolma3_cc/games_high                       |   0.0308 |                0.0323 |                            0.0205 |                0.8612 |
| dolma3_cc/literature_low                   |   0.0273 |                0.0099 |                            0.0156 |                2.5109 |
| dolma3_cc/entertainment_low                |   0.0220 |                0.0221 |                            0.0024 |                0.9020 |
| dolma3_cc/health_low                       |   0.0146 |                0.0266 |                            0.0010 |                0.4992 |
| dolma3_cc/science_math_and_technology_high |   0.0132 |                0.0348 |                            0.1235 |                0.3431 |
| dolma3_cc/art_and_design_high              |   0.0132 |                0.0163 |                            0.0029 |                0.7304 |
| dolma3_cc/industrial_low                   |   0.0107 |                0.0062 |                            0.0132 |                1.5713 |
| dolmino_olmocr_pdfs_hq                     |   0.0098 |                0.0295 |                            0.1582 |                0.3000 |
| dolmino_synth_instruction                  |   0.0093 |                0.0026 |                            0.0093 |                3.2636 |
| dolmino_synth_math                         |   0.0088 |                0.0031 |                            0.0132 |                2.5477 |
| dolma3_cc/industrial_high                  |   0.0063 |                0.0117 |                            0.0137 |                0.4916 |
| dolma3_cc/science_math_and_technology_low  |   0.0059 |                0.0159 |                            0.0396 |                0.3341 |
| dolma3_finemath_3plus                      |   0.0054 |                0.0049 |                            0.0171 |                0.9992 |
| dolma3_cc/education_and_jobs_high          |   0.0054 |                0.0360 |                            0.0132 |                0.1352 |
| dolma3_cc/history_and_geography_low        |   0.0049 |                0.0053 |                            0.0054 |                0.8306 |
| dolma3_cc/finance_and_business_low         |   0.0044 |                0.0371 |                            0.0034 |                0.1072 |
| dolma3_cc/finance_and_business_high        |   0.0044 |                0.0779 |                            0.0010 |                0.0511 |
| dolma3_cc/games_low                        |   0.0034 |                0.0121 |                            0.0005 |                0.2552 |
| dolmino_synth_code                         |   0.0034 |                0.0027 |                            0.0156 |                1.1463 |
| dolmino_stem_heavy_crawl                   |   0.0029 |                0.0007 |                            0.0049 |                3.5542 |
| dolma3_arxiv                               |   0.0020 |                0.0040 |                            0.0210 |                0.4375 |
| dolma3_cc/electronics_and_hardware_high    |   0.0020 |                0.0190 |                            0.0146 |                0.0932 |
| dolma3_cc/health_high                      |   0.0005 |                0.0623 |                            0.0000 |                0.0071 |
| dolma3_cc/food_and_dining_high             |   0.0005 |                0.0245 |                            0.0005 |                0.0181 |
| dolma3_stack_edu                           |   0.0000 |                0.0192 |                            0.1030 |                0.0000 |
| dolma3_wikipedia                           |   0.0000 |                0.0005 |                            0.0020 |                0.0000 |
| dolma3_cc/food_and_dining_low              |   0.0000 |                0.0120 |                            0.0020 |                0.0000 |
| dolmino_stack_edu_fim                      |   0.0000 |                0.0192 |                            0.1006 |                0.0000 |
| dolma3_cc/electronics_and_hardware_low     |   0.0000 |                0.0089 |                            0.0000 |                0.0000 |
| dolma3_cc/education_and_jobs_low           |   0.0000 |                0.0176 |                            0.0000 |                0.0000 |
| dolma3_cc/art_and_design_low               |   0.0000 |                0.0066 |                            0.0000 |                0.0000 |
| dolmino_synth_thinking                     |   0.0000 |                0.0057 |                            0.0044 |                0.0000 |

## Outer-fold optima at the plateau cap

|   epoch_cap | source       |   predicted_worsened_aggregate_bpb |   tv_to_full_selection_optimum |   support_buckets |
|------------:|:-------------|-----------------------------------:|-------------------------------:|------------------:|
|           7 | outer_fold_0 |                             1.0914 |                         0.1147 |                31 |
|           7 | outer_fold_1 |                             1.0379 |                         0.3359 |                27 |
|           7 | outer_fold_2 |                             1.0882 |                         0.0850 |                29 |
|           7 | outer_fold_3 |                             1.0928 |                         0.1567 |                25 |
|           7 | outer_fold_4 |                             1.0731 |                         0.1074 |                31 |
