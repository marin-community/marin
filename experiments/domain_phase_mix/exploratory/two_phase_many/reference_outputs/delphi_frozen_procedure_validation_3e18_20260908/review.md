# Frozen-procedure validation review

**Gate: PASS** (3 of 3 rows pass).

| candidate_id       | target      |   epoch_cap |   predicted |   measured |   miss |   incumbent |   incumbent_miss |   tv_to_incumbent |   olmix_best |   wspu_additive |   threshold | within_2sd_of_incumbent   | below_olmix   | not_above_wspu_margin   | miss_not_larger_than_flat   | gate   |
|:-------------------|:------------|------------:|------------:|-----------:|-------:|------------:|-----------------:|------------------:|-------------:|----------------:|------------:|:--------------------------|:--------------|:------------------------|:----------------------------|:-------|
| lwspu_u_snc_cap06  | uncheatable |           6 |      0.9810 |     0.9814 | 0.0004 |      0.9832 |           0.0025 |            0.0381 |       1.0022 |          0.9834 |      0.9852 | True                      | True          | True                    | True                        | pass   |
| lwspu_t9_snc_cap06 | table9      |           6 |      1.0638 |     1.0642 | 0.0004 |      1.0680 |           0.0045 |            0.0254 |       1.0769 |          1.0722 |      1.0760 | True                      | True          | True                    | True                        | pass   |
| lwspu_t9_snc_cap08 | table9      |           8 |      1.0631 |     1.0682 | 0.0051 |      1.0685 |           0.0059 |            0.0269 |       1.0769 |          1.0736 |      1.0765 | True                      | True          | True                    | True                        | pass   |

## Uncheatable components (frozen Uncheatable proposal)

| component              |   predicted |   measured |   incumbent |    miss |
|:-----------------------|------------:|-----------:|------------:|--------:|
| ao3_english            |      1.2356 |     1.2324 |      1.2357 | -0.0032 |
| arxiv_computer_science |      0.9817 |     0.9695 |      0.9711 | -0.0122 |
| arxiv_physics          |      1.0436 |     1.0280 |      1.0317 | -0.0156 |
| bbc_news               |      1.0516 |     1.0847 |      1.0850 |  0.0331 |
| github_cpp             |      0.7570 |     0.7455 |      0.7483 | -0.0115 |
| github_python          |      0.7109 |     0.7271 |      0.7273 |  0.0162 |
| wikipedia_english      |      1.1072 |     1.1151 |      1.1137 |  0.0079 |

## Table-9 components, lwspu_t9_snc_cap06 (cap 6): per-component RMSE 0.0329, mean miss +0.0004

Largest moves against the incumbent:

| component                      |   predicted |   measured |   incumbent |    miss |   vs_incumbent |
|:-------------------------------|------------:|-----------:|------------:|--------:|---------------:|
| basic_skills_string_operations |      2.6785 |     2.6865 |      2.8073 |  0.0080 |        -0.1208 |
| squad                          |      0.5975 |     0.5918 |      0.6386 | -0.0057 |        -0.0468 |
| drop                           |      1.8380 |     1.7577 |      1.7986 | -0.0803 |        -0.0409 |
| winogrande                     |      1.4259 |     1.4077 |      1.4415 | -0.0182 |        -0.0338 |
| basic_skills_logical_reasoning |      0.4160 |     0.3873 |      0.4190 | -0.0287 |        -0.0317 |
| csqa                           |      1.4380 |     1.5114 |      1.4797 |  0.0733 |         0.0317 |
| medmcqa                        |      1.9220 |     1.9615 |      1.9320 |  0.0395 |         0.0295 |
| basic_skills_coding            |      0.6340 |     0.6333 |      0.6603 | -0.0008 |        -0.0270 |

## Table-9 components, lwspu_t9_snc_cap08 (cap 8): per-component RMSE 0.0324, mean miss +0.0051

Largest moves against the incumbent:

| component                      |   predicted |   measured |   incumbent |    miss |   vs_incumbent |
|:-------------------------------|------------:|-----------:|------------:|--------:|---------------:|
| basic_skills_common_knowledge  |      1.1597 |     1.1030 |      1.1742 | -0.0567 |        -0.0713 |
| basic_skills_arithmetic        |      1.9850 |     1.9561 |      1.9049 | -0.0289 |         0.0513 |
| basic_skills_string_operations |      2.6701 |     2.6743 |      2.7251 |  0.0042 |        -0.0508 |
| basic_skills_pattern           |      1.5790 |     1.5659 |      1.5312 | -0.0130 |         0.0347 |
| jeopardy                       |      1.5695 |     1.5822 |      1.6118 |  0.0127 |        -0.0296 |
| winogrande                     |      1.4284 |     1.4480 |      1.4247 |  0.0196 |         0.0233 |
| piqa                           |      1.2848 |     1.3046 |      1.2813 |  0.0198 |         0.0233 |
| basic_skills_coding            |      0.6280 |     0.6492 |      0.6712 |  0.0211 |        -0.0220 |
