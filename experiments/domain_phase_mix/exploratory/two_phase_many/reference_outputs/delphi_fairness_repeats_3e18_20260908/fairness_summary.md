# Seed-matched comparison at 3e18 (trainer seeds 0, 1, 2; data seed 666200 Uncheatable, 662009 Table 9)

| kind              | candidate_id                                | label                                           | metric           | seeds   | values                  |    mean |     sd |   n |   standard_error |
|:------------------|:--------------------------------------------|:------------------------------------------------|:-----------------|:--------|:------------------------|--------:|-------:|----:|-----------------:|
| policy            | olmix_u_kl0p1_cap04                         | Olmix, KL 0.1, cap 4                            | uncheatable_bpb  | 0;1;2   | 1.0043;1.0017;1.0039    |  1.0033 | 0.0014 |   3 |         nan      |
| policy            | olmix_u_kl0p05_cap04                        | Olmix, KL 0.05, cap 4 (ladder policy)           | uncheatable_bpb  | 0;1;2   | 1.0030;0.9997;1.0022    |  1.0016 | 0.0018 |   3 |         nan      |
| policy            | lwspu_u_snc_cap06                           | ours, unconstrained (5.3 epochs)                | uncheatable_bpb  | 0;1;2   | 0.9814;0.9834;0.9825    |  0.9825 | 0.0010 |   3 |         nan      |
| policy            | olmix_t9_kl0p005_cap05                      | Olmix, KL 0.005, cap 4                          | table9_macro_bpb | 0;1;2   | 1.0777;1.0803;1.0910    |  1.0830 | 0.0071 |   3 |         nan      |
| policy            | lwspu_t9_snc_cap08                          | ours, unconstrained (7.5 epochs)                | table9_macro_bpb | 0;1;2   | 1.0682;1.0624;1.0727    |  1.0678 | 0.0052 |   3 |         nan      |
| policy            | lwspu_t9_snc_cap06                          | ours, cap 6                                     | table9_macro_bpb | 0;1;2   | 1.0642;1.0639;1.0694    |  1.0658 | 0.0031 |   3 |         nan      |
| paired_difference | lwspu_u_snc_cap06 - olmix_u_kl0p1_cap04     | lwspu_u_snc_cap06 minus olmix_u_kl0p1_cap04     | uncheatable_bpb  | 0;1;2   | -0.0229;-0.0182;-0.0214 | -0.0208 | 0.0024 |   3 |           0.0014 |
| paired_difference | lwspu_u_snc_cap06 - olmix_u_kl0p05_cap04    | lwspu_u_snc_cap06 minus olmix_u_kl0p05_cap04    | uncheatable_bpb  | 0;1;2   | -0.0216;-0.0162;-0.0197 | -0.0192 | 0.0027 |   3 |           0.0016 |
| paired_difference | lwspu_t9_snc_cap08 - olmix_t9_kl0p005_cap05 | lwspu_t9_snc_cap08 minus olmix_t9_kl0p005_cap05 | table9_macro_bpb | 0;1;2   | -0.0095;-0.0179;-0.0183 | -0.0152 | 0.0050 |   3 |           0.0029 |
| paired_difference | lwspu_t9_snc_cap06 - olmix_t9_kl0p005_cap05 | lwspu_t9_snc_cap06 minus olmix_t9_kl0p005_cap05 | table9_macro_bpb | 0;1;2   | -0.0135;-0.0164;-0.0216 | -0.0172 | 0.0041 |   3 |           0.0024 |
| paired_difference | lwspu_t9_snc_cap06 - lwspu_t9_snc_cap08     | lwspu_t9_snc_cap06 minus lwspu_t9_snc_cap08     | table9_macro_bpb | 0;1;2   | -0.0040;+0.0015;-0.0034 | -0.0020 | 0.0030 |   3 |           0.0017 |
