
==============================================================================
RATIONALE GROUNDING - HYPOTHESIS TESTS
==============================================================================

[H1] Variant_A grounds in spec text more than full_spec.
  Metric: mean spec_distinctive_hit_rate (per-judgment, normalized by
          per-statement distinctive-token count). Direction: variant_A > full_spec.
  judge                 variant_A              full_spec      delta  cohen_d
  gpt          0.0347 [0.0329,0.0366] (n=2758)     0.1990 [0.1955,0.2024] (n=2756)  -0.1643  -2.141
  gemini       0.0461 [0.0443,0.0480] (n=2748)     0.0896 [0.0875,0.0920] (n=2757)  -0.0435  -0.797
  glm          0.0530 [0.0511,0.0551] (n=2756)     0.1624 [0.1590,0.1658] (n=2668)  -0.1094  -1.509

[H2] Rubric-only judges cannot reason from spec text (no leakage).
  Metric: mean spec_distinctive_hit_rate for variant_B. Expected: ~0.
  judge          mean     median                     95% CI      n
  gpt          0.0194     0.0139    [  0.0184,   0.0204]   2758
  gemini       0.0178     0.0139    [  0.0171,   0.0186]   2743
  glm          0.0194     0.0146    [  0.0186,   0.0203]   2752

[H3] Phase-4: when tension flag fires, judge grounds more in rubric than spec.
  Metric: mean spec_distinctive_hits vs rubric_distinctive_hits, split by tension.
  gpt: tension=true n=303, tension=false n=2455
    tension=true   spec_dist= 11.60  rubric_dist= 18.41  delta(rub-spec)=+6.80
    tension=false  spec_dist= 10.97  rubric_dist= 17.63  delta(rub-spec)=+6.66
  gemini: tension=true n=17, tension=false n=2724
    tension=true   spec_dist=  6.53  rubric_dist=  6.24  delta(rub-spec)=-0.29
    tension=false  spec_dist=  6.75  rubric_dist=  8.14  delta(rub-spec)=+1.39

[H4] On statements where rubric DROPPED a qualifier, judges with spec context
     cite the qualifier; rubric-only judges cannot.
  Metric: any_qualifier_rate, restricted to statements where qualifier_dropped=true.
  (n statements with qualifier dropped: 16/46)
  judge         variant_A      variant_B      full_spec       rub+spec
  gpt              0.0427         0.0063         0.3333         0.1562
  gemini           0.0094         0.0094         0.1990         0.0765
  glm              0.1493         0.0293         0.4670         0.1661

[H5] Verbatim quote discipline: spec_quotes / rubric_quotes are real substrings.
  spec_quote_verify_rate (mean over judgments where any spec_quote was provided):
  condition              judge          mean  n_with_quotes
  variant_A              gpt          0.9895           2758
  variant_A              gemini       0.9822           2748
  variant_A              glm          0.9138           2749
  rubric_plus_spec       gpt          0.9967           2758
  rubric_plus_spec       gemini       0.9759           2741
  rubric_plus_spec       glm          0.9316           2292

  rubric_quote_verify_rate:
  condition              judge          mean  n_with_quotes
  variant_B              gpt          0.9947           2758
  variant_B              gemini       0.9969           2743
  variant_B              glm          0.8331           2750
  rubric_plus_spec       gpt          0.9935           2758
  rubric_plus_spec       gemini       0.9861           2741
  rubric_plus_spec       glm          0.7749           2328

[H6] Reasoning length per judge (Gemini-binary-scorer hypothesis: shorter).
  condition              judge       mean_tokens     median
  variant_A              gpt                45.5         42
  variant_A              gemini             41.7         40
  variant_A              glm                49.6         48
  variant_B              gpt                37.0         36
  variant_B              gemini             35.0         33
  variant_B              glm                44.5         43
  full_spec              gpt               506.9        509
  full_spec              gemini            145.3        142
  full_spec              glm               254.1        252
  rubric_plus_spec       gpt               125.8        125
  rubric_plus_spec       gemini             66.6         64
  rubric_plus_spec       glm                85.2         89

[BONUS A] Spec vs rubric grounding by (condition x judge) — mean per judgment
  condition              judge     spec_dist   rub_dist    5g_spec     5g_rub   5g_other
  variant_A              gpt            3.54       6.28       0.74       0.42       0.02
  variant_A              gemini         4.43       4.05       0.74       0.26       0.02
  variant_A              glm            5.11       4.88       1.68       0.44       0.02
  variant_B              gpt            1.98       7.43       0.06       0.40       0.02
  variant_B              gemini         1.94       6.61       0.06       0.56       0.01
  variant_B              glm            2.23       9.22       0.10       1.99       0.02
  full_spec              gpt           22.21      27.59       5.52       0.62       3.49
  full_spec              gemini        10.20      10.07       4.04       0.30       2.68
  full_spec              glm           17.53      16.24      12.93       0.70      11.63
  rubric_plus_spec       gpt           11.04      17.71      11.23       7.19       0.06
  rubric_plus_spec       gemini         6.75       8.13       4.14       2.28       0.03
  rubric_plus_spec       glm            7.29      12.81       6.25       6.29       0.04

[BONUS B] Rubric-style language ('score N' citation) presence rate
  condition              judge      score_n_rate   anchor_word_mean
  variant_A              gpt              0.0000               0.07
  variant_A              gemini           0.0004               0.01
  variant_A              glm              0.0000               0.04
  variant_B              gpt              0.0004               0.12
  variant_B              gemini           0.0004               0.02
  variant_B              glm              0.0062               0.08
  full_spec              gpt              0.0065               0.05
  full_spec              gemini           0.0011               0.01
  full_spec              glm              0.0019               0.01
  rubric_plus_spec       gpt              0.0022               0.38
  rubric_plus_spec       gemini           0.0047               0.06
  rubric_plus_spec       glm              0.0913               0.34

[BONUS C] Phase-4 tension flag rates
  gpt: 303/2758 = 0.1099
  gemini: 17/2741 = 0.0062
  glm: 168/2443 = 0.0688

[BONUS D] SequenceMatcher fuzzy-match: longest contiguous substring shared between
          rationale and spec/rubric. Robust to paraphrase that breaks n-gram match.
  condition              judge      spec_tok    rub_tok     spec_%      rub_%
  variant_A              gpt            3.61       3.48     0.0832     0.0838
  variant_A              gemini         3.74       3.20     0.0930     0.0825
  variant_A              glm            4.73       3.47     0.0972     0.0751
  variant_B              gpt            2.39       3.50     0.0683     0.0997
  variant_B              gemini         2.46       3.76     0.0737     0.1122
  variant_B              glm            2.54       5.18     0.0605     0.1224
  full_spec              gpt            7.81       4.08     0.0164     0.0088
  full_spec              gemini         6.26       3.47     0.0430     0.0251
  full_spec              glm           13.30       4.04     0.0566     0.0171
  rubric_plus_spec       gpt           12.94       9.74     0.1028     0.0777
  rubric_plus_spec       gemini         7.04       5.51     0.1062     0.0844
  rubric_plus_spec       glm            8.76       8.63     0.1023     0.1100

[BONUS D2] Quote fidelity (per provided quote, max contiguous-match pct vs source):
  condition              judge      spec_q_max_pct    rub_q_max_pct
  variant_A              gpt                0.9999                -
  variant_A              gemini             0.9969                -
  variant_A              glm                0.9953                -
  variant_B              gpt                     -           0.9981
  variant_B              gemini                  -           0.9998
  variant_B              glm                     -           0.9969
  full_spec              gpt                     -                -
  full_spec              gemini                  -                -
  full_spec              glm                     -                -
  rubric_plus_spec       gpt                1.0000           1.0000
  rubric_plus_spec       gemini             0.9972           0.9970
  rubric_plus_spec       glm                0.9970           0.9964

[BONUS E] In full_spec, do judges ground in target statement more than other statements?
  Metric: ratio of 5g_in_target_spec to (5g_in_target_spec + 5g_in_other_specs)
  gpt: mean target/(target+other) ratio = 0.4067  (n=2105)
  gemini: mean target/(target+other) ratio = 0.4686  (n=1833)
  glm: mean target/(target+other) ratio = 0.4973  (n=2367)

