# Delphi frontier factorial at 3e18

Submitted on 2026-09-06 at 14:14 UTC as `/calvinxu/dm-delphi-3e18-frontier-factorial-v6e8-20260906`
([Iris parent](https://iris.oa.dev/#/job/%2Fcalvinxu%2Fdm-delphi-3e18-frontier-factorial-v6e8-20260906)),
approved by Calvin ("You can launch the 18-run resolution-V factorial ... as long as it's not excessive on
east5b v6es").

Centre: the replicated Table-9 frontier coordinate
`delphi_3e18_39bucket:a1a917b1981fc2cad2c2759dccf963b51fb941fd9816c49087f7b119545d4161` (26 runs in the
frozen bank, measured mean 1.0639, SD 0.0041). Five factors, each at ±δ weight with the mass balanced
against the 26 Common Crawl cells in proportion to their centre weights:

| Factor | Buckets and +1 deltas | Centre weight (epochs) |
| --- | --- | --- |
| A code | Stack-Edu +0.02, Stack-Edu FIM +0.02 | 0.103 (4.9), 0.098 (4.6) |
| B synthetic QA | +0.04 | 0.131 (1.6) |
| C CC-HQ | +0.04 | 0.108 (0.5) |
| D synthetic reasoning | thinking, instruction, math +0.006 each | 0.041 (6.5), 0.015 (5.4), 0.031 (8.9) |
| E PDF / arXiv | olmOCR +0.015, arXiv +0.005 | 0.037 (1.1), 0.014 (3.1) |

Design: the 2^(5-1) half fraction with E = ABCD (resolution V: all five main effects and all ten two-factor
interactions are estimable and unaliased with one another; two-factor interactions alias only three-factor
ones), plus two centre replicates. Eighteen runs, about 6e19 FLOPs. TV from the centre 0.09–0.16; every row
stays within 16 epochs (the centre itself repeats Wikipedia 11.8 times, so the 8-epoch policy cap of the
validations does not apply). Candidate ids carry the `_cap16` suffix the sweep loader requires.

Seeds: every row uses the Table-9 validation data seed 662009 and trainer seed 0, except `centre_r1_cap16`,
which uses trainer seed 1 so that the centre pair measures run-to-run noise at a fixed data seed. Because the
sweep loader aliases identical mixtures within one table, the second replicate lives in its own table
(`candidate_weights_replicate.csv`) and its own sweep definition (run ids 7,394,100+; factorial rows
7,394,000+). Each run gets the inline Uncheatable evaluation and the native Table-9 evaluation. CPU parent in
us-east5-a, v6e-8 children in us-east5-b, all paths on gs://marin-us-east5, max_concurrent 18.

Analysis plan: fit Table-9 mean (and Uncheatable) on the 16 corners with main effects and the ten two-factor
interactions (16 parameters incl. intercept on 16 corners is saturated; use the centre replicates and the
three-factor aliases for error, or fit main effects + interactions with the effect SE of 0.0019 BPB from the
0.0038 repeat SD). Report every effect in BPB per ±δ with its SE; a two-factor interaction beyond about
0.005 BPB is a measured synergy the O(M) surrogates cannot represent. Compare the centre pair with the 26-run
mean 1.0639.

Evidence: `design.csv`, `candidate_weights.csv` (sha256
`ab1fcd8d62ff10f8b78bf94d0e490ecbc23d58be7e314d4c0d95ccea49b63b08`), `candidate_weights_replicate.csv`
(`52154d7264dfa2929160e8110613b2289d55e2bdd1b963ee5a8d6c989359fe93`), `README.md`, `launch_dry_run/`,
`submission/` (launch command, safety log with `--expected-child-zone us-east5-b`, dry-run log, three
launcher tests, lint, submit log). Fieldbook experiment `exp_01m1vh5s802cj685fbzcx3w475`, parent job
`job_01m1vh5sk7jrcjkcahhtj3p5z9`.
