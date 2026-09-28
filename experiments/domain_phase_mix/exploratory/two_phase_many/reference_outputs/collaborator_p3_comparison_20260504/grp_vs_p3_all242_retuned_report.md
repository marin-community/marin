# GRP vs P3 Replication on Updated 300M Data

## What Was Run

This reproduces the collaborator gist `grp_vs_p3_comparison.py` against the current 300M raw metric matrix. The raw gist is a Marimo notebook; export mostly worked locally, but the final `_table` display cell failed under `marimo export`. I therefore reproduced the notebook logic in `replicate_grp_vs_p3_updated.py`.

## Data Alignment

- Updated raw signal rows: `242`.
- Rows used for this head-to-head comparison: `242` rows.
- Row mode: `all-signal`.
- Older GRP packet rows: `241`.
- Clean old-packet intersection: `240` rows.
- Updated-only rows: `baseline_olmix_loglinear_uncheatable_bpb, baseline_stratified`.
- GRP-packet-only rows: `baseline_olmix_loglinear`.

For `row_mode=all_signal`, the GRP packet is rebuilt directly from the current raw matrix, so the renamed Olmix row and `baseline_stratified` are included instead of relying on the older packet's row list.

## IRT Target

The target is the collaborator's task aggregate:

- Start from BPB-like columns under `eval/uncheatable_eval/`, `lm_eval/`, `mcq_smooth/`, and `teacher_forced/`.
- Drop aggregate BPBs and two known-bad/special columns: `teacher_forced/gsm8k_5shot_answer_hash/bpb`, `mcq_smooth/sciq_5shot/bpb`.
- For MCQ-style `lm_eval/` and `mcq_smooth/` tasks, use `choice_logprob` instead of BPB when available.
- Orient every item as higher-is-better, z-score by the swarm, estimate variable-subset noise shares, and fit a nonnegative anchored factor model.
- Horn parallel analysis selected `5` factor(s) on the aligned data.
- Selected task/proxy item count: `27`.

## Models

**GRP no-L2.** Uses the standalone `grp_no_l2_exact.py` implementation from the collaborator packet. The GRP target is `-IRT` because the GRP optimizer minimizes its target. In this run, GRP nonlinear parameters were `retuned`.

**P3.** Uses:

```text
yhat = alpha0
     + sum_d beta_d * (w0_d + eta * w1_d)^a
     - gamma0 * sum_d (c0_d * w0_d)^p
     - gamma1 * sum_d (c1_d * w1_d)^p
```

`eta`, `a`, and `p` are chosen by the collaborator's nested 5x5 CV grid; the linear head is ridge; the reported optimum is the mean over 200 bootstrap Frank-Wolfe simplex argmaxes.

## Fit Results

- GRP in-sample R2 on `-IRT`: `0.7639`.
- GRP retuned: `True`.
- P3 nested-CV R2 on `IRT`: `0.7967`.
- P3 full-data in-sample R2 on `IRT`: `0.8613`.
- P3 selected hyperparameters: `eta=10`, `a=0.5`, `p=1.5`, `ridge_alpha=10`.

## Optimum Agreement

- Pearson total weight: `+0.652`.
- Pearson phase-0 weight: `+0.088`.
- Pearson phase-1 weight: `+0.541`.
- Pearson total epochs: `+0.884`.
- L1 distance between total-weight recommendations: `1.335`.

## Top Domains

| domain                                     |   grp_total_weight |   p3_total_weight |   grp_total_epochs |   p3_total_epochs |
|:-------------------------------------------|-------------------:|------------------:|-------------------:|------------------:|
| dolma3_cc/literature_high                  |        0.267983    |         0.222254  |        5.2347      |          2.23251  |
| dolmino_synth_code                         |        0.298318    |         0.137369  |       80.032       |         10.1211   |
| dolmino_stack_edu_fim                      |        0.246943    |         0.146739  |        2.3617      |          1.96832  |
| dolmino_common_crawl_hq                    |        0.0498097   |         0.29595   |        0.0531467   |          1.12676  |
| dolmino_synth_qa                           |        0.0845706   |         0.238211  |        0.204268    |          2.01875  |
| dolma3_stack_edu                           |        0.175367    |         0.129873  |        1.65726     |          1.78384  |
| dolma3_cc/entertainment_high               |        0.116408    |         0.119234  |        1.38759     |          1.25151  |
| dolma3_cc/history_and_geography_high       |        0.149521    |         0.0549734 |        5.59519     |          0.928656 |
| dolmino_olmocr_pdfs_hq                     |        0.0968502   |         0.101971  |        0.660613    |          1.31938  |
| dolma3_cc/science_math_and_technology_high |        0.125264    |         0.0615874 |        0.658376    |          0.903355 |
| dolma3_arxiv                               |        0.0738768   |         0.0763899 |        3.33403     |          4.04293  |
| dolma3_cc/crime_and_law_high               |        0.0791683   |         0.0350176 |        0.723886    |          0.648224 |
| dolma3_cc/finance_and_business_high        |        0.0143174   |         0.0758138 |        0.133185    |          0.677847 |
| dolma3_cc/science_math_and_technology_low  |        1.35587e-10 |         0.0889614 |        1.87035e-09 |          1.44827  |
| dolmino_synth_math                         |        0.0664444   |         0.016762  |       15.4076      |          1.30255  |
| dolma3_cc/art_and_design_high              |        0.0557806   |         0.0237814 |        2.4722      |          0.523939 |
| dolmino_synth_instruction                  |        0.0421331   |         0.0252216 |        3.11205     |          2.4743   |
| dolma3_cc/literature_low                   |        0.035362    |         0.021194  |        0.650144    |          0.610201 |
| dolma3_cc/health_high                      |        0.0197733   |         0.0291762 |        0.229764    |          0.323647 |
| dolma3_cc/crime_and_law_low                |        9.01877e-11 |         0.0329262 |        2.22656e-09 |          0.866137 |

## Interpretation

P3 is not a generic regression; it is a deliberately simplified GRP-style law. It keeps the phase-1 exposure multiplier and an explicit concentration penalty, but drops retained exposure, CC-pair aggregation, family totals, per-family curvature, per-family thresholds, and NNLS sign constraints. The useful question is therefore not whether it is more expressive than GRP; it is whether the simpler inductive bias is better matched to this task-IRT target.

If GRP is retuned, this is the stronger comparison requested by Calvin: P3 tuned on IRT versus GRP nonlinear body retuned on IRT, both fit on the same current rows.
