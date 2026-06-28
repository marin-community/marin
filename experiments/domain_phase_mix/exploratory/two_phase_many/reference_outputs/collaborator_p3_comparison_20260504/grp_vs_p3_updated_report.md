# GRP vs P3 Replication on Updated 300M Data

## What Was Run

This reproduces the collaborator gist `grp_vs_p3_comparison.py` against the current 300M raw metric matrix. The raw gist is a Marimo notebook; export mostly worked locally, but the final `_table` display cell failed under `marimo export`. I therefore reproduced the notebook logic in `replicate_grp_vs_p3_updated.py`.

## Data Alignment

- Updated raw signal rows: `242`.
- Older GRP packet rows: `241`.
- Clean intersection used for this head-to-head comparison: `240` rows.
- Updated-only rows: `baseline_olmix_loglinear_uncheatable_bpb, baseline_stratified`.
- GRP-packet-only rows: `baseline_olmix_loglinear`.

Using the intersection avoids silently assigning missing IRT targets to the older GRP packet row names.

## IRT Target

The target is the collaborator's task aggregate:

- Start from BPB-like columns under `eval/uncheatable_eval/`, `lm_eval/`, `mcq_smooth/`, and `teacher_forced/`.
- Drop aggregate BPBs and two known-bad/special columns: `teacher_forced/gsm8k_5shot_answer_hash/bpb`, `mcq_smooth/sciq_5shot/bpb`.
- For MCQ-style `lm_eval/` and `mcq_smooth/` tasks, use `choice_logprob` instead of BPB when available.
- Orient every item as higher-is-better, z-score by the swarm, estimate variable-subset noise shares, and fit a nonnegative anchored factor model.
- Horn parallel analysis selected `5` factor(s) on the aligned data.
- Selected task/proxy item count: `27`.

## Models

**GRP no-L2.** Uses the standalone `grp_no_l2_exact.py` implementation from the collaborator packet. The 9 nonlinear parameters are kept at the included no-L2 checkpoint originally tuned for uncheatable BPB; only the NNLS linear head is refit to `-IRT` because the GRP optimizer minimizes its target.

**P3.** Uses:

```text
yhat = alpha0
     + sum_d beta_d * (w0_d + eta * w1_d)^a
     - gamma0 * sum_d (c0_d * w0_d)^p
     - gamma1 * sum_d (c1_d * w1_d)^p
```

`eta`, `a`, and `p` are chosen by the collaborator's nested 5x5 CV grid; the linear head is ridge; the reported optimum is the mean over 200 bootstrap Frank-Wolfe simplex argmaxes.

## Fit Results

- GRP linear-head refit in-sample R2 on `-IRT`: `0.7676`.
- P3 nested-CV R2 on `IRT`: `0.7785`.
- P3 full-data in-sample R2 on `IRT`: `0.8633`.
- P3 selected hyperparameters: `eta=10`, `a=0.5`, `p=1.5`, `ridge_alpha=1`.

## Optimum Agreement

- Pearson total weight: `+0.649`.
- Pearson phase-0 weight: `+0.074`.
- Pearson phase-1 weight: `+0.397`.
- Pearson total epochs: `+0.256`.
- L1 distance between total-weight recommendations: `1.367`.

## Top Domains

| domain                                     |   grp_total_weight |   p3_total_weight |   grp_total_epochs |   p3_total_epochs |
|:-------------------------------------------|-------------------:|------------------:|-------------------:|------------------:|
| dolma3_cc/literature_high                  |         0.241773   |         0.196997  |        2.88321     |          1.97094  |
| dolmino_common_crawl_hq                    |         0.100188   |         0.330338  |        0.103925    |          1.24959  |
| dolmino_synth_qa                           |         0.160881   |         0.195958  |        0.491862    |          1.61135  |
| dolmino_stack_edu_fim                      |         0.165191   |         0.136711  |        6.23165     |          1.83978  |
| dolmino_olmocr_pdfs_hq                     |         0.173761   |         0.0878922 |        4.1627      |          1.14546  |
| dolma3_cc/science_math_and_technology_high |         0.196715   |         0.0602983 |        1.15965     |          0.853325 |
| dolma3_cc/history_and_geography_high       |         0.167775   |         0.0584333 |        1.66618     |          0.945821 |
| dolmino_synth_code                         |         0.095487   |         0.129958  |        8.93011     |          9.66881  |
| dolma3_stack_edu                           |         0.099484   |         0.117463  |        2.25777     |          1.62284  |
| dolma3_cc/entertainment_high               |         0.0991058  |         0.106387  |        1.18299     |          1.06792  |
| dolma3_cc/crime_and_law_high               |         0.127738   |         0.042787  |        1.29941     |          0.721706 |
| dolma3_cc/art_and_design_high              |         0.110994   |         0.0256929 |        4.88368     |          0.540871 |
| dolma3_arxiv                               |         0.0607106  |         0.0661    |        8.9607      |          3.6165   |
| dolma3_cc/finance_and_business_high        |         0.0204532  |         0.0972186 |        0.189965    |          0.842824 |
| dolmino_synth_instruction                  |         0.0633152  |         0.0269764 |       17.3561      |          2.58822  |
| dolma3_cc/science_math_and_technology_low  |         8.6822e-09 |         0.0800178 |        2.70888e-07 |          1.31467  |
| dolma3_cc/health_high                      |         0.00428392 |         0.0433832 |        0.0497237   |          0.464349 |
| dolma3_cc/industrial_high                  |         0.0304087  |         0.0116331 |        1.87905     |          0.33744  |
| dolmino_synth_math                         |         0.0247236  |         0.016315  |        5.71408     |          1.27295  |
| dolma3_cc/games_high                       |         0.00183502 |         0.0351911 |        0.0410872   |          0.560878 |

## Interpretation

P3 is not a generic regression; it is a deliberately simplified GRP-style law. It keeps the phase-1 exposure multiplier and an explicit concentration penalty, but drops retained exposure, CC-pair aggregation, family totals, per-family curvature, per-family thresholds, and NNLS sign constraints. The useful question is therefore not whether it is more expressive than GRP; it is whether the simpler inductive bias is better matched to this task-IRT target.

The caveat is important: this comparison does not retune GRP's nonlinear parameters for IRT. It tests the collaborator's exact claim: included GRP nonlinear body plus refit NNLS head versus P3 tuned directly on the IRT aggregate.
