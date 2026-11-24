# Replay low-value classifier v1

The CPU structural scorer catches 49.5% of low-value documents at 2.62%
natural-document false-drop on a frozen, model-labeled replay test. It is a
binary content filter; its score does not measure factual correctness or general
writing quality. [Usage](../references/low-value-scoring.md) and the
[research logbook](https://github.com/marin-community/marin-logbooks/issues/52)
describe the interface and reproducible experiment.

## Model and calibration

The model uses 146 deterministic structural features from merged head, middle
and tail windows of at most 2,048 tokens each. Features include compression,
character/line statistics, repetition, encoded payloads, markup and trajectory
markers. A 160-tree gradient-boosted head has depth 3, minimum leaf 10 and learning
rate .05. Inference requires the existing Marin CPU/tokenizer stack and NumPy;
scikit-learn is needed only by the archived training program.

Structural and Harrier-fusion heads tied the mean of their training-only
validation recall at the 2% and 5% false-drop targets (.4945); their average
precision was .7615/.7693.
Embedding-only reached .3626. The predeclared cost rule keeps structural when its mean recall is within
two percentage points of the best head, avoiding mandatory embedding compute.
No embedding inference is required by the released scorer.

Drop when `low_value_score >= cutoff`:

| Calibration target | Cutoff | Test low-value recall | Test natural false-drop | False-drop 95% interval |
| --- | --- | --- | --- | --- |
| 2% | 0.5949392992426772 | 71/204 = 34.80% | 8/687 = 1.16% | 0.59–2.28% |
| 5% | 0.4220386490525259 | 101/204 = 49.51% | 18/687 = 2.62% | 1.66–4.10% |

The 5% calibration point is the recommended replay operating point. The 2% point
provides a more conservative option. Targets describe calibration constraints;
observed test rates differ and neither rate is a guarantee on another corpus.

| Scorer | Recall at 2% calibration target | Actual natural false-drop | Recall at 5% calibration target | Actual natural false-drop | Test AP |
| --- | --- | --- | --- | --- | --- |
| Structural model | 34.80% | 1.16% | 49.51% | 2.62% | .7363 |
| Native rules | 13.73% | 0.15% | 18.14% | 2.91% | .3754 |
| Total-share rules | 15.20% | 0.29% | 17.65% | 2.77% | .3703 |
| Junkgate2 | 3.43% | 1.75% | 10.78% | 5.68% | .3319 |

At the 5% point, the paired recall gain over junkgate2 is 38.73 percentage points
(bootstrap 95% interval 31.37–46.57), and over native rules is 31.37 points
(24.02–38.73). Both baselines use the same calibration and test documents.
The native-rule ranking preserves its .3 machine-junk versus .6 total-share
priority. Handwritten rules classify each 4,096-character chunk as machine junk,
other low-value data, or unflagged content. Total-share score is the fraction of
flagged chunks. Native score is `max(total_share/.6, machine_share/.3)`; larger
values are riskier, and score 1 reproduces the published fixed rule. The native
ranking was added after the first test evaluation without changing the learned
model or its cutoffs. Both variants are retained.

## Labels and split

The sample has 3,600 distinct rows from 388 replay cluster/quality/length caches,
with approximately equal numbers of document rows per nonempty cache. Each
cache is one domain cluster, prior Harrier quality bucket (q0–q4), and length
class (at most or above 65,536 tokens); small caches are capped at their available
rows. These quality buckets define sampling strata, not the new binary labels. The existing 400-document seed
and matching visible content are development-only. Exact visible-content groups
were assigned by SHA256 before annotation: 2,038 training, 655 calibration and
907 test rows. Usable conclusive labels total 2,009/643/891 respectively.
Forty-two rows have no usable final annotation and 15 abstain.

Together GLM-5.3-Flash labeled the views at temperature .7 and concurrency 8,
with durable request receipts, disabled SDK retries and no replay of uncertain
requests. An independent GPT-6 Astra panel labeled 300 fresh documents. Of 296
usable pairs, 291 are conclusive in both: binary agreement 93.81%, Cohen's kappa
.7983. Category agreement over all pairs is 89.19%. Fresh-context Astra
adjudicated 60 disagreement or tokenizer-marker audit cases. Agreement is not a
measurement of human-ground-truth accuracy; checker and adjudicator share a
model family.

“Natural” is the conclusive negative category used as the false-drop
denominator. The rubric protects useful code, math, multilingual prose and
legible OCR.
Low-value categories are structured dumps, machine output, dominant raw markup
or tables, damaged OCR, embedded trajectories/benchmark bundles, repetitive
templates and explicit generated QA/instruction artifacts. Polished style alone
does not establish model authorship. A short QA tail does not by itself invalidate
coherent primary content. Document judgments are based only on recorded views.

Model choice used an inner training validation split. Cutoffs used only the
643 calibration documents. Learned parameters and cutoffs were published in
[pretest commit 90d61498a6](https://github.com/marin-community/marin-logbooks/tree/90d61498a6/logbooks/52-replay-quality-classifier/selected)
before opening test labels. Packaging subsequently bound the exact replay
tokenizer: it renames two reserved IDs to tool-call markers. Neither ID occurs
in any sampled view; parameters, features, labels and cutoffs are unchanged.

## Artifacts

The versioned directory is
`s3://marin-us-east-02a/marin/datakit/models/quality/replay_low_value_v1/2026.11.15.2/`
([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fdatakit%2Fmodels%2Fquality%2Freplay_low_value_v1%2F2026.11.15.2%2F)).
The same relative directory is mirrored in `marin-us-east-08a` for Exp3.
`SHA256SUMS.json` covers the model, calibration, tokenizer files, sampled scores,
label manifest, selection record, evaluation JSON/Markdown and PR curves.

| Artifact | SHA256 |
| --- | --- |
| `model.json` | `1396ad6cdaa9b7d1d9b5c0b2cabde124c4e3fe0e8c869da55e2e2b72bc7f545d` |
| `calibration.json` | `e650d4e9b71cc55a7feb8d54272d50a195f35a10e8894d9fea4fed119ee41c83` |
| `evaluation.json` | `cd555e720fefb6e49c549fbf9b4f7315ea5e5b7a9517c617d381fc9ca7c8c91f` |
| `label-manifest.json` | `66e4c3c12176d17ce451fff63b495616a0fbb7a6bd1abaf88fee2cd17d71e5d5` |
| Replay `tokenizer.json` | `6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45` |

The manifest contains sample IDs, split, category, binary labels and hashes of
the JSON lists of visible decoded windows. These are view fingerprints, not
full-document hashes. Text and annotation rationales remain private pending
explicit publication approval. The public report and curves are also archived
with the logbook; source code contains the rubric and experiment commands.

## Limits

These are unweighted rates on a balanced, capacity-capped cache sample. They do
not estimate full-corpus prevalence, retained tokens or downstream training
improvement. Bounded views miss some interior content; exact grouping does not
eliminate near duplicates. Intervals condition on this fitted model, calibration
and model-assisted judgments, excluding annotation error and all selection
uncertainty. Recalibrate and validate on a new corpus before adopting its cutoff.
Full replay scoring and Exp3 adoption are pending. A materialization run should
record its complete cache frame, model/calibration digests, per-cache coverage
and retention, and the data owner's cutoff decision. The released artifacts
support scoring now; those later receipts establish that the filter was applied
to a particular candidate.
