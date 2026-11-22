# Two-stage decontamination with an encoder classifier (proof of concept)

This directory holds a proof of concept for removing eval items from pretraining data. It works in
two stages and uses one rule for all evals, with no values set per eval:

1. **Stage 1, the wide net.** Each eval item gets one key: the rarest 20-character window of its
   text stream. The rarity is the count of records that hold the window in a reference sample. A
   record that holds an item's key is a candidate for that item.
2. **Stage 2, the classifier.** A fine-tuned multilingual encoder reads the pair
   `(left, right)`, where `left` is the item (question, choices, answer) and `right` is the record
   text around the key. It gives two probabilities: that `right` holds the question (`label_q`),
   and that `right` holds the question with its answer (`label_qa`).

The text stream of a text is its lowercase letters and digits, after NFKC folding and with LaTeX
commands removed (`text_keys.py`). Spacing, punctuation, Markdown and markup do not change it.
Stage 1 reads each record three ways: as it is, and with its JSON string escapes read once and
twice. Without this, a copy stored in a JSON dump loses about a third of its windows, because
`\n` before a word reads as a LaTeX command.

## Pipeline

| Step | Module | Output |
|---|---|---|
| Item table: one row per required eval record (question, choices, answer, passage) | `items.py` | `items/<eval>.parquet` |
| Key choice: count candidate windows in reference shards, keep the rarest | `rarest_key.py prepare, count, choose` | `keys.parquet` |
| Candidate volume, and a sample of key-1 candidates, on measure shards | `rarest_key.py measure` | `measure/*.json`, `sample/*.parquet` |
| Candidate pairs: the record text around each sampled key hit | `pairs.py items, extract` | `pairs/*.parquet` |
| Synthetic pairs with exact labels | `synthetic.py generic, context` | one parquet |
| Judge label sets from real candidates, and training rows from two judges | `labels.py sample, merge`, `RUBRIC.md` | batches, one parquet |
| Silver negatives: real candidates with low word overlap | `labels.py silver` | one parquet |
| Fine-tuning (one H100) | `train.py` | `model/`, `results.json`, `test_scores.parquet` |
| Scoring pairs with a saved model | `score.py` | one parquet |
| Stage-1 recall on judged copies | `key_recall.py` | `key_recall.json` |
| Planted copies in 14 forms, stage 1 and stage 2 (gate G2) | `planted.py` | `stats/*.json`, `pairs/*.parquet` |

`sources.json` lists the 160 normalized sources of the sample. `rarest_key.py` draws one measure
shard per source (seed 7) and two other reference shards per source (seed 11), in list order. A
validation sample uses another seed and `--exclude-shards` with the first run's `shards.json`, so it
shares no shard with it; copy the first run's `keys.parquet` into its output to keep the keys frozen.

## Commands

All steps run on Iris (`cw-us-east-02a`), at interactive priority. `$T` is a scratch prefix, for
example `s3://marin-us-east-08a/tmp/ttl=7d/$USER`. GPU jobs need `--disk 100GB`: the default 5 GB
disk is too small for the model and the pairs.

```bash
# Item table (4 parts, about 15 minutes each).
iris --cluster=cw-us-east-02a job run --extra=cpu --extra=lm_eval --memory 32GB --cpu 4 --disk 200GB \
  --enable-extra-resources -e MARIN_PREFIX s3://marin-us-east-02a/marin --job-name decon-items-0 -- \
  python -m experiments.datakit.decontam.encoder.items \
  --manifest s3://marin-us-east-02a/marin/datakit/decontam/evals/aa-index-v4.1.1-v6/lmh/_manifest.json \
  --output $T/items-table --part 0 --parts 4

# Stage 1, one step after the other. prepare and choose run in the job itself (32 GB); count and
# measure start one Zephyr worker per shard (--ram per worker).
STAGE1=(--sources experiments/datakit/decontam/encoder/sources.json --input-root s3://marin-us-east-02a/marin/normalized
        --items $T/items-table/items --output $T/stage1)
for step in prepare count choose measure; do
  iris --cluster=cw-us-east-02a job run --extra=cpu --memory 32GB --cpu 4 --disk 50GB --enable-extra-resources \
    --job-name decon-$step -- python -m experiments.datakit.decontam.encoder.rarest_key $step $STAGE1 --ram 24g
done

# Candidate pairs (items: one job; extract: one Zephyr worker per sample file), then scores.
PAIRS=(--stage1 $T/stage1 --items $T/items-table/items --output $T/pairs)
iris --cluster=cw-us-east-02a job run --extra=cpu --memory 32GB --cpu 4 --disk 50GB --enable-extra-resources \
  --job-name decon-pairs-items -- python -m experiments.datakit.decontam.encoder.pairs items $PAIRS
iris --cluster=cw-us-east-02a job run --extra=cpu --job-name decon-pairs-extract -- \
  python -m experiments.datakit.decontam.encoder.pairs extract $PAIRS
iris --cluster=cw-us-east-02a job run --extra=gpu --gpu H100x1 --enable-extra-resources --memory 64GB --cpu 8 --disk 100GB \
  -e TOKENIZERS_PARALLELISM false --job-name decon-score -- python -m experiments.datakit.decontam.encoder.score \
  --model $T/train/<tag>/model --pairs $T/pairs --output $T/pairs/scored.parquet

# Training data (CPU jobs). EXCLUDE lists the item files of every label set, so that no test item
# is seen in training through synthetic or silver pairs.
EXCLUDE=($T/data/set11.parquet $T/data/set12.parquet)
CPU=(--extra=cpu --memory 16GB --cpu 2 --disk 20GB --enable-extra-resources)
iris --cluster=cw-us-east-02a job run $CPU --job-name decon-synth-generic -- \
  python -m experiments.datakit.decontam.encoder.synthetic generic --items $T/items-table/items \
  --hosts "s3://marin-us-east-02a/marin/normalized/*/outputs/main/part-00000-*.parquet" \
  --exclude-items $EXCLUDE --output $T/data/synthetic_generic.parquet
iris --cluster=cw-us-east-02a job run $CPU --job-name decon-synth-context -- \
  python -m experiments.datakit.decontam.encoder.synthetic context --candidates $T/pairs/scored.parquet \
  --exclude-items $EXCLUDE --output $T/data/synthetic_ctx.parquet
iris --cluster=cw-us-east-02a job run $CPU --job-name decon-silver -- \
  python -m experiments.datakit.decontam.encoder.labels silver --scored $T/pairs/scored.parquet \
  --exclude-items $EXCLUDE --output $T/data/silver.parquet

# Training (about 25 minutes on one H100 for 2 epochs of about 80k pairs).
iris --cluster=cw-us-east-02a job run --extra=gpu --gpu H100x1 --enable-extra-resources --memory 64GB --cpu 8 --disk 100GB \
  -e TOKENIZERS_PARALLELISM false -e OMP_NUM_THREADS 8 --job-name decon-train -- \
  python -m experiments.datakit.decontam.encoder.train --tag <tag> --model jhu-clsp/mmBERT-base \
  --data $T/data --synthetic synthetic_generic.parquet --extra $T/data/set11.parquet --extra $T/data/set12.parquet \
  --extra $T/data/silver.parquet --extra $T/data/synthetic_ctx.parquet --output $T/train
```

Label sets run outside Iris, because the judges are coding agents. `labels.py sample` writes
batches of tasks; each judge reads `RUBRIC.md` and writes `<set>/<judge>/batch_*.jsonl`; and
`labels.py merge` keeps the pairs that both judges agree on. `--data` must also hold
`judged_pairs_v2.parquet`, the judged pairs of the earlier block-rule candidates.

## Model

- Base model: `jhu-clsp/mmBERT-base` (ModernBERT architecture, multilingual), with two sigmoid outputs
  and a binary cross-entropy loss. Inputs are cut to 1,024 tokens; `left` is cut to 1,500 characters
  and `right` to 6,000.
- `left` is the question, a `[choices]` line and an `[answer]` line, then a hint: the question and
  choice words that `right` lacks (`[missing]`, up to 32 words) and the answer words that `right`
  lacks (`[answer missing]`). The hint raised the set-11 test AUC from 0.931 to 0.947. Keeping the
  answer words out of the question hint, and showing the choices, raised the pooled real-test AUC
  from 0.950 / 0.960 to 0.971 / 0.968 (two seeds each).
- The threshold is the lowest score that keeps a precision of at least 0.95 on a dev split: one in
  four train items of the label sets. Silver, synthetic and the older judged pairs never set it.

## Measured results

All numbers come from cluster runs of these modules or of their earlier versions. A *candidate* is a
(record, item) pair that stage 1 sends to stage 2. *Real test* pairs are the test splits of the
judge-labeled candidate sets; their items never appear in training.

**Stage 1.** The measure sample holds one shard per source, so a projection to the catalog weights
each shard by its source's shard count. Calibration (160 shards, 27.4M records) and validation (128
new shards, 20.8M records) agree on the weighted numbers (catalog: 1.9e10 records):

| Key | Records with a candidate | Candidates in the catalog |
|---|---|---|
| First 20-character window | 24.1% | - |
| Rarest window (key 1) | 0.09-0.10% | 4.2e7 (0.0022 per record) |
| Key 1 or key 2 | - | 8.2e9 |

At 441 pairs per second on one H100 (mmBERT-base, 1,024 tokens), key 1 costs about 27 H100-hours
for the catalog. Stage 1 itself costs about 24,000 CPU worker-hours for the catalog with the JSON
readings, and 17,500 without them. The readings do not change the load. An unweighted projection
(every source equal) gives 6.9e8 key-1 candidates: small sources have higher candidate rates per
record, and the unweighted projection counts them as much as large ones.

Key recall on judged copies from earlier candidate sets: 0.963 with the plain reading on item table
v1. On table v3 it is 0.882 for the 237 judged copies that still match an item, and 0.932 with the
JSON readings. The other misses are long items whose judged excerpt is shorter than the item.

**Stage 2.** Iterations on real test pairs:

| Iteration | Change | Real test | Question AUC |
|---|---|---|---|
| 4 | + 20,000 silver negatives, + set 11 (part), hint | set 11 (8 copies) | 0.947 (no hint: 0.931) |
| 5 | mmBERT-base for mmBERT-small | set 11 (17 copies) | 0.948 (small: 0.924) |
| 6 | + context synthetic pairs (items inside real candidate text) | set 11 | 0.972; repeat run 0.948 |
| 8 | + set 12 | sets 11-12 (69 copies) | 0.963-0.971 |
| 9 | item input: choices shown, answer words out of the question hint | sets 11-12 | 0.971 / 0.968 (old input 0.950 / 0.960) |
| 10 | + set 13, dev split from the label sets | sets 11-13 (151 copies) | 0.961 / 0.973 |
| 11 | answer labels only for layouts that mark the answer; no context variant in a candidate that holds the item | sets 11-13 | 0.979 / 0.971 (answer output 0.958 / 0.943, was 0.922 / 0.923) |

Rejected on the way: 512 and 2,048 tokens, repeated judge-labeled pairs, a reference-frequency hint
(set-11 AUC 0.936 against 0.972), one output instead of two, and averaging two seeds.

The reference model is iteration 11, seed 0, chosen by pooled AUC before the validation labels
existed. Stage 2 sees only key-1 candidates, so the key-1 view is the one that matters for a run;
about 60% of the real test pairs came from key 2 in older samples (bound: Jeffreys 5th percentile):

| Score at least | All test pairs (691, 151 copies): marks, precision [bound], recall | Key-1 pairs (260, 118 copies) | Key-1 without DROP (173, 54 copies) |
|---|---|---|---|
| 0.5 | 171, 0.81 [0.76], 0.92 | 133, 0.81 [0.75], 0.92 | 65, 0.69 [0.59], 0.83 |
| 0.9 | 128, 0.89 [0.84], 0.75 | 105, 0.87 [0.80], 0.77 | 42, 0.69 [0.57], 0.54 |
| 0.99 | 85, 0.95 [0.90], 0.54 | 70, 0.94 [0.88], 0.56 | 18, 0.78 [0.59], 0.26 |

Question AUC is 0.979 on all test pairs and 0.940 on key-1 pairs; the key-1 negatives are near
misses. Seed 1 of the same recipe gives key-1 precision 0.92 [0.86] at 0.9 and 0.95 [0.89] at 0.99.

The answer output has AUC 0.958 (57 copies show the answer); at 0.9 it gives precision 0.94 and
recall 0.54. The test pairs are a stratified sample: most come from candidates whose record holds at
least 90% of the question's words, where the copies are. On set 13, which the iteration-9 model never
saw, weighting the strata to the whole candidate sample gives that model a precision of 0.95 at its
dev threshold (0.94) and 0.98 at 0.99. Recall outside the high-overlap stratum rests on 2 copies among
100 labeled low-score pairs, so it is not measured.

**Validation on new shards.** Set 14 comes from 128 shards that share none with calibration, with
the keys, the models and the thresholds frozen before labeling. Both judges agree on 338 of 356
pairs (160 copies, 106 of them DROP). Precision is weighted to the validation candidate pool, and
the bound is an item-clustered stratified bootstrap 5th percentile:

| Model | Score at least 0.9: precision [bound], high-band recall | At least 0.99 | At least 0.99, without DROP |
|---|---|---|---|
| Iteration 11, seed 0 (reference) | 0.958 [0.928], 0.86 | 0.971 [0.941], 0.62 | 0.954 [0.871], 0.37 (21 marks) |
| Iteration 10, seed 1 | 0.891 [0.847], 0.84 | 0.980 [0.954], 0.64 | 1.000, 0.31 (16 marks) |

154 of the 160 copies have word overlap of at least 0.9 (the high band), and recall is reported
there. The low band holds 1 copy in 99 labeled pairs.

**Exact rules on the same pairs.** On the real test pairs (697 pairs, 152 copies) and on validation
set 14 (338, 160), unweighted, the exact rules do worse than the classifier on both precision and recall:

| Rule | Real test: precision / recall | Validation: precision / recall |
|---|---|---|
| Block rule (`marin.datakit.item_match` in PR #10836: question, then every choice) | 0.80 / 0.45 | 0.91 / 0.68 |
| Exact question only | 0.65 / 0.52 | 0.81 / 0.69 |
| Classifier at 0.9 | 0.89 / 0.76 | 0.96 / 0.85 |
| Classifier at 0.99 | 0.95 / 0.53 | 0.97 / 0.62 |

Outside DROP, most judged copies are edited or cut at the excerpt edge, and the block rule finds
about 11% of them.

**Planted copies (gate G2).** `planted.py` plants up to 50 items per eval (41,350 items over 827
evals) into real record text in 14 forms and checks stage 1. Stage 1 finds every planted copy in
the plain, parenthesized, extra-option, no-line-break and JSON-escaped (once and twice) forms.
It misses three kinds:
- **Short items in a layout without letters (recall 0.858 in the sample).** Markdown, numbered,
  JSON and question-only layouts. A question shorter than 20 stream characters is keyed on the
  question with its lettered choices (2.9% of all items: ZhoBLiMP, BBQ), so its key needs the letters.
- **Shuffled choices (0.952).** The same short items.
- **No punctuation (0.987) and upper case (0.989).** LaTeX commands without their backslash become
  words, and some casing does not round-trip (Turkish).

206 sampled items (0.5%) have no key. With the excerpt read as the pipeline reads it, the frozen
classifier marks 0.94-0.96 of planted copies at 0.99 in most layouts (escaped 0.91-0.93, ASCII JSON
0.81, no punctuation 0.71). It rejects a changed number or sign (2% marked). A changed or missing
choice and choices far from the question are marked (83-97%), because the record still holds the
whole question.

Most remaining false marks are variants that drop or add one condition ("of the first half"), or that
show the same passage with another question. Most missed copies are edited copies (reworded
problems, other versions of a code file) and short CoLA sentences.

## Limits

- **The precision target is met only in part.** The goal is a precision whose lower bound is at
  least 0.95 on real candidates. On the validation shards, iteration 10 seed 1 meets it at score 0.99
  over all evals (0.954), and the reference model just misses it (0.941); outside DROP there are too
  few marks for a bound. On key-1 test pairs the reference model reaches a point precision of 0.94 at
  score 0.99, with a bound of 0.88, at 56% recall; without DROP it is 0.78. The dev split (261
  pairs, 45 copies) puts the 0.95 point at a score of 0.991; on key-1 test pairs that threshold
  gives precision 0.94 [0.88] at 55% recall. A bound of 0.95 needs about 290 labeled marks at a
  true precision of 0.97, and about 1,200 at 0.96.
- **One training run is noisy.** The same recipe and data, with only GPU nondeterminism between
  runs, gave set-11 test AUCs of 0.973 and 0.948. Compare recipes on the pooled real test with more
  than one seed.
- **Real labels are few.** Sets 11 to 13 hold 326 agreed copies, and 151 of them are in the test
  splits; 127 of the 179 copies in set 13 are DROP questions. A safety classifier stopped one Claude
  judge in set 12, so 288 of its 859 pairs have only one judge and were dropped. When set 12 was labeled, the item table gave every DROP item the
  placeholder answer `number,date,spans,worker_id,hit_id` (fixed in `items.py`), so the DROP answer
  labels in set 12 lean to "not shown".
- **Silver negatives are noisy.** Of the judged pairs with word overlap below 0.8, 5 of 561 (0.9%)
  were copies, so about 1% of the silver negatives are copies.
- **A short question of a passage-based eval is generic.** The item question is the prompt without
  its passage. A DROP question such as "How many yards was the longest field goal?" is then a common
  sentence, and the rubric labels a record that holds it as a copy, even when the record is about
  another game. This follows the recall-first policy (remove any record that holds a required
  question), so the removal budget has to catch these evals.
- **One key covers one place in a long item.** A record that copies only part of a long item (an
  Uncheatable document, a long code file) holds the item's key only if the copied part contains it.
- **GPQA choice order is random.** lm-eval shuffles GPQA choices without a seed, so the question
  text, which shows the choices, and the answer letter change between builds of the item table. A key
  that crosses two choices misses a copy with another order.
- **Stage-1 recall comes from a biased sample.** It is measured on judged copies from earlier
  candidate sets and on items planted as JSON. No unbiased sample of copies exists yet.
