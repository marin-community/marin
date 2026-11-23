# Two-stage decontamination with an encoder classifier (proof of concept)

This directory holds a proof of concept for removing eval items from pretraining data. It works in
two stages and uses one rule for all evals, with no values set per eval:

1. **Stage 1, the wide net.** Each eval item gets up to three key streams: its question; for a
   question shorter than a key, the question and its choices without letters; and its passage, when
   it has one. Each stream's key is its rarest 20-character window, where rarity is the count of
   records that hold the window in a reference sample. A stream other than the question keeps its key
   only when no reference record holds that window, so a generic stem adds no load. A record that
   holds an item's key is a candidate for that item.
2. **Stage 2, the classifier.** A fine-tuned multilingual encoder reads the pair `(left, right)`,
   where `left` is the item (passage, question, choices, answer) and `right` is the record text
   around the key. It gives two probabilities: that `right` holds the item (`label_q`), and that
   `right` holds it with its answer (`label_qa`). A record is removed when one of its pairs scores at
   or above the cutoff that the [cutoff rule](#cutoff-rule) chooses.

The text stream of a text is its lowercase letters and digits, after NFKC folding and with LaTeX
commands removed. Spacing, punctuation, Markdown and markup do not change it. The text streams, key
streams, key choice, key table and per-record candidates live in the library module
`marin.datakit.decon_keys`; this directory holds the pipelines, the item table, the label sets and
the classifier.
Stage 1 reads each record three ways: as it is, and with its JSON string escapes read once and
twice. Without this, a copy stored in a JSON dump loses about a third of its windows, because
`\n` before a word reads as a LaTeX command.

## What counts as a copy

`RUBRIC.md` is the labeling rule for judges and for the label sets. In short:

- The item is its question with its choices. The same question with changed, removed or added
  choices is a copy (`EDITED`, note "choices changed"); the order of the choices does not matter.
- A changed number, name, condition or asked quantity makes a `VARIANT`, not a copy.
- A short generic stem ("Which of the following is true?") is the item only together with its
  choices; the stem alone is `GENERIC`.
- An item with a passage is the passage and the question together. The passage alone, or the
  question alone, is `GENERIC`; the passage with another question is a `VARIANT`.

## Pipeline

| Step | Module | Output |
|---|---|---|
| Item table: one row per required eval record (question, choices, answer, passage) | `items.py` | `items/<eval>.parquet` |
| Key choice: count candidate windows of every key stream in reference shards, keep the rarest | `rarest_key.py prepare, count, choose` | `keys.parquet` (one row per item and stream) |
| Candidate volume, and a sample of key-1 candidates, on measure shards | `rarest_key.py measure` | `measure/*.json`, `sample/*.parquet` |
| Candidate pairs: the record text around each sampled key hit | `pairs.py items, extract` | `pairs/*.parquet` |
| Synthetic pairs with exact labels | `synthetic.py generic, context` | one parquet |
| Label sets from real candidates | `labels.py sample`, `judge.py`, `RUBRIC.md` | batches, `<model>/batch_*.jsonl` |
| Silver negatives: real candidates with low word overlap | `labels.py silver` | one parquet |
| Fine-tuning (one H100) | `train.py` | `model/`, `results.json`, `test_scores.parquet` |
| Scoring pairs with a saved model | `score.py` | one parquet |
| Planted copies in 14 forms, stage 1 and stage 2 (gate G2) | `planted.py` | `stats/*.json`, `pairs/*.parquet` |
| Whole catalog: locator table, scan (resumable), GPU scoring in parts, removal list | `catalog.py locate, scan, collect`, `candidates.py`, `catalog_score.py` | `pairs/`, `scores/`, `removed.parquet`, `summary.json` |

Pairs and label rows store an item id. Training, scoring and context synthesis read the item's texts
from the item table (`tables.item_texts`), so a fix of the table reaches every stored label.

`sources.json` lists the 160 normalized sources of the sample. `rarest_key.py` draws one measure
shard per source (seed 7) and two other reference shards per source (seed 11), in list order. A
validation sample uses another seed and `--exclude-shards` with the earlier runs' `shards.json`, so
it shares no shard with them; copy the first run's `keys.parquet` into its output to keep the keys
frozen.

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
ITEMS=$T/items-table/items

# Stage 1, one step after the other. prepare and choose run in the job itself (32 GB); count and
# measure start one Zephyr worker per shard (--ram per worker).
STAGE1=(--sources experiments/datakit/decontam/encoder/sources.json --input-root s3://marin-us-east-02a/marin/normalized
        --items $ITEMS --output $T/stage1)
for step in prepare count choose measure; do
  iris --cluster=cw-us-east-02a job run --extra=cpu --memory 32GB --cpu 4 --disk 50GB --enable-extra-resources \
    --job-name decon-$step -- python -m experiments.datakit.decontam.encoder.rarest_key $step $STAGE1 --ram 24g
done

# Candidate pairs (items: one job; extract: one Zephyr worker per sample file), then scores.
PAIRS=(--stage1 $T/stage1 --items $ITEMS --output $T/pairs)
iris --cluster=cw-us-east-02a job run --extra=cpu --memory 32GB --cpu 4 --disk 50GB --enable-extra-resources \
  --job-name decon-pairs-items -- python -m experiments.datakit.decontam.encoder.pairs items $PAIRS
iris --cluster=cw-us-east-02a job run --extra=cpu --job-name decon-pairs-extract -- \
  python -m experiments.datakit.decontam.encoder.pairs extract $PAIRS
GPU=(--extra=gpu --gpu H100x1 --enable-extra-resources --memory 64GB --cpu 8 --disk 100GB -e TOKENIZERS_PARALLELISM false)
iris --cluster=cw-us-east-02a job run $GPU --job-name decon-score -- python -m experiments.datakit.decontam.encoder.score \
  --model $T/train/<tag>/model --items $ITEMS --pairs $T/pairs --output $T/pairs/scored_<tag>.parquet

# Label set: sample pairs, judge them with GLM and DeepSeek (GLM_BULK_TOKEN from Secret Manager,
# project hai-gcp-models), and keep the pairs both judges agree on. Copy the batches to $T/labels/<set>.
python -m experiments.datakit.decontam.encoder.labels sample --set label/<set> --scored $T/pairs/scored_<tag>.parquet \
  --exclude-items $EXCLUDE
iris --cluster=cw-us-east-02a job run --extra=cpu -e GLM_BULK_TOKEN "$GLM_BULK_TOKEN" --job-name decon-judge -- \
  python -m experiments.datakit.decontam.encoder.judge --tasks $T/labels/<set> --output $T/labels/<set> \
  --relay /muchanem/glm53-relay-h
python -m experiments.datakit.decontam.encoder.labels merge --set label/<set> --labels $T/labels/<set> \
  --output $T/data/<set>.parquet

# Training data (CPU jobs). EXCLUDE lists the item files of every label set, so that no test item
# is seen in training through synthetic or silver pairs.
CPU=(--extra=cpu --memory 16GB --cpu 2 --disk 20GB --enable-extra-resources)
iris --cluster=cw-us-east-02a job run $CPU --job-name decon-synth-generic -- \
  python -m experiments.datakit.decontam.encoder.synthetic generic --items $ITEMS \
  --hosts "s3://marin-us-east-02a/marin/normalized/*/outputs/main/part-00000-*.parquet" \
  --exclude-items $EXCLUDE --output $T/data/synthetic_generic.parquet
iris --cluster=cw-us-east-02a job run $CPU --job-name decon-synth-context -- \
  python -m experiments.datakit.decontam.encoder.synthetic context --items $ITEMS \
  --candidates $T/pairs/scored_<tag>.parquet --exclude-items $EXCLUDE --output $T/data/synthetic_ctx.parquet
iris --cluster=cw-us-east-02a job run $CPU --job-name decon-silver -- \
  python -m experiments.datakit.decontam.encoder.labels silver --scored $T/pairs/scored_<tag>.parquet \
  --exclude-items $EXCLUDE --output $T/data/silver.parquet

# Training, two seeds (about 25 minutes each on one H100). --data also holds judged_pairs_v3.parquet,
# the judged pairs of the earlier block-rule candidates.
for seed in 0 1; do
  iris --cluster=cw-us-east-02a job run $GPU -e OMP_NUM_THREADS 8 --job-name decon-train-s$seed -- \
    python -m experiments.datakit.decontam.encoder.train --tag <tag>-s$seed --seed $seed --model jhu-clsp/mmBERT-base \
    --data $T/data --synthetic synthetic_generic.parquet --items $ITEMS --extra $T/data/<set>.parquet \
    --extra $T/data/silver.parquet --extra $T/data/synthetic_ctx.parquet --output $T/train
done

# Whole catalog: locator table, scan (one Zephyr worker per shard; records over 5,000,000 characters
# are read in overlapping pieces), scores on 30 GPUs, then the removal list at the cutoff.
C=$T/catalog
iris --cluster=cw-us-east-02a job run --extra=cpu --memory 32GB --cpu 4 --disk 50GB --enable-extra-resources \
  --job-name decon-catalog-locate -- python -m experiments.datakit.decontam.encoder.catalog locate \
  --stage1 $T/stage1 --items $ITEMS --output $C
iris --cluster=cw-us-east-02a job run --extra=cpu --job-name decon-catalog-scan -- \
  python -m experiments.datakit.decontam.encoder.catalog scan --sources experiments/datakit/decontam/encoder/sources.json \
  --input-root s3://marin-us-east-02a/marin/normalized --output $C --records all --ram 32g
for k in $(seq 0 29); do
  iris --cluster=cw-us-east-02a job run $GPU --no-wait --job-name decon-catalog-score-p$k -- \
    python -m experiments.datakit.decontam.encoder.catalog_score --output $C --scores $C/scores \
    --models $T/train/<tag>-s0/model $T/train/<tag>-s1/model --items $ITEMS --part $k --parts 30
done
iris --cluster=cw-us-east-02a job run --extra=cpu --memory 32GB --cpu 8 --enable-extra-resources \
  --job-name decon-catalog-collect -- python -m experiments.datakit.decontam.encoder.catalog collect \
  --output $C --scores $C/scores --cutoff 0.5
```

## Model

- Base model: `jhu-clsp/mmBERT-base` (ModernBERT architecture, multilingual), with two sigmoid
  outputs and a binary cross-entropy loss. Inputs are cut to 1,024 tokens; `right` is cut to 6,000
  characters.
- `left` has marked sections: `[passage]` (cut to 700 characters), `[question]`, `[choices]` and
  `[answer]`, then hints: the passage words that `right` lacks (`[passage missing]`), the question
  and choice words that `right` lacks (`[missing]`, up to 32 words), and the answer words that
  `right` lacks (`[answer missing]`).
- Training data: the judged label sets (including catalog candidates of the families that the first
  models flagged wrongly), generic synthetic pairs (copies in 14 layouts, edits, variants, sibling
  negatives: another instance of the same eval template, and for items with a passage: the passage
  alone, the passage with another question, the question and choices without the passage), context
  synthetic pairs (items placed in real candidate text), and silver negatives. Training rows on test
  or validation items are removed.

### Cutoff rule

The rule was fixed with a peer review before the iteration-15 results existed: score each pair with
the mean of two seeds' scores, and take the lowest cutoff in {0.05, 0.1, 0.2, ..., 0.9, 0.95, 0.99}
where the stratum-weighted precision on a validation set is at least 0.75 and its item-clustered
stratified bootstrap 5% bound is at least 0.65. The floor follows the policy of removing every copy at
a reasonable cost. The validation set that applies the rule must be drawn per eval family from catalog
candidates: draws from the measure shards held few flags of the families that fail in the catalog
(below). The removal check (set 20) then applies the same rule to the removed records themselves, at
the chosen cutoff and at 0.5, and the run keeps the lower one that passes. Set 23 chose 0.3; set 20
failed at 0.3 and passed at 0.5, so the run removes at 0.5.

## Measured results

All numbers come from cluster runs of these modules. A *candidate* is a (record, item) pair that stage 1
sends to stage 2. A *flag* is a candidate whose mean score reaches the cutoff. Labels come from two LLM
judges (GLM-5.3 and DeepSeek-v4.1) with `RUBRIC.md`; a pair is labeled only when both agree. On pairs
that the earlier Codex and Claude judges both labeled, the two LLM judges agree with them on 99.7%.

**Stage 1 (calibration shards, size-weighted to the catalog of 1.9e10 records).**

| Keys | Records with a candidate | Key-1 candidates in the catalog | Stage 2 per model |
|---|---|---|---|
| Question only | 0.092% | 4.2e7 | 27 H100-hours |
| Question, letter-free short question, passage (with the guard) | 0.108% | 5.5e7 | 34 H100-hours |

The scan costs about 4,100 CPU worker-hours for the catalog (160 shards: 4.8 worker-hours, the same
1,005,362 pairs as the measure step). A key-table pre-filter and one hash pass per record made it 6.9
times faster than the measure step.

**Planted copies (gate G2).** `planted.py` plants up to 50 items per eval (41,151 items over 827 evals)
into real record text in 14 forms. Stage 1 finds 96.0% of the planted copies: every planted copy in the
plain, parenthesized, extra-option, no-line-break and JSON-escaped (once and twice) forms, and 99.85% in
Markdown. Almost all misses are ZhoBLiMP: its items are two short Chinese sentences (97% shorter than a
key), so a separator between them (a number, a JSON field name) breaks every window. In the planted
stage-2 check (iteration 13, cutoff 0.9) no planted passage without its question was flagged; real
records are less clean (see the limits). With the template-free keys (`prepare --windows content`, `choose --tie
shared`), stage 1 finds 95.98% (96.02% with the earlier keys on the same item table).

**Validation.** Two sets, weighted to their pools, with an item-clustered stratified bootstrap 5% bound.
Set 19: 1,239 pairs from 114 measure shards that no earlier step used (every flag of iteration 15, a
spread of lower scores). Set 23: 2,679 pairs from 2,000 catalog pair files that hold no labeled pair,
drawn per family (every flag of either model up to 60 per family, 40 per family below the cutoff, 400
low-score pairs). Iteration 15 is the earlier recipe; iteration 16 adds 2,035 judged catalog candidates
of the weak families, sibling negatives and the current rubric's relabels.

| Model, cutoff | Set 19: precision [bound] / recall | Set 23: precision [bound] / recall |
|---|---|---|
| Iteration 15, 0.5 | 0.810 [0.781] / 0.814 | 0.495 [0.459] / 0.761 |
| Iteration 15, 0.95 (lowest passing on set 23) | - | 0.795 [0.722] / 0.513 |
| Iteration 16, 0.3 (lowest passing on set 23) | 0.799 [0.765] / 0.879 | 0.757 [0.716] / 0.815 |
| Iteration 16, 0.5 | 0.862 [0.829] / 0.818 | 0.818 [0.777] / 0.750 |

Set 19 held few flags of the families that fail in the catalog; set 23 chose iteration 16 at 0.3.
Recall counts edited copies: on set 23 at 0.5, iteration 16 removes 97% of the copies that both judges
call exact and 64% of edited ones (changed choices, a reworded question, OCR errors), which carry about
two thirds of the weight; most misses are in BBQ, SQuAD, MMLU, RACE and MMLU-Pro. On
labeled flags of iteration 16 at 0.3 (sets 19 and 23), precision is 0.84-1.00 for MMLU, MMLU-Pro,
Minerva, Hendrycks math, ASDiv, GSM8K, HumanEval, ARC and RTE (54-125 flags each), and low for SQuAD
(0.26), DROP (0.13), QQP (0.08), QNLI (0.21), RACE (0.45), BBH (0.38), AGIEval (0.54), CareQA and
Uncheatable (0.60) and FDA (0.00). Iteration 15 at 0.5 had BBQ at 0.15, BBH at 0.30 and CoLA at 0.

**Whole catalog (iteration 16).** The scan read every one of the 167,533 shards with no failure:
18,974,236,438 records and 9.84e13 characters. Stage 1 gave 21,044,008 records a candidate (0.111%) and
65,011,108 candidate pairs, all scored. The 293,105 records over 5,000,000 characters (6.2e12 characters,
6% of the catalog; stack-v3 5.1e12 in 162,501 records) were read in a second pass, in pieces (`scan
--records oversized`): 1,058,329 pairs, all scored.

Set 20 checks the removals: 1,926 removed records, 20 per family and score band (below 0.5, 0.5 or
more) and 199 drawn uniformly. The judges split on 221 (11.5%), which are left out. The planned estimate
weights each family-band stratum by its removed records; but within a stratum the draw goes round robin
over the family's evals, not in proportion to their removals, so the corrected estimate weights cells of
(eval, band) instead.

| Removed records | Records | Characters | Planned [5% bound] | Corrected [5% bound] | Uniform [5% bound] |
|---|---|---|---|---|---|
| Best score 0.3 or more | 321,339 (0.0017%) | 1.27e10 (0.013%) | 0.677 [0.620] | 0.699 [0.678] | 0.756 [0.705] (176) |
| Best score 0.5 or more | 263,774 (0.0014%) | 9.75e9 (0.0099%) | 0.760 [0.692] | 0.789 [0.758] | 0.841 [0.795] (151) |

0.3 fails the 0.75 floor under both weightings, so the run removes at 0.5. With the second pass, which
removes 928 long records (8.8e10 characters; HumanEval 121, SQuAD 88, GSM8K 53, Uncheatable code 88; precision 0.803 [0.743] on 200 drawn uniformly (set 25)),
the run removes 264,702 records (0.0014%) and 9.76e10 characters (0.093%; `removed-final-05.parquet`).
Stack-v3 loses the most, 0.80% of its characters; no source loses more than 2%. Set 23 and set 20 disagree at 0.3 because removals concentrate in MMLU and ARC, whose
precision in the 0.3-0.5 band is 0.20 and 0.30, while set 23 drew at most 60 flags per family. At 0.5,
precision is 0.84-1.00 for the leaderboard evals (mostly MMLU-Pro), Hendrycks math, AGIEval, Minerva,
ASDiv, HumanEval, PIQA, GSM8K, MRPC, HellaSwag, MedQA, CoLA, BBQ and FDA (12-20 labeled records each),
0.68-0.71 for MMLU (81,303 removals), ARC (44,463), Uncheatable and CareQA, and below 0.6 for 16 families
that hold 22,362 removals, among them SQuAD 0.54, RACE 0.58, SST-2 0.59, DROP 0.35, TurBLiMP 0.35, HLE
0.33, QNLI 0.32, simple co-occurrence bias 0.11, SciCode 0.06 and QQP 0.05.

**Exact rules on the same pairs.** On the real test pairs and on the earlier validation set, the block
rule (`marin.datakit.item_match` in PR #10836: question, then every choice) finds 45% and 68% of the
copies, and the exact question rule 52% and 69%, both at lower precision than the classifier at 0.9.
These numbers use the labels of the earlier rubric, before a question with changed choices counted as
a copy.

## Limits

- **Stage 1 misses about a fifth of clear catalog copies.** On 3,020 records in 2,000 sampled catalog
  shards that the block rule of PR #10836 marks (an eval question with all its choices), stage 1 gives a
  candidate for 79.8%: MMLU 93%, HumanEval 100%, GSM8K 46%; stage 2 removes 98.7% of those, so the run
  removes 78.7% of these records (GSM8K 45%). Planted copies (G2) include the eval's prompt
  text, so they did not show this. A rendered GSM8K question starts with "Question:", and the bare
  question's windows occur in the reference shards (GSM8K copies are common), so the rarest window spans
  the prompt and a record with the bare question lacks it. Two key options fix this (`rarest_key prepare
  --windows content`, `choose --tie shared`); on the same 2,000 shards they raise the share to 99.0% (GSM8K 100%,
  MMLU 98%) at 1.44 times the candidate pairs. `--windows content` leaves out of the candidate windows the prefix
  and suffix that all of an eval's streams share; `--tie shared` breaks ties among equally rare windows by
  the number of other items that hold the window. The catalog run above used the earlier keys; a rescan
  with the new keys (`catalog-v11-full`) needs scoring and a removal check.
- **Candidate load concentrates on shared question stems.** One Nemotron formal-logic shard holds 886,186
  candidate pairs: 260 MMLU formal-logic and MMLU-Pro items keyed on one question stem ("which of the
  following propositions is an immediate (one-step) consequence"), which 48,133 of its records hold.
  Stage 2 flags 373 of its records at 0.3 (11 at 0.5); the 5 checked by hand are other instances of the
  template. With `--tie shared` the shard has 89,043 pairs.
- **1,052 items (0.07%) have no key:** every key stream is shorter than a window once reduced to letters
  and digits (BBH 552: multi-step arithmetic and boolean expressions such as "not ( True ) and ( True )
  is"; ZhoBLiMP 477; AGIEval 15, Gaokao math cloze items that are mostly LaTeX).
- **ZhoBLiMP copies in a numbered or JSON layout are not found** (see the planted copies above).
- **Records with a passage but not its question are removed** for SQuAD, DROP, QQP, QNLI and RACE: at
  0.5 their removals have precision 0.05-0.58 (set 20), and they hold most of the 22,362 removals of the
  16 families below 0.6.
- **Validation is small per family.** Set 20 labels 12-20 removals per family and band, and sets 19 and
  23 give 10 families at least 50 labeled flags; per-family precision is a rough estimate.
- **Flags concentrate in dump records.** A record holding a whole eval can carry thousands of flagged
  pairs, so pair-weighted precision leans on a few records; the removal check measures records.
- **One-sentence evals.** CoLA and SST-2 sentences found without their fixed prompt ("Does this sentence
  make sense?") are labeled not a copy; whether a sentence alone counts is a policy question.
- **One key covers one place in a long item.** A record that copies only part of a long item (an
  Uncheatable document, a long code file) holds the item's key only if the copied part contains it.
- **GPQA choice order is random.** lm-eval shuffles GPQA choices without a seed, so the question text,
  which shows the choices, changes between builds of the item table.
- **Records over 5,000,000 characters are read in pieces** of that size that overlap by 9,000 characters
  (`scan --records all`), so a hit and its excerpt lie in one piece; each shard's stats list their ids. An
  item whose excerpt context is longer than the overlap (a long passage) can get a cut excerpt at a piece
  edge.
- **Cutting an excerpt is linear per record except in one rare case:** a stretch of more than 16,384
  characters with no whitespace, digit, punctuation or non-Latin letter whose body is not plain ASCII
  (fullwidth letters, combining marks, or LaTeX commands with nothing between them). There, H hits cost
  O(H * L * log L) for a stretch of length L (`marin.datakit.decon_keys.Reading`).
