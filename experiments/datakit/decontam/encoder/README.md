# Two-stage decontamination with an encoder classifier (proof of concept)

This directory holds a proof of concept for removing eval items from pretraining data. It works in
two stages and uses one rule for all evals, with no values set per eval:

1. **Stage 1, the wide net.** Each eval item gets one key: the rarest 20-character window of its
   text stream. The rarity is the count of records that hold the window in a reference sample. A
   record that holds an item's key is a candidate for that item.
2. **Stage 2, the classifier.** A fine-tuned multilingual encoder reads the pair
   `(left, right)`, where `left` is the item's question and answer and `right` is the record text
   around the key. It gives two probabilities: that `right` holds the question
   (`label_q`), and that `right` holds the question with its answer (`label_qa`).

The text stream of a text is its lowercase letters and digits, after NFKC folding and with LaTeX
commands removed (`text_keys.py`). Spacing, punctuation, Markdown and markup do not change it.

## Pipeline

| Step | Module | Output |
|---|---|---|
| Item table: one row per required eval record (question, choices, answer, passage) | `items.py` | `items/<eval>.parquet` |
| Key choice: count candidate windows in reference shards, keep the rarest | `rarest_key.py prepare, count, choose` | `keys.parquet` |
| Candidate volume and a candidate sample on measure shards | `rarest_key.py measure` | `measure/*.json`, `sample/*.parquet` |
| Candidate pairs: the record text around each sampled key hit | `pairs.py items, extract` | `pairs/*.parquet` |
| Synthetic pairs with exact labels | `synthetic.py generic, context` | one parquet |
| Judge label sets from real candidates, and training rows from two judges | `labels.py sample, merge`, `RUBRIC.md` | batches, one parquet |
| Fine-tuning (one H100) | `train.py` | `model/`, `results.json`, `test_scores.parquet` |
| Scoring pairs with a saved model | `score.py` | one parquet |
| Stage-1 recall on judged copies | `key_recall.py` | `key_recall.json` |

`sources.json` lists the 160 normalized sources of the sample. `rarest_key.py` draws one measure
shard per source (seed 7) and two other reference shards per source (seed 11), in list order.

## Commands

All steps run on Iris (`cw-us-east-02a`), at interactive priority. `$T` is a scratch prefix, for
example `s3://marin-us-east-08a/tmp/ttl=7d/$USER`. GPU jobs need `--disk 100GB`: the default 5 GB
disk is too small for the model and the pairs.

```bash
# Item table (4 parts, about 1 hour each).
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
  --model $T/train/<tag>/model --heads label_q label_qa --hint --pairs $T/pairs --output $T/pairs/scored.parquet

# Training (about 35 minutes on one H100 for 2 epochs of about 150k pairs).
iris --cluster=cw-us-east-02a job run --extra=gpu --gpu H100x1 --enable-extra-resources --memory 64GB --cpu 8 --disk 100GB \
  -e TOKENIZERS_PARALLELISM false -e OMP_NUM_THREADS 8 --job-name decon-train -- \
  python -m experiments.datakit.decontam.encoder.train --tag <tag> --model jhu-clsp/mmBERT-base --lr 3e-5 --heads 2 --hint \
  --data $T/data --synthetic synthetic_pairs_v3.parquet \
  --extra $T/data/set11_full.parquet --extra $T/data/silver_negatives_v1.parquet --extra $T/data/synthetic_ctx_v4.parquet \
  --output $T/train
```

Label sets run outside Iris, because the judges are coding agents. `labels.py sample` writes
batches of tasks; each judge reads `RUBRIC.md` and writes `<set>/<judge>/batch_*.jsonl`; and
`labels.py merge` keeps the pairs that both judges agree on.

## Model

- Base model: `jhu-clsp/mmBERT-base` (ModernBERT architecture, multilingual), with two sigmoid outputs
  and a binary cross-entropy loss. Inputs are cut to 1,024 tokens; `left` is cut to 1,500 characters
  and `right` to 6,000.
- `--hint` adds the words of `left` that `right` lacks, as `\n[missing] word ...` (up to 32 words).
  On real candidates it raised the test AUC from 0.931 to 0.947.
- The threshold is the lowest score that keeps a precision of at least 0.95 on a dev split. The dev
  split is one eighth of the train items of the judge-labeled pairs; silver and synthetic pairs
  never set it.

## Measured results

RESULTS

## Limits

LIMITS
