# Fast-transformer document-quality classifier

A pooled transformer that scores a document's value as LLM-pretraining data,
trained against the Sonnet 4.6 oracle ([`../v0/rubric.py`](../v0/rubric.py)). The
goal was a higher-fidelity replacement for the fasttext quality filter under a
<1M FLOPs/token budget.

**Result.** The pooled model reaches **AUC 0.875 / Spearman ρ 0.703 at 0.41M
FLOPs/token**, vs the fasttext baseline's 0.846 / 0.641. That meets the goal. The
score has not moved past ~0.87 across capacity, context, and pretraining sweeps;
we think that is a label-quality ceiling of the 5.6k-doc oracle set rather than
an architecture limit (see "Pretraining experiments" below).

## Architecture

`embed → pool over 64-token windows → input proj + positions → N transformer
layers over the super-tokens → pool → scalar quality head`. Pooling at the
window boundary amortizes the transformer cost by ~64×, which is what keeps it
under the FLOPs budget while still running real self-attention. The winning
config is `meanmaxmin` pooling, `pool_window=64`, `embed_dim=256`,
`hidden_dim=512`, `num_layers=4`, `max_tokens=1024`.

## Files

Core library:

- [`data.py`](data.py) — tokenize the oracle-scored parquets and pack dense padded arrays + a compact vocab.
- [`model.py`](model.py) — the pooled `FastTransformer` regressor (the deliverable).
- [`train.py`](train.py) — `fit` / `train_regressor` (MSE-on-sigmoid, early stopping, data-parallel across chips) and the holdout metrics.

Deliverable drivers:

- [`sweep.py`](sweep.py) — architecture grid that selected the winning config.
- [`eval_best.py`](eval_best.py) — train the winner and report holdout metrics, a val-calibrated operating point, and a per-source breakdown.

Pretraining experiments ([`pretrain/`](pretrain/)) — attempts to beat the plateau with more (cheaper) labels:

- [`pretrain/transfer.py`](pretrain/transfer.py) — shared scratch → pretrain → finetune comparison used by the supervised-pretraining drivers.
- [`pretrain/source_prior.py`](pretrain/source_prior.py) — pretrain on each doc's source-mean oracle score (source-of-origin prior).
- [`pretrain/nemotron_bucket.py`](pretrain/nemotron_bucket.py) — pretrain on Nemotron-CC quality buckets.
- [`pretrain/ntp.py`](pretrain/ntp.py) + [`pretrain/encoder.py`](pretrain/encoder.py) — token-level encoder pretrained with next-token prediction, then fine-tuned with a pooled head.
- [`pretrain/nemotron_sample.py`](pretrain/nemotron_sample.py) — sample the Nemotron-bucket pretraining corpus.

## Results

AUC and Spearman ρ of predicted quality vs the oracle on the 961-doc holdout.
FLOPs/token is forward inference cost; the 1M budget is the design constraint.

| variant | AUC | ρ | FLOPs/tok | notes |
|---|---|---|---|---|
| fasttext baseline | 0.846 | 0.641 | — | bag-of-bigrams |
| source-of-origin (label only, no text) | 0.852 | — | 0 | per-source mean oracle score; ceiling of the "which source" signal |
| **pooled fast-transformer, from scratch** | **0.875** | **0.703** | **0.41M** | the deliverable |
| pooled + source-prior pretrain → finetune | 0.858 | 0.691 | 0.41M | below scratch; representation collapsed to source identity |
| pooled + nemotron-bucket pretrain → finetune | 0.814 | 0.599 | 0.41M | below scratch; Nvidia buckets misaligned with our rubric |
| token encoder + NTP pretrain → finetune | 0.858–0.877 | 0.69–0.72 | 14.7M | over budget; the 0.877 run did not reproduce at larger scale |

## Pretraining experiments: why they did not help

Every free label source available is at or below the from-scratch pooled model:
source-of-origin (0.852), our own fasttext (0.846), the dolma3 classifier. The
two supervised-pretraining attempts (`source_prior`, `nemotron_bucket`) both
landed below the from-scratch control — the pretrained representation collapses
toward the weak signal and is a worse init than random for the within-source
per-doc task. NTP helped the weaker token-level encoder but stays over budget and
under the pooled plateau.

The remaining lever above the current model is more *real* oracle labels (the
[`../v0/`](../v0) scoring pipeline at larger scale). That is the only signal
richer than what the model already learns from the 5.6k gold set.
