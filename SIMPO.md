# SimPO (Simple Preference Optimization) Notes

## Overview

The [SimPO repo](https://github.com/princeton-nlp/SimPO) implements **Simple Preference Optimization (SimPO)**, a reference-free alternative to DPO. The repo does **not** contain vanilla DPO training code—DPO baselines were run using the [alignment-handbook](https://github.com/huggingface/alignment-handbook) repository.

This document is meant to be a *practical* summary of what SimPO is doing **and how to implement it correctly** (matching the Princeton-NLP reference implementation in `/Users/ahmed/code/SimPO/scripts/simpo_trainer.py`).

## Dataset

- **Name**: `HuggingFaceH4/ultrafeedback_binarized`
- **Splits**: `train_prefs`, `test_prefs`
- Same dataset used in Marin's `exp2101_dpo_ultrafeedback.py`

## DPO Baseline Hyperparameters (from SimPO paper)

These were run using alignment-handbook, not code in the SimPO repo:

| Setting | β (beta) | Learning Rate |
|---------|----------|---------------|
| Mistral-Base | 0.01 | 5e-7 |
| Mistral-Instruct | 0.01 | 5e-7 |
| **Llama3-Base** | **0.01** | **5e-7** |
| Llama3-Instruct | 0.01 | 7e-7 |
| Llama3-Instruct v0.2 | 0.01 | 3e-7 |
| Gemma | 0.01 | 5e-7 |

## SimPO Hyperparameters

| Setting | β | γ/β | Learning Rate |
|---------|---|-----|---------------|
| Mistral-Base | 2.0 | 0.8 | 3e-7 |
| Mistral-Instruct | 2.5 | 0.1 | 5e-7 |
| **Llama3-Base** | **2.0** | **0.5** | **6e-7** |
| Llama3-Instruct | 2.5 | 0.55 | 1e-6 |
| Gemma | 10 | 0.5 | 8e-7 |

## Shared Training Settings (Llama 3 8B Base)

From `training_configs/llama-3-8b-base-simpo.yaml`:

| Parameter | Value |
|-----------|-------|
| Base model | `princeton-nlp/Llama-3-Base-8B-SFT` (SFT-tuned first) |
| Epochs | 1 |
| Per-device batch size | 2 |
| Gradient accumulation | 16 |
| Effective batch size | 128 (with 4 GPUs) |
| Max sequence length | 2048 |
| Max prompt length | 1800 |
| Optimizer | AdamW |
| LR scheduler | Cosine |
| Warmup ratio | 0.1 |
| Precision | bf16 |

## Comparison: SimPO DPO vs Marin DPO

| Parameter | SimPO's DPO Baseline | Marin Config |
|-----------|---------------------|--------------|
| **beta** | 0.01 | 0.1 |
| learning_rate | 5e-7 | 5e-7 |
| batch_size | 128 | 64 |
| max_seq_len | 2048 | 4096 |
| training | 1 epoch | 5000 steps |
| base_model | SFT-tuned Llama 3 | Llama 3.1 Instruct |

## Key Observations

1. **Beta difference**: SimPO uses β=0.01 for DPO, Marin uses β=0.1 (10x larger)
2. **Base model**: SimPO starts from an SFT checkpoint (`princeton-nlp/Llama-3-Base-8B-SFT`), not raw base or instruct
3. **No vanilla DPO code**: SimPO repo only has SimPO trainer; DPO baselines used alignment-handbook
4. **SimPO vs DPO**: SimPO removes the reference model and uses much larger beta (2.0 vs 0.01)

## SimPO Loss Function

### The actual objective (includes length normalization)

The SimPO paper defines a *length-normalized* pairwise objective (Eq. 6):

\[
\mathcal{L}_{\text{SimPO}}(\pi_\theta) =
-\mathbb{E}_{(x, y_w, y_l)\sim\mathcal{D}}\left[
\log \sigma\left(
\frac{\beta}{|y_w|}\log \pi_\theta(y_w\mid x)
-\frac{\beta}{|y_l|}\log \pi_\theta(y_l\mid x)
-\gamma
\right)\right]
\]

Key detail: the \(\tfrac{1}{|y|}\) terms mean **average log-probability per completion token**, not the sum over tokens.

### How the Princeton-NLP code matches this

In `/Users/ahmed/code/SimPO/scripts/simpo_trainer.py`, the trainer computes:

- `policy_chosen_logps`: **average** log-prob per (non-masked) token for the chosen completion.
- `policy_rejected_logps`: same for rejected.

Then it applies the sigmoid loss with a margin using `gamma_beta_ratio := gamma / beta`:

```python
# simpo_trainer.py: concatenated_forward -> get_batch_logps(..., average_log_prob=True)
avg_logp_chosen = policy_chosen_logps      # already length-normalized
avg_logp_rejected = policy_rejected_logps  # already length-normalized

# simpo_trainer.py: simpo_loss()
pi_logratios = avg_logp_chosen - avg_logp_rejected
logits = pi_logratios - gamma_beta_ratio  # gamma_beta_ratio = gamma / beta

losses = -F.logsigmoid(beta * logits)
```

Note: in this context `logits` are the *pairwise preference logits* (a scalar per example), not the model's `[B, T, V]` token logits.

This is equivalent to the paper form because:

\[
\beta(\text{avg\_logp}_w - \text{avg\_logp}_l - \gamma/\beta)
=
\beta\cdot \text{avg\_logp}_w - \beta\cdot \text{avg\_logp}_l - \gamma
\]

### Correct way to compute the length-normalized log-probabilities

The key implementation detail is **how** you compute \(\frac{1}{|y|}\log \pi_\theta(y\mid x)\) in practice for a causal LM:

1. Tokenize `prompt` and `completion` and build `input_ids = prompt_ids + completion_ids`.
2. Build `labels = input_ids.copy()`, then set the prompt portion (and any padding) to `label_pad_token_id` (default `-100`) so those tokens do not contribute.
3. Run the model forward to get `logits` (shape `[B, T, V]`).
4. Compute per-token log-probs for the label tokens and **average over non-masked tokens**.

This matches the reference implementation (see `get_batch_logps()` in `simpo_trainer.py`):

```python
def average_log_prob_of_labeled_tokens(logits, labels, label_pad_token_id=-100):
    # For a causal LM, predict token t using logits at t-1.
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    loss_mask = labels != label_pad_token_id
    labels[~loss_mask] = 0  # dummy token id; masked out later

    per_token_logps = logits.log_softmax(-1).gather(2, labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
```

If you accidentally use the **sum** instead of the **average**, you are no longer implementing Eq. (6), and you will re-introduce a strong length bias (the thing SimPO is explicitly trying to reduce).

### Hyperparameters: what beta and gamma mean in code

- `beta` (\(\beta\)): scales the preference separation. In the SimPO repo, it is typically **much larger** than DPO (e.g. 2.0, 2.5, 10).
- `gamma_beta_ratio` (\(\gamma/\beta\)): the code parameter. The paper uses \(\gamma\) directly; the repo uses the ratio for easier tuning across different \(\beta\).
  - Convert between them via: `gamma = beta * gamma_beta_ratio`.

### Variants in the repo (important if you want to match it exactly)

In `/Users/ahmed/code/SimPO/scripts/simpo_trainer.py`:

- `loss_type="sigmoid"` (default): standard logistic loss (optionally with `label_smoothing`).
- `loss_type="hinge"`: hinge loss `relu(1 - beta * logits)` (label smoothing ignored).
- Optional `sft_weight > 0`: adds a standard cross-entropy SFT loss on the **chosen** sequence.
  - Note: the implementation uses `nn.CrossEntropyLoss()` with PyTorch's default `ignore_index=-100`, so keep `label_pad_token_id=-100` if you enable SFT regularization.

### Tokenization / masking pitfalls (easy ways to get SimPO subtly wrong)

- The SimPO code treats \(|y|\) as **the number of completion tokens that are actually trained on**:
  - It masks prompt tokens in `labels` to `-100` so they don't count in the numerator or denominator.
  - It appends an EOS token to completions (so EOS is included in the averaged log-prob).
- Llama-family tokenizers can violate the naive assumption that `tokenize(prompt + answer) == tokenize(prompt) + tokenize(answer)` due to token merging.
  - The reference implementation has a `build_tokenized_answer()` helper to ensure prompt/answer boundaries are handled consistently.

## References

- Paper: [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
- Code: https://github.com/princeton-nlp/SimPO
- Models: https://huggingface.co/collections/princeton-nlp/simpo-66500741a5a066eb7d445889
