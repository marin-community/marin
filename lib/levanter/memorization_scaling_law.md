# Memorization Scaling Law

## Motivation

Understanding how neural networks memorize training data is crucial for predicting model behavior and data contamination effects. This experiment investigates how memorization scales with two key factors:

1. **Number of epochs**: How many times the model sees the same data
2. **Seed set size**: The size of the training data pool (in tokens)

By varying both dimensions independently, we can derive a scaling law that predicts memorization behavior across different training regimes.

## Experimental Setup

### Model Architecture
- **Model**: 150M parameter language model (comma_150m)
- **Evaluation metric**: P(z) - the probability the model assigns to a specific set of held-out documents from the training distribution

### Experimental Design
We trained the same model architecture on different combinations of:

- **Seed set sizes**: 1M, 10M, 100M tokens
- **Epoch counts**: 1, 2, 5, 10, 20, 50, 75, 100, 150, 200, 375, 500, 562, 750, 1000, 1500, 3000 epochs

For each (seed_set, epochs) pair, we measured the final `mean_pz` value on a fixed evaluation set.

### Data Source
All runs used the Wikimedia subset of Common Pile for training data. The evaluation set (`pz_eval/common_pile/wikimedia/mean_pz`) measures memorization of specific documents from this distribution.

## Proposed Scaling Law

We hypothesize that memorization follows a power-law relationship:

```
log(P(z)) = c₁·log(epochs) + c₂·log(seed_set_tokens) + c₃
```

Where:
- **c₁**: Scaling coefficient for epochs (expected to be positive - more epochs → more memorization)
- **c₂**: Scaling coefficient for seed set size (expected to be negative - larger seed set → less memorization per document)
- **c₃**: Intercept term capturing baseline memorization

### Intuition

- **Epochs scaling (c₁)**: Each additional pass over the data increases exposure to specific sequences, enhancing memorization
- **Seed set scaling (c₂)**: Larger seed sets dilute the model's capacity to memorize individual documents, as attention is distributed across more unique data
- **Interaction**: The law assumes these factors combine multiplicatively (additive in log-space)

## Analysis Plan

1. Load experimental results from WandB runs
2. Fit the scaling law using linear regression in log-space
3. Evaluate fit quality (R² score)
4. Visualize:
   - Actual vs predicted P(z) values
   - Residuals analysis
   - Scaling curves for each seed set size
5. Interpret coefficients to understand memorization dynamics
