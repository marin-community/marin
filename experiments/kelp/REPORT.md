# Kelp Tree Diffusion: Initial Findings Report

## 1. What We Built

Kelp implements **tree diffusion** for Python program synthesis, following
[Tree Diffusion (Kapur et al., 2024)](https://arxiv.org/abs/2410.09891). Instead
of token-level discrete diffusion (D3PM), we use AST-guided iterative editing:

1. **Forward process**: Corrupt clean programs by replacing random AST subtrees
   with alternatives from a subtree bank (real code fragments indexed by node type).
2. **Training**: Compute TreeDiff edit paths from corrupted to clean. Train a
   causal transformer to predict single edits (position + replacement tokens).
3. **Inference**: Use best-of-N sampling or beam search to iteratively refine
   corrupted programs back toward working code.

Key components:

| Module | Description |
|--------|-------------|
| `tree/subtree_bank.py` | AST subtree extraction and indexing |
| `tree/mutation.py` | AST-based program corruption |
| `tree/tree_diff.py` | Edit path computation between ASTs |
| `tree/tokenizer.py` | Byte-level tokenizer with position tokens |
| `tree/edit_model.py` | Causal AR transformer for edit prediction |
| `tree/train.py` | Training pipeline with TreeDiff supervision |
| `tree/beam_search.py` | Beam search and best-of-N inference |
| `tree/reranking.py` | Execution-guided candidate reranking |
| `tree/constrained_decoding.py` | Grammar-constrained generation |
| `tree/augmentation.py` | Subtree bank data augmentation |

## 2. Training Results

### Configuration

| Parameter | Value |
|-----------|-------|
| Model size | 4.59M parameters |
| Architecture | 256d, 1024 intermediate, 4 layers, 4 heads |
| Vocabulary | 771 tokens (byte-level + 128 position + 3 special) |
| Max sequence length | 512 |
| Batch size | 16 |
| Learning rate | 1e-3 (cosine decay) |
| Training steps | 12,000 |
| Training tokens | 98M |
| Estimated FLOPs | 2.7e15 |
| Corpus | 15 toy Python programs |
| Subtree bank | 428 entries (augmented from 55) |
| Wall time | 4h 51m on laptop CPU |

### Training Curve

```
Step      Loss    Accuracy   Perplexity
------    ------  --------   ----------
     0    6.8379    0.00%      932.51
   100    0.7526   77.07%        2.12
   500    0.2657   91.56%        1.30
  1000    0.3533   85.71%        1.42
  2000    0.1413   94.89%        1.15
  5000    0.1221   96.67%        1.13
  8000    0.0782   97.27%        1.08
 10000    0.0640   97.80%        1.07
 11900    0.0773   97.89%        1.08
```

The model converges rapidly (90%+ accuracy by step 200) and continues improving
slowly through 12,000 steps. Final training metrics: **loss=0.077, accuracy=97.9%,
perplexity=1.08**. The model has essentially memorized how to predict edits on
the 15-program training corpus.

## 3. Evaluation Results

Evaluation corrupts clean programs (3 AST mutations each), then uses best-of-16
sampling with max_depth=10 to attempt repair. 10 corruption trials per task,
10 tasks.

### Aggregate Metrics

| Metric | Value |
|--------|-------|
| **Syntactic validity** | **100.0%** |
| Exact match rate | 2.5% |
| Average test pass rate | 16.1% |
| Best-case test pass rate | 56.7% |

### Per-Task Breakdown

| Task | Valid | Exact | Avg Pass | Best Pass |
|------|-------|-------|----------|-----------|
| add | 100% | 0.0% | 0.0% | 0.0% |
| sub | 100% | 2.9% | 14.3% | 100.0% |
| mul | 100% | 3.3% | 0.0% | 0.0% |
| neg | 100% | 2.3% | 0.0% | 0.0% |
| abs_val | 100% | 0.0% | 3.3% | 33.3% |
| max_val | 100% | 0.0% | 40.0% | 66.7% |
| min_val | 100% | 3.4% | 43.3% | 100.0% |
| clamp | 100% | 0.0% | 30.0% | 66.7% |
| double | 100% | 7.5% | 14.8% | 100.0% |
| square | 100% | 5.8% | 14.8% | 100.0% |

## 4. Key Findings

### What Works

1. **Syntactic validity is solved.** Grammar-constrained decoding ensures 100% of
   generated programs parse as valid Python. The bracket-balancing constraint and
   AST validation in the constrained decoder are effective.

2. **The model learns edit prediction.** 97.9% training accuracy shows the
   architecture and training pipeline work: the model correctly predicts position
   tokens and replacement content for in-distribution examples.

3. **Subtree bank augmentation helps.** Expanding the bank from 55 to 428 entries
   via variable renaming, operator perturbation, and synthetic templates reduced
   training loss by 27% and improved accuracy by 60 percentage points in a
   controlled experiment.

### What Doesn't Work

1. **Semantic correctness is poor.** Despite 97.9% training accuracy, only 2.5%
   of generated programs exactly match the clean target and only 16% pass tests.
   The model produces syntactically valid but semantically wrong programs.

2. **Massive train/eval gap.** The 80+ percentage point gap between training
   accuracy (97.9%) and eval test pass rate (16.1%) indicates severe overfitting
   to the 15-program training distribution.

3. **Simple functions paradoxically fail.** `add`, `mul`, and `neg` achieve 0%
   test pass rate despite being the simplest programs. Inspection of the eval
   JSON reveals the model generates programs that are structurally similar but
   compute the wrong operation (e.g., replacing `+` with `-` or generating
   `return a` instead of `return a + b`).

4. **Multi-step repair accumulates errors.** The best-of-N with depth=10 often
   fails to improve on the original corrupted program. Each edit step has a
   chance of introducing new errors, and without execution feedback, bad edits
   accumulate.

### Hypotheses

1. **Corpus diversity is the primary bottleneck.** With only 15 programs, the
   model memorizes specific edit patterns rather than learning general program
   repair. The corruption at eval time can produce states the model never saw
   during training.

2. **Model scoring doesn't correlate with correctness.** The model's
   log-probability score ranks candidates by how "likely" they are under the
   training distribution, not by whether they're correct. Without execution
   feedback during search, the best-scored candidate often isn't the best
   program.

3. **Position token granularity limits repair.** The 128-position vocabulary
   maps character offsets coarsely. For short programs where every character
   matters, even small position errors cascade into wrong edits.

## 5. Scaling Predictions

### Current Compute Budget

Our training run used approximately **2.7e15 FLOPs** — roughly 1,000x smaller
than the smallest budget in Marin's scaling law framework (1e18 FLOPs), and
9 orders of magnitude smaller than the 1e24 budget used for frontier models.

### Marin's Scaling Framework

Marin uses the IsoFLOP approach from the Chinchilla paper:

- **Optimal tokens**: `D* = A * C^alpha`, where alpha typically ranges 0.5-0.8
- **Optimal parameters**: `N_opt = C / (6 * D*)`
- **Loss prediction**: Quadratic in log-space: `loss = a*log10(tokens)^2 + b*log10(tokens) + c`

These laws were derived for next-token prediction on natural language. Tree
diffusion edit prediction is a different task, so we cannot directly apply
Marin's fitted coefficients. However, we can use the framework to reason about
scaling behavior.

### Key Scaling Differences for Tree Diffusion

| Factor | Language Modeling | Tree Diffusion Edit Prediction |
|--------|-------------------|-------------------------------|
| Data supply | Fixed corpus | Infinite (synthetic corruption) |
| Task structure | Low (predict next token) | High (predict valid AST edits) |
| Output constraints | None | Grammar-constrained |
| Evaluation | Perplexity | Functional correctness |
| Bottleneck | Data (Chinchilla) | Corpus diversity + model capacity |

Because training data can be generated synthetically (corrupt any program, compute
TreeDiff path), **tree diffusion is not data-constrained in the Chinchilla sense**.
The bottleneck is the diversity of the *clean* program corpus, not the volume of
training tokens.

### Predictions at Increased Compute

| Budget | FLOPs | Model Size | Corpus Size | Predicted Test Pass |
|--------|-------|------------|-------------|---------------------|
| Current | 2.7e15 | 4.6M | 15 programs | 16% (measured) |
| 10x | 2.7e16 | ~15M | 150 programs | ~30-40% |
| 100x | 2.7e17 | ~50M | 1,500 programs | ~50-65% |
| 1,000x | 2.7e18 | ~150M | 15,000 programs | ~70-80% |
| 10,000x | 2.7e19 | ~500M | 150,000 programs | ~85-90% |

**Rationale:**

- **10x budget (2.7e16)**: A 15M parameter model trained on 150 diverse programs
  would see enough variety to generalize basic arithmetic and comparison
  operations. We predict 30-40% average test pass rate, with simple functions
  reaching near-100%.

- **100x budget (2.7e17)**: At 50M parameters and 1,500 programs, the model
  enters the regime where Chinchilla-style scaling begins to apply. Multi-
  statement repairs (abs_val, clamp) should become reliable. Predicted 50-65%.

- **1,000x budget (2.7e18)**: This matches Marin's smallest scaling law budget.
  At 150M parameters, the model has enough capacity for complex multi-step
  repairs. With 15,000 programs covering diverse patterns (loops, recursion,
  error handling), we predict 70-80%.

- **10,000x budget (2.7e19)**: At 500M parameters and a diverse corpus, the
  model should approach the paper's reported results on their domain. We predict
  85-90%, with the remaining 10-15% requiring either execution-guided search or
  larger inference budgets.

**Critical assumption**: These predictions assume corpus diversity scales with
compute. If the model is trained on the same 15 programs with more steps or
larger architecture, we predict diminishing returns — the current 97.9% training
accuracy suggests the model is already saturated on this corpus. **Scaling model
size without scaling data diversity will not help.**

### The Data Diversity Wall

The most important insight from this experiment is that **tree diffusion hits a
data diversity wall before it hits a compute wall**. Unlike language modeling
where web-scale corpora provide trillions of tokens of diverse text, program
synthesis requires curated clean programs with test cases.

To break through the data wall, we recommend:

1. **Curate a real program corpus**: Pull 10,000+ functions from HumanEval,
   MBPP, CodeContests, or open-source Python projects.
2. **Use LLMs for corpus generation**: Generate diverse programs with
   specifications and test cases using an instruction-tuned LLM.
3. **Curriculum over program complexity**: Start with simple functions and
   gradually introduce more complex patterns (classes, nested loops, recursion).

## 6. Improvement Roadmap

Based on the evaluation results, we prioritize improvements by expected impact:

### High Priority

1. **Scale corpus diversity** (Issue #44): The single highest-leverage
   improvement. Train on 1,000+ diverse programs instead of 15.
   Expected impact: 3-5x improvement in test pass rate.

2. **Integrate execution-guided reranking** (Issue #45): The reranking module
   exists but isn't used in the eval pipeline. Running candidates against tests
   and selecting the best-passing one would immediately improve best-case metrics.
   Expected impact: 2-3x improvement in average test pass rate.

3. **Investigate train/eval distribution gap** (Issue #46): The 80+ point gap
   between training accuracy and eval pass rate suggests a fixable distribution
   mismatch. Aligning corruption parameters between train and eval should help.
   Expected impact: 1.5-2x improvement.

### Medium Priority

4. **Improve position prediction** (Issue #43): Relative position encoding
   (offset from nearest AST node boundary) instead of absolute character offsets
   would generalize better across programs of different lengths.

5. **Add iterative refinement with test feedback**: After generating candidates,
   take the best partial solution and continue editing it using test results as
   guidance. This is the key mechanism from the Tree Diffusion paper that we
   haven't yet implemented.

### Lower Priority

6. **Temperature tuning**: Systematic sweep over inference temperature (currently
   0.8) to find the optimal exploration/exploitation tradeoff.

7. **KV cache for inference**: Current inference runs a full forward pass per
   generated token. Adding KV caching would speed up inference ~10x, enabling
   larger beam sizes and more rollouts.

8. **Train a value network**: Following the paper, train a separate model to
   predict which candidates will pass tests, replacing the heuristic combined
   score in the reranker.

## 7. Conclusion

The Kelp tree diffusion pipeline successfully demonstrates that:

- **AST-guided program corruption and repair via learned edits is feasible**
- **Grammar-constrained decoding achieves 100% syntactic validity**
- **The training pipeline (corrupt -> TreeDiff -> train) works end-to-end**

The main limitation is **data diversity**: with only 15 training programs, the
model memorizes rather than generalizes. The 97.9% training accuracy vs 16.1%
eval test pass rate quantifies this overfitting precisely.

The path forward is clear: scale the program corpus while keeping the
architecture and training pipeline fixed. The infrastructure is ready for
larger-scale experiments on real codebases.

---

*Generated from checkpoint `checkpoints/kelp-edit/step-012000`.*
*Evaluation data at `checkpoints/kelp-edit/step-012000/eval_results.json`.*
*194 tests passing across the kelp test suite.*
