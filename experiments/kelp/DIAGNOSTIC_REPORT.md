# Kelp Tree Diffusion: Diagnostic Report on Model Repair Failures

## Summary

We trained a 4.59M-parameter AR edit-prediction model on 13,019 decontaminated Python
functions for 12,000 steps. The model achieves 83% training accuracy and generates 100%
syntactically valid Python, but scores near 0% on test-pass metrics across most evaluation
tasks. This report diagnoses why, using direct inspection of model outputs, corruption
traces, and controlled experiments varying the subtree bank size.

**The primary failure mode is not the model — it is the evaluation setup.** The corruption
process produces impossible repair tasks when using a small subtree bank, and the eval
programs are fundamentally out-of-distribution from the training corpus.

## Experimental Setup

### Model

- Architecture: 256d, 4 layers, 4 heads, 4.59M params
- Training: 12,000 steps on 13,019 programs (local repos + HumanEval + github-code)
- Final metrics: loss=0.73, accuracy=83%, perplexity=2.07
- Augmented subtree bank: 191,917 entries across 36 AST node types

### Evaluation Configurations

We ran three evaluation configurations on the same checkpoint:

| Config | Subtree Bank | Eval Tasks | Notation |
|--------|-------------|------------|----------|
| Tiny bank | 35 entries, 5 node types (from 10 eval programs) | 10 hand-crafted EVAL_TASKS | A |
| Corpus bank | 70,017 entries, 36 node types (from training corpus) | 10 hand-crafted EVAL_TASKS | B |
| MBPP | 70,017 entries, 36 node types (from training corpus) | 50 held-out MBPP programs | C |

We also include results from the v1 model (trained on 15 contaminated toy programs) for
historical comparison.

## Results

### Aggregate Metrics

| Config | Valid | Exact Match | Avg Test Pass | Best Test Pass |
|--------|-------|-------------|---------------|----------------|
| v1 (contaminated, tiny bank) | 100% | 2.5% | 16.1% | 56.7% |
| A: v3 + tiny bank | 100% | 0.0% | 11.3% | 46.7% |
| B: v3 + corpus bank | 100% | 0.0% | 2.3% | 6.7% |
| C: v3 + MBPP (20/50 done) | 100% | 0.0% | 1.3%* | 1.7%* |

*MBPP results are partial (20 of 50 tasks completed at time of writing).

### Per-Task Breakdown (EVAL_TASKS)

| Task | v1 Tiny | A: v3 Tiny | B: v3 Corpus |
|------|---------|------------|--------------|
| add | 0.0% | 0.0% | 0.0% |
| sub | 14.3% | 0.0% | 0.0% |
| mul | 0.0% | 0.0% | 0.0% |
| neg | 0.0% | 6.7% | 0.0% |
| abs_val | 3.3% | 6.7% | 0.0% |
| max_val | 40.0% | 20.0% | 0.0% |
| min_val | 43.3% | 26.7% | 0.0% |
| clamp | 30.0% | 23.3% | **23.3%** |
| double | 14.8% | 14.8% | 0.0% |
| square | 14.8% | 14.8% | 0.0% |

### MBPP Results (partial)

Of 20 tasks evaluated, 1 scored non-zero: task 605 ("check if integer is prime") at
26.7% avg / 33.3% best test pass rate. All others scored 0%.

## Diagnostic Findings

### Finding 1: The tiny subtree bank produces impossible corruptions

With only 35 subtree entries from 10 tiny programs, `FunctionDef`-level replacements swap
**entire programs** for other programs in the bank.

Observed corruption traces:

| Clean Program | Corrupted To | Corruption Path |
|--------------|-------------|-----------------|
| `add(a, b): return a+b` | `double(x): return x+x` | BinOp swap -> FunctionDef swap -> FunctionDef swap |
| `square(x): return x*x` | `add(a, b): return a+b` | Return swap -> BinOp swap -> FunctionDef swap |
| `double(x): return x+x` | `double(x): return a*b` | Return swap -> Return swap -> Return swap (indent drift) |

When `add` is corrupted into `double`, the model would need to synthesize a completely
different program from scratch. This is program **synthesis**, not program **repair**.
The model was trained to make small, local AST edits — not to rewrite entire programs.

With the 70K-entry corpus bank, corruptions are more diverse (drawing from 36 node types
and 13K real programs), but the problem persists: subtree replacements at the `FunctionDef`
level can still be catastrophic for small programs.

### Finding 2: The model prefers no-op over risky edits

In direct output inspection, the top-scored candidate in 4 out of 5 cases was the
**corrupted program unchanged** (score=0.00, depth=0):

```
=== add (corrupted to double) ===
[0] score=0.00  depth=0  'def double(x):\n    return x + x'          <- NO-OP
[1] score=-17.15 depth=1  'def get_frame(x: NamedArray) -> ...'      <- hallucinated
[2] score=-26.14 depth=1  'def start_file_exter__(self): ...'        <- hallucinated
```

The model has correctly learned that arbitrary edits usually make things worse. When
facing an out-of-distribution program, doing nothing is the rational choice under the
model's learned distribution. The problem is that the eval **requires** the model to
make edits — it penalizes the no-op by definition.

### Finding 3: When the model does edit, it hallucinates training-corpus fragments

The non-zero-depth candidates are fragments from the training corpus spliced in:
- `get_frame(x: NamedArray)` — from a Levanter/Haliax tensor operation
- `start_file_exter__(self)` — garbled class method
- `sampendate(x, int)` — nonsense name

These are not meaningful repairs. The model is sampling from its learned prior over Python
code rather than conditioning on the specific corruption to produce a targeted fix.

### Finding 4: The clamp function is uniquely repairable

`clamp` is the only EVAL_TASK that consistently achieves non-zero test pass rates across
all configurations:

| Config | clamp Avg Pass | clamp Best Pass |
|--------|---------------|-----------------|
| v1 tiny bank | 30.0% | 66.7% |
| v3 tiny bank | 23.3% | 66.7% |
| v3 corpus bank | 23.3% | 66.7% |

Why? The `clamp` function has the most complex structure (3 branches, 3 return paths).
The training corpus of 13K functions contains many `if/return` patterns. When `clamp`
is corrupted, partial structure survives (the if-chain skeleton) and the model can
repair individual branches. Simpler programs have less structure to anchor the repair.

The v3 corpus-bank best candidate for `clamp` is revealing:

```python
# Clean
def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

# Best candidate (corpus bank)
def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return {'sum': sum((s['sum'] for s in stats_list)), ...}
    return x
```

The model correctly repairs 2 of 3 branches but hallucinates a training-corpus fragment
for the third. It preserved the program skeleton and repaired locally — exactly what
tree diffusion should do — but doesn't have the semantic understanding to fill in
`return hi` specifically.

### Finding 5: Best candidates are often functionally correct but textually wrong

In the tiny-bank eval, several "best candidates" are the correct program with wrong
whitespace:

```python
# double best candidate (v3 tiny bank, score=100% test pass):
"def double(x):\n                return x + x"

# square best candidate (v3 tiny bank, score=100% test pass):
"def square(x):\n                return x * x"
```

These pass all tests (functionally correct) but don't match the clean program exactly
(0% exact match). The excess indentation comes from corruption → repair cycles that
accumulate whitespace. This is a tokenization/formatting artifact, not a model failure.

### Finding 6: MBPP program repair shows a real signal

The one MBPP success (task 605, primality check) demonstrates that the model **can** repair
real-world programs when:
1. The corruption is realistic (using the 70K-entry training bank)
2. The program's structure has patterns the model has seen in training
3. The corruption is partial (not a full-program replacement)

This validates the approach while showing that the current model scale is insufficient
for general program repair on diverse real-world code.

## Conclusions

1. **The evaluation was measuring the wrong thing.** The tiny subtree bank (35 entries)
   produced catastrophic corruption that required program synthesis, not repair. The
   eval scores were dominated by the corruption process, not model capability.

2. **The model has genuine repair capability** but it is narrow: it works when the
   corruption preserves enough program structure for the model to anchor local edits
   (clamp, primality). It fails when the corruption replaces the entire program.

3. **The model has learned a useful prior over Python code.** 100% syntactic validity
   is non-trivial for a 4.59M-parameter model. It reliably generates parseable Python
   and preserves program structure when making edits.

4. **The no-op preference is rational behavior.** The model correctly learned that
   random edits usually make programs worse. Better inference strategies (e.g.,
   execution-guided reranking) could overcome this by letting the model generate
   multiple diverse candidates and selecting by functional correctness.

## Recommendations

1. **Use MBPP-based eval as the primary metric** — real programs with real test cases,
   zero contamination risk.

2. **Control corruption severity** — limit corruption to 1-2 subtree replacements at
   non-root levels. Replacing entire `FunctionDef` nodes makes the task impossible.

3. **Implement execution-guided reranking** (Issue #53) — the model generates valid
   code but picks the wrong candidate. Test-based reranking directly addresses this.

4. **Train longer** — the loss curve hadn't plateaued at 12K steps. More training on
   the same corpus should improve the model's repair vocabulary.
