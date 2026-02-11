# Kelp

```
        ~  ~  ~  ~  ~  ~  ~
       ~  ~  ~  ~  ~  ~  ~  ~
      )  )  )  )  )  )  )  )  )
     (  (  (  (  (  (  (  (  (
      )  )  )  )  )  )  )  )  )
     (  (  (  (  (  (  (  (  (
      )  )  )  )  )  )  )  )  )
     (  (  (  (  (  (  (  (  (
      )  )  )  )  )  )  )  )  )
     (  (  (  (  (  (  (  (  (
      \  |  |  |  |  |  |  |  /
       \ |  |  |  |  |  |  | /
        \|  |  |  |  |  |  |/
         |  |  |  |  |  |  |
         |  |  |  |  |  |  |
    ~~~~~|~~|~~|~~|~~|~~|~~|~~~~~
    -----+--+--+--+--+--+--+-----
         |__|__|__|__|__|__|
              KELP
    Tree Diffusion for Program Repair
```

## Goals

Kelp explores a novel approach to program synthesis: **tree diffusion** over Python abstract syntax trees (ASTs). Instead of generating programs token-by-token like a language model, Kelp learns to *repair* corrupted programs by predicting structured edits at the AST level.

The core hypothesis: by constraining the search space to syntactically valid AST transformations, a small model can learn to reliably fix broken programs — and every intermediate state in the repair process is a valid Python program.

Long-term, the project aims to:

1. **Validate tree diffusion** as a program repair strategy on real-world Python code
2. **Scale from scratch to transfer**: train small models from scratch first, then adapt pretrained LLMs (Marin 8B) into tree diffusion models
3. **Demonstrate scaling laws**: measure how repair quality improves with corpus diversity, model size, and compute

## Architecture

### The Pipeline

```
  Clean Program     Corrupted Program       Edit Sequence         Repaired Program
  ┌───────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
  │ def add(   │     │ def add(      │     │ POS_7         │     │ def add(      │
  │   a, b):   │ ──► │   a, b):     │ ──► │ return a + b  │ ──► │   a, b):     │
  │   return   │     │   return     │     │ EOS           │     │   return     │
  │   a + b    │     │   len(a)     │     │               │     │   a + b      │
  └───────────┘     └───────────────┘     └───────────────┘     └───────────────┘
                    Forward Process        AR Model Predicts      Applied Edit
                    (AST Corruption)       (Position + Tokens)
```

### Forward Process (Corruption)

The forward process corrupts a clean Python program through a sequence of AST-level mutations:

- Parse the program into an AST
- Select a random non-root subtree node
- Replace it with a type-compatible subtree from the **SubtreeBank** (a pre-indexed palette of real code fragments)
- The result is always a syntactically valid Python program

Multiple corruption steps create a "diffusion trajectory" — a sequence of progressively more corrupted programs, each one valid Python.

### SubtreeBank

The SubtreeBank is a dictionary mapping AST node types (BinOp, Return, If, etc.) to lists of real code fragments extracted from the training corpus. It is augmented from multiple sources:

- **Original**: subtrees extracted directly from training programs
- **Renamed**: variable names systematically swapped for diversity
- **Perturbed**: numeric literals shifted by small deltas
- **Synthetic**: template-generated expressions (arithmetic, comparisons, boolean ops)
- **E-graph**: semantically equivalent variants generated via [egglog](https://egglog-python.readthedocs.io/) equality saturation (e.g., `a + b` → `b + a`, `x > 0` → `0 < x`)

### Training

Each training step:
1. Pick a random clean program from the corpus
2. Corrupt it N steps using the SubtreeBank (forward process)
3. Compute the `TreeDiff` — the minimal edit path back to the clean program
4. Pick a random step along that path as the training target
5. The model learns to predict: `[position_token, replacement_tokens..., EOS]`

The model is a standard causal transformer (using [Grug](https://github.com/marin-community/marin/tree/main/lib/levanter/src/levanter/grug) building blocks from Levanter) that operates on flat token sequences, not tree structures directly.

### Inference

At inference time, the model iteratively repairs a corrupted program:

1. Tokenize the corrupted program
2. The model autoregressively predicts an edit: position token → replacement tokens → EOS
3. Apply the edit to produce a new (hopefully less corrupted) program
4. Repeat for up to `max_depth` steps

Two inference strategies:
- **Best-of-N**: generate N independent repair trajectories, return the best
- **Beam Search**: maintain a beam of candidates, expand and prune by cumulative log-probability

### Evaluation

Programs are evaluated by:
1. **Syntactic validity**: does the output parse as Python?
2. **Exact match**: is the output identical to the original?
3. **Test pass rate**: does the output pass the test cases? (execution-guided reranking selects the best candidate)

## Model Presets

| Preset | Dims | Layers | Heads | Params | Target Hardware |
|--------|------|--------|-------|--------|-----------------|
| `toy` | 64 | 2 | 2 | ~0.1M | Unit tests |
| `overnight_cpu` | 256 | 4 | 4 | ~4.6M | Laptop (overnight) |
| `laptop` | 512 | 6 | 8 | ~125M | Laptop (multi-day) |
| `single_gpu` | 768 | 12 | 12 | ~300M | 1x A100 |
| `tpu_v4_8` | 2048 | 24 | 16 | ~1B | TPU v4-8 |

## Experiment History

### v1–v3: Toy Corpus (15 programs)

The first training runs used a hardcoded corpus of 15 simple Python functions (`add`, `sub`, `mul`, `neg`, `abs_val`, `max_val`, `min_val`, `clamp`, `double`, `square`, plus 5 slightly more complex programs).

**Key results (v3, 12K steps, overnight_cpu preset):**
- 97.9% training accuracy — the model memorized the tiny corpus
- 4.0% average test pass rate on eval tasks
- 100% syntactic validity

**What we learned:**
- The pipeline works end-to-end: corruption → training → inference → evaluation
- The model achieves perfect syntax (the tree diffusion invariant holds)
- The model rationally prefers no-op over random edits on out-of-distribution inputs — this is correct behavior with a tiny bank, not a bug
- Corpus diversity is the bottleneck, not model capacity

**Bugs found and fixed:**
- Catastrophic corruption: root-level AST mutations destroyed entire programs (#57)
- No-op bias: beam search always selected unchanged programs over edited ones (#58)
- Eval contamination: training data included eval task programs (#56)
- Tiny eval-time subtree bank (35 entries) made repair nearly impossible (#52)

### v4: Toy Corpus + E-Graph Augmentation

Added e-graph-based expression augmentation using egglog equality saturation. Rewrite rules generate semantically equivalent expression variants (commutativity, comparison flips, double negation, etc.) to diversify the SubtreeBank without new source programs.

**Key results (v4 step-4000, best checkpoint):**
- 55.3% average test pass rate (13.8x improvement over v3)
- 96.7% best-of-16 test pass rate
- 1.7% exact match rate
- 100% syntactic validity

**What we learned:**
- All the v3 bug fixes compound: the model genuinely repairs programs now
- Step 4K beats step 6K on all metrics — the toy corpus overfits quickly
- E-graph augmentation contributes to bank diversity (616 entries vs 55 original)
- The toy dataset has exhausted its signal value; further optimization has diminishing returns

### v5: Scaled Corpus (10,442 programs) — In Progress

Built a diverse training corpus from multiple sources:

| Source | Programs |
|--------|----------|
| Marin repo (local Python extraction) | 5,949 |
| codeparrot/github-code (streaming) | 5,000 |
| HumanEval (OpenAI) | 164 |
| **Total (after dedup + decontamination)** | **10,442** |

MBPP is held out entirely for evaluation. Eval task signatures are blocklisted from training to prevent leakage.

**Training status:** 12K steps on overnight_cpu preset. SubtreeBank: 160,264 augmented entries. Learning curve shows steady improvement without overfitting (loss ~1.9, acc ~50% at step 2K — qualitatively different from toy corpus which saturated by step 2K).

**Expected impact (from scaling predictions):** 50–65% average test pass rate on eval tasks, with simple functions reaching near-100%.

## Project Structure

```
experiments/kelp/
├── README.md                 # This file
├── CHANGELOG.md              # Detailed change history
├── DIAGNOSTIC_REPORT.md      # Analysis of v3 failure modes
├── kelp.md                   # Original research proposal
├── train.py                  # Training entry point
├── evaluate.py               # Eval on 10 hand-crafted tasks
├── evaluate_mbpp.py          # Eval on held-out MBPP tasks
├── prepare_corpus.py         # Corpus preparation (multi-source)
├── corpus.py                 # Corpus loading utilities
├── corpus.txt                # Generated training corpus (10,442 programs)
├── checkpointing.py          # Checkpoint save/load
├── model/
│   ├── config.py             # TreeDiffusionConfig dataclass
│   ├── presets.py            # Hardware-specific presets
│   └── model.py              # Transformer model (Grug blocks)
├── tree/
│   ├── train.py              # Training loop (corruption → TreeDiff → loss)
│   ├── mutation.py           # AST corruption (forward process)
│   ├── tree_diff.py          # Edit path computation
│   ├── beam_search.py        # Inference (best-of-N, beam search)
│   ├── subtree_bank.py       # SubtreeBank indexing
│   ├── augmentation.py       # Bank augmentation orchestrator
│   ├── egraph_augmentation.py # E-graph variant generation (egglog)
│   ├── reranking.py          # Execution-guided reranking
│   └── constrained_decoding.py # Grammar-constrained generation
└── training/
    └── optimizer.py          # Levanter AdamConfig integration
```

## Usage

```bash
# Prepare a training corpus
uv run python experiments/kelp/prepare_corpus.py \
  --output experiments/kelp/corpus.txt --max-github 5000

# Train (laptop, overnight)
JAX_PLATFORMS=cpu uv run python experiments/kelp/train.py \
  --preset overnight_cpu --steps 12000 --augment \
  --corpus-file experiments/kelp/corpus.txt \
  --checkpoint-interval 2000 --output-dir checkpoints/kelp-edit-v5

# Evaluate on hand-crafted tasks
JAX_PLATFORMS=cpu uv run python experiments/kelp/evaluate.py \
  --checkpoint-dir checkpoints/kelp-edit-v5 \
  --corpus-file experiments/kelp/corpus.txt \
  --num-corruptions 10

# Evaluate on MBPP
JAX_PLATFORMS=cpu uv run python experiments/kelp/evaluate_mbpp.py \
  --checkpoint-dir checkpoints/kelp-edit-v5 \
  --corpus-file experiments/kelp/corpus.txt
```

## References

- [Tree Diffusion (Tseng et al., 2024)](https://arxiv.org/abs/2405.20519) — AST-based diffusion for program synthesis
- [MBPP (Austin et al., 2021)](https://arxiv.org/abs/2108.07732) — Mostly Basic Programming Problems benchmark
- [HumanEval (Chen et al., 2021)](https://arxiv.org/abs/2107.03374) — OpenAI's code generation benchmark
- [egglog](https://egglog-python.readthedocs.io/) — Equality saturation for rewrite-based optimization
- [Marin](https://marin.community/) — Open research infrastructure for LLM development
