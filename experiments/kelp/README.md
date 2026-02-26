## Kelp

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

### v5: Scaled Corpus (10,442 programs)

Built a diverse training corpus from multiple sources:

| Source | Programs |
|--------|----------|
| Marin repo (local Python extraction) | 5,949 |
| codeparrot/github-code (streaming) | 5,000 |
| HumanEval (OpenAI) | 164 |
| **Total (after dedup + decontamination)** | **10,442** |

MBPP is held out entirely for evaluation. Eval task signatures are blocklisted from training to prevent leakage.

**Key results (v5, 12K steps, overnight_cpu preset):**

| Metric | Step 10K | Step 12K |
|--------|----------|----------|
| Syntactic validity | 100% | 100% |
| Exact match | 0% | 0% |
| Avg test pass rate (MBPP) | 2.3% | 1.9% |
| Best test pass rate (MBPP) | 10.0% | 8.0% |

SubtreeBank: 160,264 augmented entries. Learning curve showed steady improvement without overfitting (loss ~1.9, acc ~50% at step 2K — qualitatively different from toy corpus which saturated by step 2K). Step 10K slightly outperformed step 12K, suggesting mild overfitting.

**What we learned:**
- Corpus diversity scaled well (10K programs vs 15), no overfitting for most of training
- MBPP test pass rate dropped from v4's 55% — the model saw more diversity but had the same capacity, spreading its learning thinner
- 0% exact match on held-out corpus programs confirmed the model hasn't memorized training data
- 100% syntactic validity still holds at scale

### v6: Corruption Difficulty Curriculum

Added a noise difficulty curriculum that gradually increases corruption severity during training. Instead of always applying the maximum number of AST mutations, the curriculum ramps from easy (1 mutation) to hard (max mutations) over a configurable warmup fraction.

Available schedules: `constant` (default, same as before), `linear`, `cosine`.

**Key results (v6, overnight_cpu, linear curriculum):**

| Metric | Step 16K (corpus) | Step 24K (corpus) |
|--------|-------------------|-------------------|
| Syntactic validity | 100% | 100% |
| Exact match | 0% | 0% |
| Normalized match | 0% | 0% |
| Avg candidates/program | 7.2 | 7.6 |

MBPP eval was halted at step 16K after discovering a critical data error: the model was receiving corrupted programs without sufficient semantic signal about what the target program should be. The model could repair syntax perfectly but had no way to know *which* valid program to produce — it was solving an underdetermined problem.

**What we learned:**
- The curriculum helped training stability (longer before overfitting)
- 0% exact match across all checkpoints confirmed the core problem: the model needs intent signal, not just more data or training
- This directly motivated the v7 prompt conditioning work

### v7: Prompt Conditioning + Stack Edu (in progress)

Fundamental change: added **prompt/intent conditioning** so the model knows *what* program to repair toward. The encoding now supports an optional prompt prefix:

```
[PROMPT_START] docstring_bytes [PROMPT_END] [POS_k] context_bytes [EOS]
```

The prompt (typically a docstring) tells the model what the function should do, turning an underdetermined repair problem into a conditioned one.

**Key changes:**
- **Tokenizer**: 5 special tokens (PAD, SOS, EOS, PROMPT_START, PROMPT_END) when `prompt_tokens=True`; backward-compatible with 3-token layout for old checkpoints
- **Training**: extracts docstrings from clean programs, includes as prompt with probability `p_prompt` (default 0.5), strips docstrings from function bodies to prevent leakage
- **Inference**: `beam_search()` and `best_of_n()` accept an optional `prompt` string
- **Eval**: corpus eval uses extracted docstrings; MBPP eval uses task descriptions as prompts
- **Data**: Stack Edu (HuggingFaceTB/stack-edu Python subset) — educational Python code with high docstring coverage (~50-70%), streamed via `prepare_corpus.py --stack-edu-max N`

**Training setup:**
- Model: overnight_cpu preset (10M params, 4 layers, hidden_dim=256)
- Data: 50K functions from Stack Edu
- Hardware: Lambda Cloud GPU via SkyPilot
- 50K training steps, linear corruption curriculum, prompt conditioning enabled
- Checkpoints synced to `s3://oa-fomo-outputs/kelp/`
- W&B logging (project: `kelp`, run: `kelp-v7-prompt-conditioning`)

## Project Structure

```
experiments/kelp/
├── README.md                 # This file
├── CHANGELOG.md              # Detailed change history
├── DIAGNOSTIC_REPORT.md      # Analysis of v3 failure modes
├── kelp.md                   # Original research proposal
├── train.py                  # Training CLI (presets, W&B, prompt conditioning)
├── evaluate.py               # Eval on hand-crafted tasks
├── evaluate_corpus.py        # Held-out corpus repair evaluation
├── evaluate_mbpp.py          # MBPP benchmark evaluation
├── prepare_corpus.py         # Corpus preparation (multi-source + Stack Edu)
├── corpus.py                 # Corpus loading, docstring extraction
├── checkpointing.py          # Checkpoint save/load
├── infra/
│   ├── kelp-v7-train.yaml    # SkyPilot training task (Lambda/GCP)
│   ├── kelp-v7-eval.yaml     # SkyPilot eval task
│   └── launch_v7.sh          # End-to-end orchestrator (train → eval → download)
├── model/
│   ├── config.py             # TreeDiffusionConfig (includes prompt_tokens flag)
│   ├── presets.py            # Hardware-specific presets
│   └── model.py              # Transformer model (Grug blocks)
├── tree/
│   ├── train.py              # Training loop (corruption → TreeDiff → loss)
│   ├── mutation.py           # AST corruption (forward process)
│   ├── tree_diff.py          # Edit path computation
│   ├── beam_search.py        # Inference (best-of-N, beam search, prompt support)
│   ├── subtree_bank.py       # SubtreeBank indexing
│   ├── augmentation.py       # Bank augmentation orchestrator
│   ├── egraph_augmentation.py # E-graph variant generation (egglog)
│   ├── reranking.py          # Execution-guided reranking
│   ├── tokenizer.py          # AST edit tokenizer (prompt prefix encoding)
│   └── constrained_decoding.py # Grammar-constrained generation
└── training/
    └── optimizer.py          # Levanter AdamConfig integration
```

## Usage

### Prepare a corpus

```bash
# Basic corpus (Marin repo + GitHub Code + HumanEval)
uv run python experiments/kelp/prepare_corpus.py \
  --output experiments/kelp/corpus.txt --max-github 5000

# With Stack Edu educational Python (recommended for v7+)
uv run python experiments/kelp/prepare_corpus.py \
  --output experiments/kelp/corpus_v7.txt --stack-edu-max 50000
```

### Train

```bash
# Laptop, overnight (CPU)
JAX_PLATFORMS=cpu uv run python experiments/kelp/train.py \
  --preset overnight_cpu --steps 12000 --augment \
  --corpus-file experiments/kelp/corpus.txt \
  --checkpoint-interval 2000 --output-dir checkpoints/kelp-edit

# GPU with prompt conditioning + corruption curriculum (v7 recipe)
uv run python experiments/kelp/train.py \
  --preset overnight_cpu --steps 50000 --augment \
  --corpus-file experiments/kelp/corpus_v7.txt \
  --prompt-conditioning --p-prompt 0.5 \
  --corruption-curriculum linear \
  --wandb-project kelp --wandb-run-name my-run \
  --checkpoint-interval 5000 --output-dir checkpoints/kelp-edit-v7
```

### Evaluate

```bash
# Corpus repair evaluation (exact match, syntactic validity)
JAX_PLATFORMS=cpu uv run python experiments/kelp/evaluate_corpus.py \
  --checkpoint-dir checkpoints/kelp-edit-v7 \
  --corpus-file experiments/kelp/corpus_v7.txt \
  --num-tasks 50 --n-best-of 16

# MBPP benchmark (test pass rate)
JAX_PLATFORMS=cpu uv run python experiments/kelp/evaluate_mbpp.py \
  --checkpoint-dir checkpoints/kelp-edit-v7 \
  --corpus-file experiments/kelp/corpus_v7.txt \
  --max-tasks 50 --n-best-of 16
```

### Cloud training via SkyPilot

```bash
# One-command pipeline: train → eval → download → teardown
bash experiments/kelp/infra/launch_v7.sh --wandb

# Or step-by-step:
sky launch -c kelp-v7 experiments/kelp/infra/kelp-v7-train.yaml \
  --env WANDB_API_KEY --retry-until-up -y
sky exec kelp-v7 experiments/kelp/infra/kelp-v7-eval.yaml
rsync -avz kelp-v7:~/sky_workdir/checkpoints/kelp-edit-v7/ checkpoints/kelp-edit-v7/
sky down kelp-v7 -y
```

## Roadmap & Contributing

Kelp is an open experiment within the [Marin project](https://marin.community/). Contributions are welcome — here's what's ahead and how to help.

### Roadmap

**Near-term (validating prompt conditioning):**
- Analyze v7 results to measure the impact of prompt conditioning on exact match and test pass rates
- Improve edit position prediction accuracy — the model often picks the right replacement but the wrong location (see `tree/beam_search.py`)
- Fix whitespace accumulation in corruption/repair cycles that causes spurious indentation diffs (see `tree/mutation.py`)
- Experiment with `p_prompt` values — currently 0.5, higher values may improve conditioned repair at the cost of unconditioned generalization

**Medium-term (scaling model and data):**
- Scale to the `single_gpu` preset (768d/12L, ~300M params) on a longer A100 run
- Increase Stack Edu corpus to 100K–500K programs
- Add multi-edit prediction — currently the model predicts one edit per forward pass; batching edits could speed inference significantly
- Implement MBPP pass@k metrics for direct comparison with code generation baselines

**Long-term (transfer learning & scale):**
- Transfer from Marin's pretrained 8B model into a tree diffusion model (the original vision from [kelp.md](kelp.md))
- Support multi-language tree diffusion (TypeScript, Rust) by swapping the AST parser
- Condition on richer prompts (test cases, type signatures, natural language specs)
- Scale to TPU pods using Marin's Ray-based executor infrastructure

### How to Contribute

**Run an experiment.** The fastest way to contribute is to train a model and report results. The whole pipeline runs on a laptop:

```bash
# 1. Set up the environment
git clone https://github.com/marin-community/marin && cd marin
uv sync

# 2. Prepare a corpus (streams Stack Edu, ~5 minutes)
uv run python experiments/kelp/prepare_corpus.py \
  --output experiments/kelp/corpus.txt --stack-edu-max 10000

# 3. Train overnight (~5 hours on Apple Silicon)
JAX_PLATFORMS=cpu uv run python experiments/kelp/train.py \
  --preset overnight_cpu --steps 12000 --augment \
  --prompt-conditioning \
  --corpus-file experiments/kelp/corpus.txt \
  --checkpoint-interval 2000 --output-dir checkpoints/my-run

# 4. Evaluate
JAX_PLATFORMS=cpu uv run python experiments/kelp/evaluate_corpus.py \
  --checkpoint-dir checkpoints/my-run \
  --corpus-file experiments/kelp/corpus.txt
```

**Pick up a known issue.** Some concrete improvements we know are needed:

- **Whitespace fix** — corruption/repair cycles accumulate extra indentation; small, well-scoped bug in `tree/mutation.py`
- **Edit position accuracy** — the model often predicts a valid replacement but applies it at the wrong AST location; see `tree/beam_search.py`
- **New corpus sources** — write a function that returns `list[str]` of Python programs, plug it into `prepare_corpus.py`. Dedup and decontamination are automatic.
- **New augmentation strategies** — add a new source to `augmentation.py`'s `augment_bank()` pipeline; e-graph augmentation (`tree/egraph_augmentation.py`) is a good example.

**Run on a GPU.** The SkyPilot configs in `infra/` make cloud training easy. If you have Lambda, GCP, or AWS credits:

```bash
# Edit infra/kelp-v7-train.yaml to set your cloud provider, then:
bash experiments/kelp/infra/launch_v7.sh --wandb
```

### Development

All code lives under `experiments/kelp/` with tests in `tests/kelp/`. Follow [Marin's contribution guidelines](../../CONTRIBUTING.md).

```bash
# Run kelp tests
JAX_PLATFORMS=cpu uv run pytest tests/kelp/ -x -q

# Run pre-commit checks
./infra/pre-commit.py --all-files
```

## References

- [Tree Diffusion (Tseng et al., 2024)](https://arxiv.org/abs/2405.20519) — AST-based diffusion for program synthesis
- [MBPP (Austin et al., 2021)](https://arxiv.org/abs/2108.07732) — Mostly Basic Programming Problems benchmark
- [HumanEval (Chen et al., 2021)](https://arxiv.org/abs/2107.03374) — OpenAI's code generation benchmark
- [egglog](https://egglog-python.readthedocs.io/) — Equality saturation for rewrite-based optimization
- [Marin](https://marin.community/) — Open research infrastructure for LLM development
