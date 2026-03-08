# JPEG Tokenizer Project: Repo-Aligned Implementation Plan

## Research Framing

This is **not** primarily an image-modeling project and **not** a project to build a better JPEG model.

It is a research project about:

> what makes a tokenizer compatible with autoregressive next-token prediction in non-symbolic domains

JPEG is useful here because it gives us a controlled ladder of representations over the same underlying content:

- entropy-coded bytes,
- explicit syntax symbols,
- quantized coefficient structures.

That makes JPEG a good **instrument** for isolating tokenizer properties, not the real subject of the project.

The core claim we want to test is:

> a good tokenizer for a non-symbolic domain is not merely compressive or reconstructive; it induces a sequence with useful prefixes, manageable effective state complexity, and rollout stability under small prediction errors

If the project works, the output should be a clearer taxonomy of tokenizer properties for autoregressive modeling, not just a statement that one JPEG-derived representation trained better than another.

## Why This Should Generalize Beyond JPEG

JPEG is the first testbed because it is familiar, inspectable, and gives a clean ladder from entropy-coded bytes to more explicit structure.

But the intended lesson is broader. The same style of decomposition should apply to other non-symbolic domains, especially domains with codec structure and hidden decoder state. Audio is the obvious next case:

- raw/container bytes,
- packet or frame syntax,
- quantized transform/codebook structure,
- explicit streaming boundaries and resets,
- varying amounts of entropy/state coupling.

In particular, an audio codec such as Opus should let us ask the same question in a second domain:

> are the properties that make a tokenizer good for autoregressive modeling stable across modalities, or are they codec-specific accidents?

That follow-on is important because it pushes the project toward a general account of tokenizer quality for non-symbolic domains rather than a JPEG-only story.

## Goal

Build a controlled image-tokenizer research stack that fits Marin/Levanter conventions:

- simple, copyable experiment layout,
- deterministic data loading,
- random-access training data first,
- minimal model surface,
- clear decision gates before adding infrastructure.

The question is not just "can we model JPEG-derived tokens?" It is:

> Which JPEG-like tokenizations produce sequences that are learnable and stable under autoregressive next-token prediction?

More generally:

> Which properties make a tokenizer good for autoregressive modeling in non-symbolic domains?

JPEG is the testbed for that broader question.


## Repo-Aligned Principles

This should follow the **Grug philosophy** without being a Grug project directly.

1. **Template-first, not framework-first.**
   Start from a small experiment directory with explicit files and copy-paste variants. Do not begin by adding a reusable tokenizer framework.

2. **Use existing Levanter dataset interfaces first.**
   The default target should be `AsyncDataset` / `DirectDatasetComponent` or a standard Levanter token cache. Do not adopt WebDataset as the training interface unless we can show the existing path is the bottleneck.

3. **Random access is a requirement, not a nice-to-have.**
   Levanter’s loader and shuffle story assumes indexable datasets. If we store raw shards in tar/WebDataset form, that is a storage detail; training should still see a random-access dataset or cache.

4. **One strong baseline before a ladder of abstractions.**
   Start with a small decoder-only transformer and 2-3 tokenizer variants. Do not build RVQ, learned codebooks, byte packing, restart-marker sweeps, and large-scale infrastructure all at once.

5. **Deterministic, inspectable preprocessing.**
   Fix resize/crop/color handling, JPEG quality/tables, and token linearization early. If we change them, we version the variant rather than hiding behavior behind conditionals.


## Proposed Project Layout

Use a dedicated experiment tree that mirrors the Grug "base plus copied variants" workflow:

```text
experiments/jpeg_tokenizer/
  README.md
  base/
    tokenizers.py
    data.py
    model.py
    train.py
    launch.py
    eval.py
  bytes_restart/
  symbols/
  coeffs/
```

Notes:

- `base/` is the canonical simple path.
- Variant directories are copy-first and local, not inheritance-heavy.
- If a change wins, upstream it back into `base/`.


## Initial Scope

We should narrow the first pass to the minimum set that can answer the core question.

### Keep in scope

- Canonical image preprocessing to a fixed size.
- A small tokenizer ladder:
  - normalized JPEG bytes,
  - pre-Huffman JPEG symbol stream,
  - low-frequency quantized coefficient stream.
- A small decoder-only transformer baseline.
- Teacher-forced loss, prefix sensitivity, and rollout validity.

### Push out of the first pass

- learned VQ or RVQ tokenizers,
- arithmetic/range-coded controls,
- large model scaling sweeps,
- multimodal conditioning,
- elaborate executor/pipeline integration before the data path is validated.

The rule for scope is:

> every early implementation choice should help us answer the tokenizer question more cleanly, not make the image-modeling stack more elaborate


## Model Plan

Use a **grug-ish** model in philosophy:

- decoder-only,
- plain causal transformer,
- minimal config knobs,
- no MoE,
- no special multimodal machinery,
- easy to fork into local variants.

This does **not** need to live under `experiments/grug/`. It should just copy the same engineering style:

- explicit `model.py`, `train.py`, `launch.py`,
- simple dataclass configs,
- readable code over reusable abstractions,
- metrics that line up with existing Levanter conventions.

Suggested first config:

- `hidden_dim`: 512-768
- `num_layers`: 8-12
- `num_heads`: 8-12
- `max_seq_len`: choose per tokenizer family, initially 2048 or 4096
- `vocab_size`: tokenizer-dependent, surfaced explicitly

The first baseline should be small enough to run frequent ablations, not impressive enough to publish by itself.


## Data Strategy

### Canonical source representation

Pick one canonical image corpus and one canonical preprocessing recipe:

- fixed resolution, e.g. `256x256`,
- fixed color handling, preferably start with `Y` or `RGB` and do not mix,
- fixed resize/crop policy,
- fixed JPEG encoder settings when generating canonical JPEGs.

The core rule is:

> Store one canonical image representation and derive token streams from it deterministically.

That avoids storing separate raw corpora per tokenizer.


### Training data interface: random access first

The repo-native first choice is:

1. produce token sequences,
2. expose them as an `AsyncDataset`,
3. plug them into `LmDataConfig` via `DirectDatasetComponent`, or
4. materialize a Levanter token cache if we want standard cache-backed loading.

This is the right first shape because Levanter already expects random access for:

- shuffling,
- deterministic mixture behavior,
- resume behavior,
- fast evaluation.

Concrete sketch:

```python
from levanter.data.text import DirectDatasetComponent, LmDataConfig

data = LmDataConfig(
    tokenizer="passthrough",
    vocab_size=vocab_size,
    components={
        "jpeg_tokens": DirectDatasetComponent(
            datasets={"train": train_ds, "validation": val_ds}
        )
    },
    shuffle=True,
)
```

If precomputing token sequences is easier than JIT tokenization, we can also build a standard cache and use the passthrough tokenizer:

```yaml
data:
  tokenizer: "passthrough"
  vocab_size: 4096
  cache_dir: "gs://.../jpeg-symbol-cache"
```


### Where WebDataset fits

WebDataset is a **possible raw storage format**, not the default answer to the training interface question.

Use WebDataset only if one of these becomes true:

1. the canonical image corpus is most naturally acquired as tar shards,
2. per-object remote reads are the dominant bottleneck,
3. we need shard-friendly bulk preprocessing before cache construction.

Even then, the preferred shape is:

`raw shard storage -> preprocessing/token extraction -> random-access cache or AsyncDataset -> training`

Not:

`raw tar stream -> directly feed LM training`

That direct streaming path cuts against Levanter’s random-access/shuffle assumptions.


## Tokenizer Ladder

The first ladder should be deliberately small.

The purpose of the ladder is to separate candidate properties of a "good tokenizer":

- how much predictive information prefixes carry,
- how much hidden decoder state the model must implicitly track,
- how brittle rollout is after a local mistake,
- how geometrically local token perturbations are in decoded space.

### Tier 1: normalized JPEG bytes

Token stream:

- canonicalized JPEG bytes,
- strip metadata,
- fix quantization/Huffman tables if needed,
- optionally isolate scan payload vs full file bytes.

Purpose:

- maximum codec-state brittleness baseline,
- easiest baseline to explain,
- likely unstable at rollout, which is useful.


### Tier 2: JPEG symbol stream

Token stream:

- DC deltas,
- AC `(run, size, amplitude)` style symbols,
- EOB markers,
- no Huffman coding.

Purpose:

- preserve JPEG structure,
- remove entropy-coder state,
- test whether the failure mode is really entropy/state brittleness rather than JPEG syntax itself.


### Tier 3: low-frequency coefficient stream

Token stream:

- first `K` zigzag coefficients per block,
- signed integer representation with an explicit bounded vocabulary,
- fixed block ordering.

Purpose:

- geometry-respecting baseline,
- simpler semantics,
- likely longer but more stable sequence.


### Explicitly deferred

Only add these after the first three work:

- block codebooks,
- RVQ on DCT vectors,
- entropy-coded index controls,
- restart-marker sweeps,
- byte-pair/uint16 packing ablations.


## Implementation Phases

## Phase 0: Feasibility spike

Deliverables:

- choose corpus,
- choose canonical preprocessing,
- implement token extraction for the first 3 tokenizers on a small sample,
- measure sequence-length distributions and vocab sizes,
- verify deterministic decode/re-encode behavior.

Questions to answer:

- Are byte lengths and coefficient lengths in a range we can realistically batch?
- Is the symbol vocabulary compact enough for a first-pass softmax?
- Do we need separate train/eval sequence lengths by tokenizer family?

Exit criteria:

- one notebook or script that can inspect 1k-10k images,
- token stats dumped to disk,
- a concrete decision on `Y` vs `RGB`,
- a concrete decision on `256` vs another size.


## Phase 1: Minimal training path

Deliverables:

- a small `experiments/jpeg_tokenizer/base/` experiment,
- one decoder-only baseline model,
- one data path using `AsyncDataset` or direct cache construction,
- training on one tokenizer family end to end.

Recommended default:

- implement a simple indexable dataset of precomputed token sequences,
- feed it through `DirectDatasetComponent`,
- use `tokenizer="passthrough"` and explicit `vocab_size`.

Exit criteria:

- train loss goes down,
- eval path works,
- checkpoint/resume works,
- no special storage format has been introduced yet.


## Phase 2: Core ladder comparison

Deliverables:

- bytes baseline,
- symbol-stream baseline,
- coefficient-stream baseline,
- matched small-run comparison with the same model family and similar compute budget.

Metrics:

- teacher-forced loss,
- bits-per-pixel or another normalized cross-tokenizer metric,
- sequence length statistics,
- tokens/sec and loader stall time,
- decode-validity under rollout.

Exit criteria:

- at least one plot/table showing meaningful separation across tokenizers,
- evidence about whether bytes are actually brittle in practice,
- evidence about whether the existing data path is keeping up.


## Phase 3: Stability diagnostics

Deliverables:

- prefix perturbation harness,
- content-local perturbation protocol,
- codec-desync perturbation protocol for byte streams,
- rollout degradation metrics.

Metrics:

- next-token KL/TV after perturbation,
- decode survival length,
- reconstruction collapse under forced token errors.

Exit criteria:

- a clean answer to the main research claim:
  does entropy-coder state break rollout much more than syntax/coefficient tokenization?


## Phase 4: Infrastructure decision point

Only after Phases 1-3 should we decide whether we need:

- cache-backed preprocessing,
- WebDataset raw shards,
- executor-integrated preprocessing steps,
- larger-scale sweeps,
- codebook/RVQ variants.

This is the right point to ask whether we should operationalize the pipeline in Marin proper.


## Decision Gates

### Decision 1: Direct dataset vs cache

Start with `DirectDatasetComponent` if:

- the corpus subset fits operationally,
- token extraction can be precomputed once,
- random access is easy to implement.

Move to cache construction if:

- loader latency dominates,
- direct dataset startup/resume is clumsy,
- we want standard Levanter cache semantics and tooling.


### Decision 2: Cache vs WebDataset

Prefer Levanter cache if:

- training is the immediate goal,
- random access matters more than raw ingest convenience.

Introduce WebDataset only if:

- raw storage format is the bottleneck,
- object-store GET overhead is dominating,
- tar-shard preprocessing materially simplifies ingestion.


### Decision 3: JIT tokenization vs precomputed tokens

Prefer precomputed tokens first.

JIT tokenization sounds attractive, but it adds noise to the main experiment:

- loader timing becomes tokenizer-dependent,
- reproducibility is harder,
- failures blend data bugs and modeling bugs.

If we later want JIT tokenization, do it as an explicit throughput experiment.


## Evaluation Plan

Keep the first evaluation stack narrow and aligned with the research question.

### Primary metrics

- training/eval NLL,
- normalized likelihood metric such as bits-per-pixel,
- tokens/sec,
- loader stall time,
- rollout decode success rate.

### Stability metrics

- next-token distribution shift after a small perturbation,
- decoded-image degradation after forced token corruption,
- fraction of future blocks that remain semantically aligned.

### What not to optimize first

- downstream image quality benchmarks,
- giant model scaling laws,
- polished generation demos.

Those may become useful later, but they are downstream of the main scientific question. The first bar is explanatory power about tokenizer properties, not leaderboard performance.


## Recommended First Variant Set

If we want the fastest credible first pass, do this:

1. `base/`: coefficient tokens only.
2. `bytes/`: normalized JPEG bytes.
3. `symbols/`: pre-Huffman JPEG symbols.

Why this ordering:

- coefficient tokens are the safest learnable baseline,
- bytes give the strongest contrast case,
- symbols test the central entropy-state hypothesis cleanly.

Do **not** start with RVQ. It is an attractive direction, but it widens the project before we have validated the core claim.


## Minimal Code Sketch

Suggested split of responsibilities:

```python
# experiments/jpeg_tokenizer/base/data.py
class JpegTokenDataset(AsyncDataset[np.ndarray]):
    ...

# experiments/jpeg_tokenizer/base/tokenizers.py
def encode_jpeg_bytes(image) -> np.ndarray: ...
def encode_jpeg_symbols(image) -> np.ndarray: ...
def encode_dct_coeffs(image) -> np.ndarray: ...

# experiments/jpeg_tokenizer/base/model.py
@dataclass(frozen=True)
class JpegLmConfig:
    vocab_size: int
    max_seq_len: int
    ...

# experiments/jpeg_tokenizer/base/launch.py
run = ExecutorStep(...)
```

The important point is not the exact names. The important point is keeping tokenizer logic, data logic, and model/training wiring separate but local.


## Risks

### Sequence length blows up

Mitigation:

- start at `256x256`,
- start with luma-only if needed,
- cap coefficient count for the coefficient baseline,
- defer packing ablations.


### Symbol vocabulary gets awkward

Mitigation:

- choose an explicit bounded encoding early,
- version the encoding,
- avoid "smart" fallback logic.


### Loader performance dominates

Mitigation:

- precompute token sequences,
- benchmark direct dataset first,
- move to cache before moving to WebDataset.


### Too many tokenizer variants too early

Mitigation:

- require each new tokenizer family to answer a new question,
- do not add RVQ/codebooks until bytes vs symbols vs coefficients is working.


## Concrete Recommendation

The revised plan should be:

1. Create `experiments/jpeg_tokenizer/base/` as a template-first experiment tree.
2. Use a small decoder-only transformer that is grug-ish in style, but separate from `experiments/grug/`.
3. Start with precomputed token sequences exposed through `AsyncDataset` + `DirectDatasetComponent`.
4. Use `passthrough` tokenization and explicit vocab sizing.
5. Compare exactly three tokenizer families first: bytes, symbols, coefficients.
6. Add Levanter cache construction if the direct path is operationally weak.
7. Add WebDataset only if raw storage/ingest proves to be the actual bottleneck.

That keeps the project aligned with the repo’s style: simple first, deterministic first, copy-first, and only as much infrastructure as the evidence justifies.

It also keeps the research claim honest: the project is about understanding what makes tokenization work for non-symbolic autoregressive modeling, with JPEG as a controlled probe.


## Detailed Implementation Steps

This section is additive to the plan above. It is intended to turn the current plan into an execution checklist without changing the framing or decisions already captured.

### Step 0: Freeze The V0 Decisions In A Short Design Note

Before writing experiment code, add a short V0 design note, either at the top of `experiments/jpeg_tokenizer/README.md` or as a sibling markdown file, that locks:

- the initial corpus,
- image resolution,
- color mode (`Y` or `RGB`),
- JPEG encoding settings,
- the first `K` for coefficient truncation,
- the variant names,
- the first model size,
- the initial evaluation metrics.

The point is to avoid drifting defaults across scripts and copied variants.

### Step 1: Create The Base Experiment Tree

Create `experiments/jpeg_tokenizer/base/` with the same top-level split as grug:

- `tokenizers.py`
- `data.py`
- `model.py`
- `train.py`
- `launch.py`
- `eval.py`

Add a minimal `README.md` in `experiments/jpeg_tokenizer/` that explains:

- the research question,
- the three initial tokenizer families,
- which directory is canonical,
- how variants are expected to be created.

### Step 2: Implement Deterministic Canonical JPEG Preprocessing

In `experiments/jpeg_tokenizer/base/tokenizers.py` or a small local helper module if the file becomes too long:

- decode images deterministically,
- resize/crop to the chosen fixed resolution,
- convert to the chosen color mode,
- encode to JPEG with fixed quality/subsampling/tables,
- strip metadata,
- return canonical JPEG bytes.

Add one small helper that computes a stable checksum for the canonical bytes so later scripts can detect accidental drift.

Exit criterion:

- running canonicalization twice on the same image yields identical bytes.

### Step 3: Build A Phase-0 Inspection Script

Add `scripts/jpeg_tokenizer/inspect_representations.py`.

That script should:

- load a small sample,
- canonicalize each image,
- emit bytes, symbols, and coefficient token streams,
- record sequence length distributions,
- record value ranges and approximate vocab sizes,
- write summary stats to disk,
- optionally dump a small set of decoded or reconstructed examples for sanity checking.

The script should answer the open Phase 0 questions already listed in the plan instead of introducing new ones.

Concrete outputs to write:

- `stats/bytes.json`
- `stats/symbols.json`
- `stats/coeffs.json`
- one short markdown summary of the decisions taken from those numbers

### Step 4: Lock The Exact Token Encodings

Once the inspection script runs on a representative sample, freeze one explicit encoding for each tokenizer family.

For bytes:

- reserve special token ids,
- decide whether to model the full file or scan payload only,
- keep this fixed across all early runs.

For symbols:

- decide the exact emitted units,
- fix one bounded representation for amplitudes,
- do not add compatibility branches for alternate encodings.

For coefficients:

- fix block order,
- fix zigzag order,
- fix `K`,
- fix how out-of-range values are handled.

Add comments describing the encoding as a whole, not line-by-line commentary.

### Step 5: Build A Precomputed Token Store

Do not make the training loop responsible for tokenization.

Add a builder script, for example `scripts/jpeg_tokenizer/build_token_store.py`, that:

- reads source images,
- canonicalizes them,
- tokenizes them into one chosen family,
- writes token sequences to a random-access store,
- writes a manifest with split, example id, token count, and checksum,
- can be rerun deterministically.

For the first pass, a simple indexable store is enough. The store only needs to support:

- `len(dataset)`,
- fetch by integer index,
- optional small-batch fetch,
- reproducible train/validation splits.

Only move to a Levanter cache if this simple store proves operationally weak.

### Step 6: Implement A Thin Dataset Adapter

In `experiments/jpeg_tokenizer/base/data.py`:

- add a small dataset class that reads precomputed token sequences by index,
- keep it finite and indexable,
- expose it through `AsyncDataset` or a synchronous dataset wrapped with `as_async_dataset()`,
- adapt it to Levanter LM examples.

Keep the first implementation boring. The job of the dataset layer is only:

- read precomputed sequences,
- shape them into LM-ready training examples,
- preserve deterministic ordering when shuffle is disabled.

Do not mix tokenization, augmentation, and training-window construction unless Phase 0 data forces that choice.

### Step 7: Wire `LmDataConfig` Through `DirectDatasetComponent`

Use the existing repo-native path first.

The first training config should:

- use `tokenizer="passthrough"`,
- set `vocab_size` explicitly,
- pass train and validation datasets through `DirectDatasetComponent`,
- enable shuffle using the standard Levanter controls.

This keeps the project aligned with Levanter’s expected data surface and minimizes custom training code.

### Step 8: Copy The Grug Model And Train Stack

Populate `experiments/jpeg_tokenizer/base/model.py`, `train.py`, and `launch.py` by copying the corresponding grug files and simplifying only where needed.

Keep:

- explicit config dataclasses,
- standard eval wiring,
- checkpoint/resume behavior,
- ordinary trainer settings and logging.

Avoid:

- new framework layers,
- reusable tokenizer registries,
- early generalization for future modalities.

The first baseline model should be small enough for quick iteration and shared across the three tokenizer families unless one family clearly needs a different sequence length.

### Step 9: Train Only The Coefficient Baseline First

Before forking bytes or symbols:

- get one coefficient-only run to train end to end,
- verify loss decreases,
- verify evaluation runs,
- verify a resumed run continues cleanly,
- verify the token store and launch config are deterministic enough to reproduce the run setup.

This step is the gate for the rest of the project. If coefficient tokens do not give a stable baseline, the rest of the ladder is much harder to interpret.

### Step 10: Fork Variants By Copying, Not Abstracting

Once `base/` works:

- copy `base/` to `bytes/`,
- copy `base/` to `symbols/`,
- keep the changes local to tokenization, vocab size, and sequence-length-related settings,
- avoid adding shared abstraction unless the same code is duplicated enough to be genuinely painful.

If a later model or train-loop change wins for all variants, upstream that specific change back into `base/`.

### Step 11: Add Focused Evaluation Helpers

In `experiments/jpeg_tokenizer/base/eval.py`, implement helpers for:

- teacher-forced loss,
- normalized likelihood-style comparisons,
- sequence length summaries,
- decode-validity rate,
- small perturbation experiments.

Keep the first evaluation stack directly tied to the research question. A metric belongs in v0 if it helps distinguish:

- prefix usefulness,
- hidden-state burden,
- rollout brittleness.

### Step 12: Build The Stability Harness

Add one script or evaluation entry point that:

- takes a prefix,
- perturbs one token or one local region,
- continues rollout,
- measures downstream degradation.

The important contrast cases are:

- local corruption in coefficient space,
- syntax corruption in symbol space,
- desynchronization in byte space.

The output should be tabular first. Fancy demos can come later.

### Step 13: Add Integration-Style Tests

Create tests that validate externally visible behavior rather than internals.

Minimum useful tests:

- canonical preprocessing is deterministic,
- tokenization is stable for a fixed input,
- token-store indexing returns the expected sequence,
- dataset length and batching are correct,
- decode-validity checks behave correctly on both valid and intentionally corrupted inputs.

Prefer tiny fixed fixtures over large sample-based tests.

### Step 14: Define The First Comparison Run Matrix

Before launching many runs, freeze one small matrix:

- `base/coeffs`
- `bytes/`
- `symbols/`

Keep matched as much as practical on:

- model family,
- optimizer,
- batch size,
- training budget,
- validation cadence.

Allow sequence length and vocab size to differ where the representation demands it, but document those differences explicitly in the run notes.

### Step 15: Only Then Revisit Infrastructure

After the first three variants have:

- trained,
- evaluated,
- and gone through perturbation analysis,

reassess whether you actually need:

- Levanter cache construction,
- WebDataset raw shards,
- more elaborate preprocessing orchestration,
- RVQ or codebook variants.

If the direct random-access path is working, keep it. Additional infrastructure needs to earn its place by clarifying results or materially reducing operational pain.

### Suggested Near-Term Execution Order

If starting implementation tomorrow, the order should be:

1. freeze V0 decisions in a short note,
2. create `experiments/jpeg_tokenizer/base/`,
3. implement canonical JPEG preprocessing,
4. write the Phase 0 inspection script,
5. lock the token encodings,
6. build the precomputed token store,
7. wire `DirectDatasetComponent`,
8. copy the grug model/train/launch stack,
9. get the coefficient baseline training,
10. fork bytes and symbols,
11. add perturbation and rollout evaluation,
12. decide whether any heavier infrastructure is justified.

### Practical File-Level Checklist

`experiments/jpeg_tokenizer/base/tokenizers.py`

- canonical JPEG conversion
- bytes tokenizer
- symbols tokenizer
- coefficient tokenizer
- representation-level comments

`experiments/jpeg_tokenizer/base/data.py`

- token-store reader
- dataset adapter
- `LmDataConfig` builder or helper

`experiments/jpeg_tokenizer/base/model.py`

- copied grug-ish decoder-only model config
- explicit `vocab_size` and `max_seq_len`

`experiments/jpeg_tokenizer/base/train.py`

- copied train loop
- eval hooks
- logging and checkpoint wiring

`experiments/jpeg_tokenizer/base/launch.py`

- one minimal `ExecutorStep`
- small default run config

`experiments/jpeg_tokenizer/base/eval.py`

- decode-validity checks
- perturbation helpers
- comparison metrics

`scripts/jpeg_tokenizer/inspect_representations.py`

- Phase 0 stats and decision support

`scripts/jpeg_tokenizer/build_token_store.py`

- deterministic precompute path

### Final Guardrail

When in doubt, prefer the change that makes the tokenizer comparison cleaner rather than the system more ambitious. The project wins by producing a crisp answer about autoregressive tokenization properties, not by accumulating image-modeling machinery.
