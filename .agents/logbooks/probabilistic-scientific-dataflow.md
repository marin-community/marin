---
topic: probabilistic-scientific-dataflow
description: Synthetic proof of concept for a scientist-facing probabilistic dataflow DSL compiled to Marin transformer calls
author: jder
---

# Probabilistic Scientific Dataflow: Research Logbook

## Scope

- Goal: Test whether a scientist-facing probabilistic dataflow abstraction can express heterogeneous conditional queries while lowering to standard Marin transformer training and inference artifacts.
- Primary metrics: end-to-end compilation for two unrelated synthetic scientific programs; correct rejection or diagnosis of leakage and factorization violations; packed calls accepted by a Marin Grug transformer; training loss reduction on the mixed synthetic workload.
- Constraints: avoid dataset ingestion work; keep the spike under `experiments/`; preserve standard token embeddings, dense transformer layers, cross-entropy, and packed attention boundaries; do not claim production-ready probabilistic semantics.
- Coordinating issue/PR: none; public visibility was not implied by the request.
- Experiment prefix: `PSD`.

## Current TL;DR

`PSD-007` preserves one ordinary Grug parameter set while selecting position IDs, attention, and label alignment from each compiled call. Synthetic text uses rotary positions `0..S-1`, causal attention, and shifted labels; scientific records use zero rotary positions, full segmented attention, scientific-position embeddings, and aligned labels. An 80-step equal-weight CPU smoke reduced combined loss from 4.1909 to 0.2022; text accuracy rose from 0% to 75% and scientific accuracy from 0% to 100%. A scientific-record permutation changed restored logits by `5.96e-08`. `Query.given`, `Query.targets`, and `Query.environment` now directly describe a conditional query; the unused `Evidence`, `Environment`, and integer availability-time abstractions were removed. Held-out generalization, symbolic indexed availability, compositional position encoders, real text tokenization, and sampled refinement remain untested.

## Baseline

- Date: 2026-07-13
- Code ref: `a9f14597e6c32da7446e8d2b8edc0925537a7d61` (`origin/main`)
- Baseline numbers: no existing probabilistic scientific dataflow implementation or matching GitHub issue found.

## Hypothesis Queue

### Active

- `PSD-004`: Compositional field, axis, coordinate, topology, and relation embeddings may transfer better than the current learned embedding for each fully qualified scientific position. Next test: add a held-out coordinate or task split before changing the position encoder.
- `PSD-005`: Refinement training needs sampled proposals instead of truth-valued feedback. Next test: execute one proposal call and materialize its sampled result into the next call.

### Blocked

None.

### Falsified / Dead End

None.

### Promoted

- `PSD-001`: The dataflow graph retained axes, provenance, environment allowlists, split keys, random ancestry, and factor IDs through query compilation. Evidence: `tests/experiment/test_probabilistic_dataflow.py`, the 2026-07-13 completion entry, and the 2026-07-14 query simplification entry.
- `PSD-002`: Parallel, autoregressive, and three-step refinement plans lowered to explicit model-call dependencies. Evidence: compiler demo output and behavior tests.
- `PSD-003`: One unchanged Grug transformer learned packed calls from both synthetic program families with standard weighted cross-entropy. Evidence: 80-step smoke result below.
- `PSD-006` (exploratory): Grug calls with zero runtime rotary positions and full segmented attention are equivariant to serialization of complete scientific records. Evidence: the 2026-07-13 semantic-record entry and `test_scientific_record_logits_are_equivariant_to_serialization_order`.
- `PSD-007` (exploratory): One Grug parameter set can train causal, physically positioned text calls and unordered, scientifically positioned calls when execution data selects positions, masks, and target alignment. Evidence: the 2026-07-13 cross-domain entry and `test_one_grug_model_learns_causal_text_and_full_attention_science`.

## Background Research Brief

- Effort: low
- Stop rule: stop once local Marin model/packing paths and direct probabilistic/scientific precedents identify the minimum vertical slice.
- Date: 2026-07-13

### Question

What is the smallest implementation that demonstrates the proposed abstraction rather than rebuilding data infrastructure or a new transformer stack?

### Current Marin Context

- `experiments/grug/base/model.py` exposes a dense Llama-style `Transformer`, causal/segmented attention, and weighted next-token cross-entropy.
- `lib/levanter/src/levanter/data/packing.py` and Grug attention masks establish segment IDs as the existing packed-example boundary mechanism.
- `marin-experiments/tiny-stories` demonstrates a standalone small-model training loop; `speech-asr` demonstrates that non-text modalities can be discretized before using the same transformer machinery.

### External Prior Art

- Pyro's trace and condition handlers show the value of separating a probabilistic model from execution-relative observations, but this spike will use explicit graph objects because availability and deployment analysis require durable metadata.
- GraphCast's typed graph, autoregressive wrapper, and rollout separation support representing scientific structure and inference strategy outside the neural core.

### Negative / Failed Leads

- No matching implementation, report, issue, or PR was found in Marin for `probabilistic scientific dataflow` or `scientific transformer DSL`.
- Reusing the standalone `marin-experiments` training script wholesale would pull in dataset, checkpoint, dispatch, and tracking machinery that is deliberately outside this spike.

### Recommended Next Experiments

#### 1. Four-IR compilation trace

- Minimum experiment: compile synthetic field forecasting and unordered-pair prediction through dataflow, conditional-query, plan, and execution IRs.
- Baseline/control: direct handwritten token sequences.
- Expected signal: both programs use the same compiler and differ only in axes, graph, evidence, and plan.
- Falsifier: scientific semantics must be manually encoded in the training loop or virtual-document layout.
- Cost/risk: low; semantics will be intentionally incomplete.

#### 2. Shared-model mixed training

- Minimum experiment: pack calls from both programs into segmented sequences and train a tiny Grug model on CPU.
- Baseline/control: initial untrained loss.
- Expected signal: final loss is lower and both task families contribute supervised tokens.
- Falsifier: custom model layers or losses are required to consume compiler output.
- Cost/risk: low; memorization is sufficient because the test is infrastructure compatibility, not scientific generalization.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Marin Grug base model | Marin code | `experiments/grug/base/model.py` | Existing dense model and weighted CE boundary | high | Current main |
| Levanter packing | Marin code | `lib/levanter/src/levanter/data/packing.py` | Segment IDs preserve packed boundaries | high | Current main |
| Marin experiments | external code | `marin-community/marin-experiments` | Minimal standalone model/training patterns | high | Pulled 2026-07-13 |
| Pyro Poutine | official docs | https://docs.pyro.ai/en/stable/poutine.html | Model/evidence separation precedent | medium | Conceptual only |
| GraphCast | external code | https://github.com/google-deepmind/graphcast | Typed scientific graph and rollout precedent | medium | Conceptual only |

## Background Research Brief: Semantic Record Order

- Effort: low
- Stop rule: stop when local attention interfaces and primary set-model references determine a falsifiable implementation.
- Date: 2026-07-13

### Question

Can the execution IR treat serialization as a packing choice while retaining scientifically meaningful identity, factorization, and standard transformer training?

### Current Marin Context

- Grug base attention applies RoPE from dense row position to every query and key. Its segment mask blocks cross-document attention but does not reset or remove RoPE.
- Grug's attention API already accepts noncausal masks. The MoE implementation also has a no-RoPE execution path, so disabling RoPE is compatible with the existing attention implementation.
- The first spike puts each descriptor in the ordinary token vocabulary and trains by shifting a realized target value after its descriptor. Full attention would leak that realized value, so order-independent calls require aligned labels that are absent from the input.

### External Prior Art

- Deep Sets characterizes permutation-invariant and permutation-equivariant set functions. This supports testing a compiled document as a collection of records rather than assigning meaning to its sequence order.
- Set Transformer builds set models from attention without sequence-order semantics. This supports using dense bidirectional attention for a learned joint factor when the factor itself has no causal order.
- Graphormer adds graph-derived structural encodings instead of relying on serialization position. This is the relevant pattern for future mesh topology or pair-relation biases.

### Evidence Map

#### Claim: no positional encoding plus full self-attention is equivariant to record permutation

- Support:
  - Set Transformer: attention-based set processing is designed to ignore input order.
  - Deep Sets: permutation equivariance is the necessary intermediate property for per-element outputs.
- Contradictions:
  - Scientific identity must move with each record. Permuting token content without its semantic position changes the scientific input.
  - A target value included as an input leaks under full attention; aligned labels must remain outside the model input.
- Directness to Marin: high; Grug accepts an explicit dense attention mask and exposes the token embedding and transformer blocks.
- Confidence: exploratory until numerical permutation parity and training checks pass.
- Action: replace descriptor/value pairs with single records carrying token content, semantic position, and an aligned optional target.

### Recommended Next Experiments

#### 1. Record permutation equivariance

- Minimum experiment: compile one advection call, permute its records, run both orders through one initialized no-RoPE Grug model, undo the output permutation, and compare logits.
- Baseline/control: the current RoPE document path.
- Expected signal: maximum absolute logit difference below `1e-5` for the semantic-record path; the RoPE baseline should differ.
- Falsifier: unpermuted logits differ beyond numerical tolerance.
- Cost/risk: low CPU test; a passing test proves serialization equivariance for the implemented mask, not scientific symmetry under relabeling axes.
- Sources: Set Transformer and Deep Sets.

#### 2. Mixed training with aligned labels

- Minimum experiment: train the existing two-layer Grug smoke model on advection and contacts using full segment attention, scientific position embeddings, and target-aligned cross-entropy.
- Baseline/control: initial loss and accuracy from the same initialized model.
- Expected signal: loss decreases for both synthetic task families without RoPE or causal masking.
- Falsifier: the workload does not learn or requires reintroducing physical positions.
- Cost/risk: low; memorization still does not measure held-out generalization.
- Sources: current Grug training path.

### Hypothesis Queue Update

- Add: `PSD-006`, serialization-order equivariance with aligned scientific records.
- Revise: `PSD-004` now tests structured scientific position embeddings rather than descriptor tokens in the ordinary vocabulary.
- Falsify / stop: none.
- Promote: none before numerical evidence.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Grug base attention | Marin code | `experiments/grug/base/model.py` | RoPE and explicit-mask behavior | high | Current branch base |
| Grug MoE attention | Marin code | `experiments/grug/moe/model.py` | Existing no-RoPE precedent | high | Current branch base |
| Deep Sets | paper | https://arxiv.org/abs/1703.06114 | Permutation invariance and equivariance | high | NeurIPS 2017 |
| Set Transformer | paper | https://arxiv.org/abs/1810.00825 | Attention over unordered inputs | high | ICML 2019 |
| Graphormer | paper | https://arxiv.org/abs/2106.05234 | Structural rather than serialization encodings | medium | Future relation-bias direction |

## Entry Log

### 2026-07-13 18:31 EDT - PSD-001 kickoff

- Hypothesis: the proposal can be tested with compiler-first synthetic programs and the existing Grug model, without implementing dataset connectors.
- Commit Hash: `a9f14597e6c32da7446e8d2b8edc0925537a7d61`
- Command: `git pull --ff-only origin main` in Marin and Marin-experiments; targeted `rg`/source inspection of Grug model, training, packing, and attention paths.
- Config: local CPU spike; no external datasets; no cluster jobs.
- Result: no existing implementation or issue found. The existing Grug model accepts token IDs, weighted CE, and segmented causal masks, which is sufficient for a vertical slice.
- Interpretation: implement the compiler under `experiments/probabilistic_dataflow` and keep dataset/deployment integration out of scope.
- Next action: implement the DSL and four IR levels, then add synthetic programs and a mixed-model smoke run.

### 2026-07-13 18:46 EDT - PSD-001 vertical slice

- Hypothesis: two unrelated scientific programs can compile into standard segmented next-token examples and train through the existing Marin Grug model.
- Commit Hash: working tree based on `a9f14597e6c32da7446e8d2b8edc0925537a7d61`
- Command: `uv run python -m experiments.probabilistic_dataflow.demo --training-steps 80`
- Config: seed 0; 8 examples each of three-step synthetic advection and contacts; sequence length 64; 104 packed rows; 144 supervised tokens; 88-token vocabulary; 2-layer Grug transformer with hidden size 48, intermediate size 96, 4 query heads, and 2 KV heads; Adam at `3e-3`; CPU.
- Result: training loss decreased from 4.536209 to 0.029560; training-token accuracy increased from 0.0208 to 1.0. The demo also produced a two-call factor-preserving structure plan, a three-call refinement plan, a factorization rejection, and an indirect future-leakage rejection with source provenance.
- Interpretation: the DSL/compiler can drive the existing dense transformer, segmented attention, and weighted cross-entropy path. The metric demonstrates compatibility and memorization only; held-out generalization was not measured. Parallel field queries explicitly record their product-of-token-marginals approximation.
- Next action: decide whether the next spike should test held-out cross-program transfer, execute sampled refinement feedback, or promote the IR into a library package.

### 2026-07-13 20:15 EDT - PSD-001 debug renderings

- Hypothesis: a single deterministic renderer can make each compiler boundary and the generated transformer documents inspectable without introducing a second serialization format.
- Commit Hash: working tree based on `a9f14597e6c32da7446e8d2b8edc0925537a7d61`
- Command: `uv run python -m experiments.probabilistic_dataflow.debug_render`; `uv run python -m experiments.probabilistic_dataflow.debug_render --check`.
- Config: synthetic advection, unordered-residue contacts, factorized contacts-to-distances structure, and a mixed packed batch with sequence length 48.
- Result: generated Markdown reports render the probabilistic dataflow, conditional query, inference plan, and transformer execution IRs. Document tables distinguish controls, context, targets, teacher-forced values, shifted next-token targets, and loss weights. The mixed report identifies document spans, padding, and cross-domain segment boundaries.
- Interpretation: the generated execution artifact is readable at both semantic-record and dense-packed-row levels. Refinement reports explicitly mark truth-valued feedback as a spike limitation rather than presenting it as sampled inference.
- Next action: use `--check` when changing the compiler or examples so reviewed reports cannot silently drift from executable IR objects.

### 2026-07-13 20:36 EDT - PSD-006 semantic records

- Hypothesis: removing physical position encodings, keeping target values out of model inputs, and using full attention within learned-joint calls will make document serialization irrelevant to model outputs.
- Commit Hash: working tree based on `a9f14597e6c32da7446e8d2b8edc0925537a7d61`
- Command: `uv run python -m experiments.probabilistic_dataflow.demo --training-steps 0`; `uv run python -m experiments.probabilistic_dataflow.demo --training-steps 80`; `uv run --with pytest --with pytest-timeout pytest -q tests/experiment/test_probabilistic_dataflow.py -m "not tpu_ci and not integration and not data_integration and not requires_cluster"`; `uv run --with pytest --with pytest-timeout pytest -q tests/test_grug_variant_contracts.py -k base`.
- Config: seed 0; one record per scientific value instance; content-token embedding plus learned fully qualified scientific-position embedding; no RoPE; full attention within packed segments; target-aligned weighted cross-entropy; two-layer Grug transformer with hidden size 48 for training and hidden size 16 for the equivariance probe.
- Result: a random permutation of all 28 records in one advection call produced a maximum restored-logit difference of `4.470348358154297e-08`. The 80-step mixed run packed 144 targets into eight length-64 rows, reduced loss from 3.725789 to 0.037660, and increased training accuracy from 0.0347 to 0.9931. The scientific-dataflow suite passed 11 tests; the Grug base contract subset passed 3 tests.
- Interpretation: serialization order is an execution optimization for these full-attention calls. Scientific identity travels with each record, and outputs are permutation-equivariant at numerical tolerance. This does not prove invariance to relabeling set identities, and the per-coordinate learned embeddings do not yet demonstrate transfer to unseen coordinates.
- Next action: test compositional position encoders on held-out coordinates, or add an explicit semantic attention-edge IR when a synthetic factor requires within-call causality.

### 2026-07-13 21:59 EDT - PSD-007 shared text and science model

- Hypothesis: cross-domain weight sharing does not require making scientific serialization semantic; one standard Grug model can select rotary positions, attention, and target alignment from each compiled call.
- Commit Hash: working tree based on `a9f14597e6c32da7446e8d2b8edc0925537a7d61`
- Command: `uv run python -m experiments.probabilistic_dataflow.demo --training-steps 80`; `uv run --with pytest --with pytest-timeout pytest -q tests/experiment/test_probabilistic_dataflow.py -m "not tpu_ci and not integration and not data_integration and not requires_cluster" -k "not checked_in_debug_outputs_are_current"`; `uv run --with pytest --with pytest-xdist pytest -q lib/levanter/tests/grug/test_attention.py -k runtime_positions`.
- Config: seed 0; one shared 64-token vocabulary, embedding table, two-layer hidden-size-48 transformer, and output projection; equal text/science task weighting; text calls use positions `0..6`, causal masks, and shifted labels; advection calls use zero rotary positions, scientific-position embeddings, full masks, and aligned labels.
- Result: combined loss decreased from 4.190883 to 0.202209. Text loss decreased from 4.161404 to 0.366694 and accuracy increased from 0 to 0.75 over 48 supervised tokens. Scientific loss decreased from 4.220362 to 0.037724 and accuracy increased from 0 to 1.0 over 96 supervised tokens. The scientific-only advection/contact smoke reduced loss from 4.226475 to 0.033981 and reached 1.0 accuracy. Record-permutation error was `5.960464477539063e-08`. The focused scientific-dataflow suite passed 12 tests and the runtime-position attention test passed.
- Interpretation: positional treatment can be data-dependent without splitting the model. Ordinary text retains its standard causal/RoPE path, while scientific calls make physical order irrelevant by passing zero positions and full masks. The result proves API and optimization compatibility on memorized synthetic workloads; it does not demonstrate language quality, cross-task transfer, or a tokenizer shared with real text corpora.
- Next action: test the same boundary with a real tokenizer and held-out text/scientific examples before changing the model architecture further.

### 2026-07-13 22:07 EDT - PSD-007 final local validation

- Hypothesis: runtime position selection preserves the existing Grug base behavior while keeping generated IR reports deterministic.
- Commit Hash: working tree based on `a9f14597e6c32da7446e8d2b8edc0925537a7d61`
- Command: `uv run --with pytest --with pytest-timeout pytest -q tests/experiment/test_probabilistic_dataflow.py -m "not tpu_ci and not integration and not data_integration and not requires_cluster"`; `uv run --with pytest --with pytest-xdist pytest -q lib/levanter/tests/grug/test_attention.py -k runtime_positions`; `uv run --with pytest --with pytest-timeout pytest -q tests/test_grug_variant_contracts.py -k base`; `uv run python -m experiments.probabilistic_dataflow.debug_render --check`; `./infra/pre-commit.py --fix <changed files>`.
- Config: local CPU; deterministic checked-in debug outputs; no cluster or external dataset access.
- Result: the dataflow suite passed 13 tests, the runtime-position attention test passed, the existing Grug base contract subset passed 3 tests, the generated reports matched, and Marin's required pre-commit checks passed.
- Interpretation: the new optional position-ID path defaults to existing physical positions and does not require call-site changes. The explicit zero-position scientific path is covered separately from the default text-compatible path.
- Next action: none for this spike.

### 2026-07-14 11:28 EDT - Query and environment simplification

- Hypothesis: the prototype's `Evidence`, `Environment`, and integer availability times add ceremony without demonstrating example-indexed availability.
- Commit Hash: working tree based on `a9f14597e6c32da7446e8d2b8edc0925537a7d61`
- Command: targeted source inspection with `rg`; focused scientific-dataflow tests; debug report regeneration.
- Config: all synthetic programs now construct `Query(program, given=..., targets=..., environment=...)`; `Source` retains provenance, environment allowlists, and split keys; `FlowInfo` propagates those fields without a time.
- Result: `Environment`, `Evidence`, `EvidenceBinding`, `Source.available_at`, and `FlowInfo.available_at` were removed. The indirect leakage example now rejects a training-only normalization supplied to a deployment query. Time-indexed future leakage is explicitly out of scope until availability can be expressed symbolically over named example indices.
- Interpretation: the reduced query surface matches what the compiler actually uses. A future availability design should bind symbolic index expressions when selecting a concrete example instead of comparing hard-coded integers before example selection.
- Next action: keep time availability out of the tutorial and DSL until the symbolic indexed design is implemented.
