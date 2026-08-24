# Paired MRCR likelihood evaluation for base checkpoints

The 67B context-extension runs need a measurement of distant-context use without post-training. Score the same MRCR response under full context and a no-context control, then compare target-only negative log-likelihood at 8K, 16K, 32K, 65K, 131K, and 262K.

The first unit covers MRCR only. RULER generation remains complementary; its 13 task families need separate likelihood adapters. See [research.md](research.md) for evidence and prior work.

## Challenges

MRCR's standard score depends on instruction following, an exact random prefix, and copying a long response. A pretrained model may use the context while failing one of those behaviors. The paired likelihood metric holds the target, task demonstrations, and final query fixed while removing the intervening conversation from the control. The control is necessarily shorter, so target position is not held fixed.

PR #7203 right-sliced overlength prompts, which can discard demonstrations or the requested response. Marin must bin complete examples with the 67B tokenizer. Because a 200K prompt can request a nearby response, the transform must also measure evidence distance.

The existing Grug evaluator is embedded in training. Post-hoc evaluation should restore only the checkpoint step and model parameters, not construct optimizer state through a zero-step training job.

## Costs / Risks

- Each 262K checkpoint evaluation processes up to 300 full-context examples across the 2/4/8-needle subsets. The primary matrix contains 60 jobs; the 8K/32K prompt-sensitivity variants add 16.
- Token-micro NLL weights long answers more heavily. Positive gain establishes higher probability on the distant answer, not reliable free generation.
- Evaluation above 65K runs against `origin/june_tpu_67b_a2b@db7ffddd339dd4db71fbb83ae2555abe3522c894`, the branch that produced the checkpoints. Porting or landing the CP implementation is a separate dependency.

## Design

### Paired MRCR datasets

Revive the useful parts of [PR #7203](https://github.com/marin-community/marin/pull/7203) in `experiments/datasets/mrcr.py`, pinned to OpenAI MRCR revision `f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d`.

Each source row produces two examples with identical scored tokens:

- `full_context`: official worked-example preamble, complete target conversation, and final query.
- `query_only`: the same worked-example preamble and final query, with the intervening target conversation removed.

Both conditions retain the two official worked examples. A one-shot variant retains the first complete example. A no-prefix variant keeps two shots but rewrites the final request to omit the nonce directive and does not supply the nonce as an assistant prefix.

In the primary and one-shot variants, the ten-character random prefix is supplied as the beginning of the assistant continuation. The scored target is always `answer.removeprefix(random_string_to_prepend)`. The no-prefix variant removes the directive and prefix entirely. This yields a paired sensitivity test for the benchmark's formatting confound.

The transform and cache builder use the same `ChatLmDatasetFormat` preprocessor. Assign each row to the smallest cap containing its complete two-shot full-context prompt and target, then keep both conditions and prompt variants in that canonical bin. This holds source sets fixed across the shot ablation. Use one document per sequence; exclude examples above 262,144 and never slice.

The dataset's `desired_msg_index` identifies the selected user request; the following message is its assistant response. In the fully rendered two-shot example, count tokens between the end of that response and the first scored answer token. Split into evidence-distance bands `≤32K`, `32K–65K`, `65K–131K`, and `>131K`. Long-context claims use the corresponding distance band, a preregistered lower-confidence-bound rule, and a minimum sample count.

Dataset tags identify the context cap, needle count, and condition. The evaluator's existing hierarchical loss output then provides each cell's token-weighted loss and BPB.

### Metrics

For each context cap, needle count, qk setting, and checkpoint, report:

`context_gain_nll = loss(query_only) - loss(full_context)`

`context_ppl_ratio = exp(context_gain_nll)`

A positive gain means the conversation increased probability on the gold response. Log raw losses, BPB, token counts, and example counts beside it.

Token-micro loss is primary; per-needle and distance results remain required. Persist paired loss sums per source and report sample-macro NLL plus deterministic paired-bootstrap 95% intervals. A matrix summary reports checkpoint-versus-source adaptation, deployable qk-arm differences, a difference-in-differences that removes the source inference-qk effect, and paired prompt sensitivity.

Preprocessing publishes example and scored-token counts, and evaluation uses no batch cap. The experiment issue preregisters the gain and regression tolerances used to choose a checkpoint.

### Checkpoint evaluation runner

Add a parameters-only runner beside the June 67B training code. Dispatch owns resources; the local evaluator restores `step` and `params`, evaluates once, logs to W&B, and atomically writes results. It builds no training data, optimizer, train state, or checkpointer and does not apply `pending_qb_betas`.

The launcher validates the model against the canonical 67B configuration. Only `max_seq_len`, `qk_mult`, and the attention implementation may differ. It records the normalized model-config fingerprint and implementation commit so a shape-compatible static-field change cannot pass silently.

One job evaluates one checkpoint at one context cap. This keeps compilation and memory failures attributable to a specific length and allows the 8K/32K smoke to finish before allocating 131K/262K jobs.

### Execution matrix

The engineering smoke evaluates the step-156,000 source and the completed qk-1.75 checkpoint at 8K and 32K. Continue only if every cell has paired counts, finite losses, nonzero target tokens, and reproducible bootstrap output.

The scientific matrix evaluates every saved checkpoint as it becomes available. It contains source@qk-1.57 and source@qk-1.75 plus the +250/+500/+750/+1000 checkpoints from both extension arms under their training qk. Every package runs at all six lengths and all three needle counts. Partial summaries remain valid while later checkpoints are missing.

After the official two-shot smoke passes, run one-shot and two-shot-no-prefix variants at 8K and 32K for the two source-qk packages and two final extension checkpoints. Compare each variant against the primary prompt on identical source IDs. This tests prompt dependence without multiplying the expensive 65K–262K cells.

## Testing

Add a transform test with two demonstrations, distractor turns, two target needles, and a final query. Verify paired conditions, deterministic one-shot extraction, exact no-prefix query rewriting, prefix removal, identical target bodies, and canonical source bins across variants.

Add tokenizer-boundary tests around every context cap using the production format preprocessor. A pair must land in exactly one bin; neither condition may be sliced; both conditions must have identical target IDs and masks. Include an example above 262K and verify that it is counted as excluded. Add examples with near and distant selected responses to verify evidence-distance bands.

Add a derived-metric test from fixed per-example loss sums. It should verify token-micro and sample-macro calculations, paired resampling, fixed-seed reproducibility, and percentile bounds. A small CPU evaluator test should verify that prompt, prefix, and EOS tokens receive zero loss weight while every response-body token receives weight one.

Run an 8K d512 or 67B smoke on TPU before the matrix. Software acceptance requires complete paired cells and finite metrics. The sign or magnitude of context gain is an experiment result, not a test assertion.

## Open Questions

- What confidence-interval and regression thresholds should the experiment issue preregister for choosing between qk arms?
- How many examples must an evidence-distance slice contain before it can support a beyond-65K or beyond-131K claim?
