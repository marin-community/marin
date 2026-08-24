# Research: paired long-context likelihood evaluation

Effort: medium. The decision affects multiple 67B checkpoint evaluations, but the implementation can reuse an existing Marin experiment and evaluator.
Stop rule: stopped when repository history, benchmark construction, and current evaluator contracts converged on an MRCR-first unit.
As of: 2026-08-24.

## Question

How should Marin measure whether the pretrained 67B checkpoints use long context without confounding the result with post-training or output-format compliance?

## Current Marin context

- The 65K source and 262K qk checkpoints already exist. Standard Paloma likelihood does not measure distant-context use.
- [MRCR experiment #7181](https://github.com/marin-community/marin/issues/7181) scored the same final answer with and without retained conversation context on a d512 base model. Full context reduced PPL from 21.09 to 10.40 at 8K.
- [PR #7203](https://github.com/marin-community/marin/pull/7203) implemented the paired datasets and tagged evaluation. Review found no correctness issue, but the PR closed from inactivity without merging.
- The June 67B context-parallel implementation is not on main. This unit targets `origin/june_tpu_67b_a2b@db7ffddd339dd4db71fbb83ae2555abe3522c894`; a selective CP port is a separate prerequisite for landing the evaluator on main.

## Benchmark evidence

- The [Michelangelo paper](https://arxiv.org/abs/2409.12640) reports successful pretraining evaluation with MRCR and says few-shot prompting is critical for pretrained models.
- The [OpenAI MRCR dataset](https://huggingface.co/datasets/openai/mrcr) contains 2,400 examples: 100 examples for each combination of 2/4/8 needles and eight context bins from 4K–8K through 524K–1M. The prompt begins with worked examples, followed by a long conversation and a final retrieval query.
- Standard generation requires a ten-character prefix and up to roughly 512 tokens of copied output. A generation failure can therefore reflect formatting, copying, or instruction following in addition to context access.
- [RULER](https://arxiv.org/abs/2404.06654) has base-model precedent, but its 13 tasks span retrieval, multi-hop tracing, aggregation, and QA. They do not define one common paired-PPL contract. RULER generation remains useful after MRCR establishes the base-checkpoint likelihood path.

## Repository findings

- [`ChatLmDatasetFormat`](https://github.com/marin-community/marin/blob/53503016b332999d377c4f9b6e7a15ceca3b918a/lib/levanter/src/levanter/data/text/formats.py#L48-L70) can mask prompt tokens and score assistant tokens through a custom generation template. Its `pack=False` mode yields one padded example and rejects overlength examples instead of slicing them.
- [`TaggedEvaluator`](https://github.com/marin-community/marin/blob/53503016b332999d377c4f9b6e7a15ceca3b918a/lib/levanter/src/levanter/eval.py#L508-L640) already emits token-weighted loss and BPB for hierarchical dataset tags.
- The [June 67B evaluator builder](https://github.com/marin-community/marin/blob/53503016b332999d377c4f9b6e7a15ceca3b918a/experiments/june_tpu_67b_a2b/moe/train.py#L158-L198) provides the correct model loss function, but the training entrypoint restores optimizer state and assumes an optimization loop. A post-hoc checkpoint matrix should load model parameters and router state only.
- PR #7203 right-sliced overlength examples. Right slicing can remove the worked examples or the requested needle, so it is unsuitable for measuring effective context. Exact tokenizer-specific binning should happen before evaluation, and overlength examples should be excluded.
- PR #7203's `final_user_only` condition removed the prompt's worked examples. The no-context control must preserve the same worked examples and final query as the full-context condition.

## Decisions used in the draft

1. Scope the first unit to MRCR paired likelihood. RULER generation and RULER likelihood adapters are follow-up work.
2. Preserve the official two-shot preamble in both conditions. After the two-shot smoke passes, run one-shot and two-shot-no-prefix sensitivity matrices at 8K/32K for both source-qk packages and both final checkpoints.
3. Compare full context with a `query_only` control containing the preamble and final query but no target conversation.
4. Put the random prefix into the prompt continuation and mask it from loss. Score the identical response body in both conditions.
5. Assign examples to 8K/16K/32K/65K/131K/262K bins using the two-shot full-context length under the 67B tokenizer. Keep every prompt variant in the same canonical bin. `desired_msg_index` identifies the selected user request; measure token distance from the end of the following assistant response to the scored continuation. Do not truncate examples into a bin.
6. Run a standalone parameters-only checkpoint evaluation. Restore the checkpoint step and model parameters; do not load optimizer/router-pending state or write checkpoints.
7. Evaluate every saved +250/+500/+750/+1000 checkpoint from both qk arms as soon as it is available; do not wait for final-checkpoint separation.
8. Persist per-example paired losses and report token-micro, sample-macro, and deterministic paired-bootstrap confidence intervals. Use the saved rows for checkpoint-versus-source and qk-arm comparisons.
9. In the no-prefix variant, keep both official demonstrations but rewrite the final retrieval request to remove the nonce directive and do not condition the target on the nonce. Keep the same answer body and canonical source bin.

## Risks and unknowns

- The pinned dataset's first message is expected to contain only the task preamble and worked examples. The transform must assert this invariant rather than silently treating a target conversation turn as a demonstration.
- Token-micro loss weights longer answers more heavily. The first unit retains it as primary, adds sample-macro loss, and bootstraps paired source records so a few long answers cannot silently determine the conclusion.
- The full two-shot matrix reaches 60 jobs across six caps, including 20 jobs at 131K/262K. A source-versus-one-extended-checkpoint smoke should catch evaluator and compilation failures, but available intermediate checkpoints do not wait for scientific separation before evaluation.
- Context gain can be positive even when free generation is poor. That is the intended separation: this unit measures pretrained context use, while MRCR/RULER generation remains a later behavioral evaluation.

## Source ledger

| Source | Type | Claim used for | Confidence |
|---|---|---|---|
| [#7181](https://github.com/marin-community/marin/issues/7181) | Experiment issue | Paired MRCR likelihood produces base-model signal | High |
| [#7203](https://github.com/marin-community/marin/pull/7203) | Closed PR | Reusable transform/tagged-eval starting point | High |
| [OpenAI MRCR](https://huggingface.co/datasets/openai/mrcr) | Dataset | Dataset size, bins, prompts, prefix, needle counts | High |
| [Michelangelo](https://arxiv.org/abs/2409.12640) | Paper | MRCR works as a pretraining eval with few-shot prompting | High |
| [RULER](https://arxiv.org/abs/2404.06654) | Paper/code | Base-model precedent; heterogeneous task semantics | High |
| [#7638](https://github.com/marin-community/marin/issues/7638) | Issue | RULER generation plumbing remains open | High |

## Handoff

Implement the [design](design.md) as one MRCR dataset/evaluator PR plus a bounded checkpoint smoke. Treat RULER as a separate unit after the paired-likelihood result is established.
