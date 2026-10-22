# Hero checkpoint averaging and format SFT

The current procedure is [Hero checkpoint preparation and format SFT](../../docs/references/hero-format-sft.md). [Experiment issue #10428](https://github.com/marin-community/marin/issues/10428) contains the baseline, decision log, failures and measured results. Exact private checkpoint, execution and storage identities remain in the experiment evidence.

## Measured status: 2026-10-22

- The LR-aware average of steps 282000, 288000 and 294000 reduced loss on the identical fixed C4 panel from 2.130703034 to 2.109083774, a relative reduction of 1.01465%. All 59 sequences had lower loss. This does not establish the quality benefits of a retrained cooldown.
- Separate output-token donor initialization completed with verified same-region provenance, permanent metadata and finite array norms. Input embeddings were unchanged at initialization; the latest routing-control bias was retained.
- The LoRA plus five-vocabulary-delta campaign completed 64 total updates. The longer segment restored step 8 with full compact Adam state, counters and deterministic data position. Extending the finite linear schedule from 8 to 64 raised the first resumed learning rate relative to the short cooldown tail; optimizer and schedule counts were not reset.
- Training-source teacher-forced EOT mean NLL improved from 3.3814 to 0.4288 between steps 8 and 64; top-1 matches improved from 1/16 to 14/16. These are boundary-token diagnostics, not average sequence NLL or held-out quality.
- The 100-rollout sanity check completed: 100 EOT terminations, zero cap hits at 2,048 tokens, maximum length 496. Manual review found 33 correct final math answers among 40 cases, 32/40 code implementations plausible by static inspection, and 18/20 clear fixed-history passes. Code was not executed. One math reference is ambiguous; content and token-formatting defects are retained in the issue.
- The merged ordinary BF16 checkpoint is a provisional input for intended-backend RL capacity measurement. The native full-prefix sampler does not measure serving throughput. Full held-out/native format/content evaluation, independent merge and serving parity, and long-context behavior remain unresolved.

## Reproduction and scope

The implementation covers streamed LR-aware averaging, donor preparation, shared-template data preparation, compact LoRA/vocabulary training, full-state continuation, ordinary-weight export and native evaluation. The optional YaRN and serving-verification paths are implemented separately; this campaign did not qualify them.

Preserve all earlier failed attempts and reports. Recover training from completed compact checkpoints with full optimizer/counter/data state. Defer acquisition model executions until each permitted gradient sequence finishes, retaining unfinished diagnostic queues across restarts. Never rerun gradients solely to regenerate diagnostics.

Production Hero, cluster configuration, serving pins and releases are outside this experiment. The separate [vLLM draft PR #97](https://github.com/marin-community/vllm/pull/97) remains independent. No larger pilot, additional training or production adoption follows from the provisional capacity-planning handoff.
