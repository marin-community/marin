# A three-curve test of epoch matching at total-parameter TPP 10

Test whether matching StarCoder repetition improves mixture selection when proxy and target use the same tokens per **total trainable parameter**. At each StarCoder fraction, train a target, a nonrepeating proxy, and a proxy with target-matched repetition. Let p be StarCoder's training-token fraction; the web domain receives 1−p. The [earlier fixed-model experiment](../starcoder_epoch_matching/refinement/results/RESULTS.md) had total TPP 1.32 versus 35.27 and produced worse selection from the matched proxy. This experiment changes model sizes, tokenizer, parent corpus, repetition regime and subset sampling as well as TPP. It tests the three arms within the new setting; an outcome change across experiments cannot be attributed to TPP alone.

## Background

The existing launcher supplies the training assembler, regional placement, and durable-success checks. New data must use a smaller vocabulary: the existing Llama3 caches would make embeddings dominate the smallest model. [Research notes](research.md) identify the pinned tokenizer, regional raw sources, and reusable cache APIs. No historical loss measurements are reused.

## Design

Use randomly initialized Qwen3 architectures with tied embeddings, MLP width four times hidden width, head width 128, sequence length 2,048, and TinyLlama's pinned 32,000-token tokenizer. Count the complete abstract model, including Q/K normalization weights, rather than relying on Qwen3's inherited parameter-count helper.

| | Unmatched proxy | Matched proxy | Target |
|---|---:|---:|---:|
| Width / layers | 256 / 8 | 256 / 8 | 1,024 / 16 |
| Total parameters | 16,587,008 | 16,587,008 | 301,241,344 |
| Nonembedding parameters | 8,395,008 | 8,395,008 | 268,473,344 |
| Training tokens | 165,937,152 | 165,937,152 | 3,012,296,704 |
| Total TPP | 10.0040 | 10.0040 | 9.9996 |
| Nonembedding TPP | 19.7662 | 19.7662 | 11.2201 |
| Batch / optimizer steps | 32 / 2,532 | 32 / 2,532 | 128 / 11,491 |
| Available StarCoder tokens | 190,316,544 | 10,485,760 | 190,316,544 |
| Maximum nominal epochs | 0.8719 | 15.8250 | 15.8278 |
| Training FLOPs per run | 2.491e16 | 2.491e16 | 6.664e18 |

The two proxies share their architecture, horizon, optimizer, web pools, training seeds and mixture sampler. StarCoder support is the treatment. The target/proxy TPP discrepancy is 0.0443%; the nominal matched-epoch discrepancy is 0.0178%. The real allocator audit finds at most 0.0504% epoch discrepancy across the grid. Matching total TPP and epochs also approximately matches available StarCoder tokens per total parameter (0.6322 matched proxy versus 0.6318 target), since P/N = (D/N)/(D/P). This is a control, not an explanation of the prior result. The parent size was chosen just above the nonrepeating-proxy requirement to create substantial target repetition at moderate model sizes.

Construct one finite StarCoder parent from nearly equal token quotas from all 49 archived Dolma StarCoder shards. Materialize a fixed global permutation of its 92,928 packed sequences before the training-time block shuffle, so prefixes do not follow raw-shard order. Construct three uniform samples without replacement of 5,120 packed sequences from its 92,928 sequences, using independent, prespecified subset seeds. The index audit attributes sequences to their midpoint shard and does not establish content representativeness. These samples may overlap each other; each contains no repeated source sequence. Materialize their caches before training. Hold membership fixed across mixture fractions and trainer seeds. These are conditional-subset replicates within one finite parent, not independent parent-corpus replicates.

The other domain retains the six historical Nemotron quality components. Its inherited Llama3-based weights are held fixed as training-token proportions under the new tokenizer; they do not preserve identical document proportions. Select 16 source shards per component by a seeded hash over the complete regional filename inventory. Allocate a finite web pool with 20% headroom above the target's maximum allocation, plus an allocator block. All arms use these same web caches. The sampler audit must establish that no web component wraps and that the unmatched proxy never wraps StarCoder, including at p=1. Reads and writes stay in us-central1. Tokenization stops at explicit token quotas and resumes completed shards; it never invokes the whole-corpus dataset builders.

Use the historical MuonH recipe (Muon LR 0.02, Adam LR 0.008, momentum 0.95, betas 0.9/0.98), batch 32 for proxies and 128 for the target, 1% warmup and cosine decay during the final 20% of steps. Within each scale, the recipe is fixed across all mixtures. This optimizer has not been validated at the new tiny geometry; the pilot tests that feasibility. Before any target run, compare proxy batches 32 and 128 at fixed p=0.5, on both arms and both trainer seeds (one matched subset). The eight-run screen adds four artifacts beyond the primary grid. If mean unmatched-proxy BPB at batch 32 exceeds batch 128 by more than 0.01, or a run is nonfinite, stop for a newly reviewed recipe. Matched losses are diagnostics and do not select the recipe. Review loss/gradient traces for instability before release. Embeddings and vector parameters use Adam; linear weight matrices use MuonH. There is no separate decoupled weight-decay transform; clipping is 1.0 per optimizer group and the LR floor is zero.

The pilot grid is p = 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1. The dense grid is 0 to 1 in steps of 0.05 and includes the pilot. Both proxy arms use two trainer seeds; the matched arm crosses them with all three subset draws. At p=0, each matched draw aliases the corresponding unmatched run. Stable component names and explicit shuffle keys make the surviving web stream independent of subset identity and zero-weight component removal; [the native dataset audit](zero_weight_stream_audit.json) verifies this on local synthetic caches. Mixture interleaving changes with p; each source's underlying sequence order remains fixed. The target uses one trainer seed. Pilot: 57 distinct runs, about 4.79e19 training FLOPs. Dense: 183 cumulative runs, about 1.44e20 FLOPs. The four extra batch-screen runs add about 9.96e16 FLOPs. The three-run canary completes p=1 for the target and one seed of each proxy, reusing those artifacts in later stages. These estimates exclude tokenization, evaluation, compilation and hardware inefficiency; chip-hours require canary timing.

## Outcomes and release rule

The primary loss is native held-out Paloma programming-languages BPB. For each matched subset, average its two trainer-seed losses at each p and select the observed grid minimum. Target regret is target BPB at the proxy-selected p minus the lowest measured target BPB on the common grid. Compare with the unmatched proxy using the same two-seed averaging rule. Exact ties select the smaller p. Both selection and target scoring use the same held-out metric. Report all three paired regret differences and their descriptive mean, plus all six individual matched seed selections. The differences share a target curve and unmatched selection; they are not independent effect estimates. A mixture chosen from the curve averaged over all subsets is secondary: it has extra information. No smooth fit chooses a mixture.

Show raw BPB and loss above each curve's own observed minimum. Report Spearman correlation and RMSE of the excess-loss curves on the common grid as secondary agreement measures. Target regret is conditional on one measured target curve; there is no target-seed confidence interval or significance claim. Target confirmation at selected fractions requires a separately reviewed release.

The pilot checks finite endpoint metrics, optimization stability, data identity, actual repetition and TPP. There is no signal-size or interior-optimum promotion gate. Dense expansion requires all pilot artifacts and a recorded human release. The sign of the matched-minus-unmatched regret is **not** a promotion gate. Flat or unfavorable curves remain reportable; a revised setting receives a new manifest rather than overwriting this one.

## Costs / risks

- A 16.6M model at TPP 10 may underfit the primary evaluation or show optimizer sensitivity. Matching total TPP does not match nonembedding TPP or guarantee comparable capability.
- Three 10.5M subsets measure only some of the composition uncertainty. A single finite parent and one target trainer seed limit generalization.
- The desired target turnover is an empirical hypothesis. An interior minimum or better matched-proxy selection is not assumed by the analysis.
- Random shard selection followed by within-shard prefixes defines a finite conditional benchmark. It does not establish a uniform document sample from the original corpora.

## Testing

Verify model parameter counts against the abstract model tree; run the real mixture allocator and source-index shuffles; test bounded tokenization, exact cache lengths and subset nesting on local synthetic text; test failed/stale output rejection, stage identity reuse, final-step collection, and adversarial selection examples. Freeze data-source generations and tokenizer/code hashes. Before TPU release, require regional source/cache receipts and a child-runtime canary. Local checks cannot substitute for that canary.

## Decisions from review

The fixed-mixture batch screen precedes the target canary. It is a two-point feasibility check, not an optimizer search. The target has 4.54 times as many optimizer updates as the proxy despite matched total TPP. The single target seed supports an illustrative, conditional comparison; any selected-point confirmation requires a separate release. Crossed trainer and subset seeds expose two sources of proxy variation, while one finite parent limits the inference.
