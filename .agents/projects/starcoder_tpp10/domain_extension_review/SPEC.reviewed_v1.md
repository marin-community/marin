# Wikipedia and FineMath-3+ repetition sweeps at TPP10

Draft for scientific review, 11 September 2026. This extends the measured StarCoder experiment with two focus domains. No new training or data preparation is released by this document. The domain-specific builders and submission manifest still need implementation and validation.

## Question and scope

Measure whether different focus domains prefer different repetition levels when optimizing the same held-out loss, with model size, total token budget, tokenizer, finite focus-pool size, background mixture, and training recipe held fixed. Each run mixes one focus domain with the same six-component web background. There are three separate two-domain sweeps, not one mixture containing Wikipedia, FineMath, and StarCoder simultaneously.

The first stage trains only matched proxies. Target and unmatched-proxy arms are specified now so a later expansion uses the same finite parents and controls. "Target without simulated epoching" means using the finite parent directly; natural repetition is allowed. The optional unmatched proxy must never repeat a source sequence, including at p=1.

The narrow result is a set of conditional mixture-response curves and observed grid minima. Distinct optima would show that a common preferred epoch count is inadequate, but do not by themselves show that every shared upper cap is harmful: a sufficiently high cap permits all optima. A later comparison of cap-constrained proposals requires a separately specified proposer and selection protocol. The review should assess whether this narrow result usefully supports the paper's intended motivation for adaptive treatment of repetition.

## Fixed model and exposure settings

These values come from the existing frozen `starcoder_tpp10_assets/design.json` and `starcoder_tpp10.py`, not a new architecture search. N counts all trainable parameters, including tied embeddings once and Q/K normalization weights. D includes focus and web tokens together.

| Setting | N | D | D/N | Batch | Steps | Focus pool | Maximum nominal epochs | Training FLOPs per point |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Matched proxy | 16,587,008 | 165,937,152 | 10.00404 | 32 | 2,532 | 10,485,760 | 15.82500 | 2.49088604e16 |
| Target | 301,241,344 | 3,012,296,704 | 9.99961 | 128 | 11,491 | 190,316,544 | 15.82782 | 6.66390283e18 |
| Optional unmatched proxy | 16,587,008 | 165,937,152 | 10.00404 | 32 | 2,532 | 190,316,544 | 0.87190 | 2.49088604e16 |

The randomly initialized Qwen3 models use width/layers 256/8 and 1024/16, MLP expansion 4, tied embeddings, head width 128, sequence length 2048, and the same frozen TinyLlama 32,000-token tokenizer (revision 59f6f375b26bde864a6ca194a9a3044570490064). Retain the existing MuonH/Adam recipe, 1% warmup and cosine decay over the last 20% of training. Proxy and target have different batch sizes and update counts; equal total TPP does not make these equal or guarantee matching capability.

For every focus domain j, build a prior finite parent P_j=190,316,544 tokens (92,928 packed sequences), then a nested matched subset S_j=10,485,760 tokens (5,120 sequences). All p values for a domain share these exact pools. At focus fraction p:

    E_target(p) = p D_target / P_j = 15.82782369 p
    E_matched(p) = p D_proxy / S_j = 15.825 p
    E_unmatched(p) = p D_proxy / P_j = 0.87190083 p.

S_j is the packed approximation to P_j D_proxy/D_target. The existing StarCoder allocator audit bounds actual epoch mismatch by 0.0504% on its dense grid. This bound must be rechecked for the new source mappings, not assumed from nominal ratios alone.

At the fixed model/token ratio, requiring P_j >= D_proxy implies a maximum target repetition <=18.15324. The selected parent leaves margin for allocator rounding and a maximum near 15.83. A still-improving endpoint is a boundary minimum; it does not establish turnover. Extending far beyond this range while preserving a nonrepeating unmatched proxy requires a new scale setting, not silently shrinking the parent.

## Data construction

Reuse StarCoder's existing parent/subset artifacts. For Wikipedia and FineMath, materialize both the 190.32M parents and the 10.49M subsets before proxy training so the eventual target cannot acquire a different parent retrospectively.

All preparation, sources, caches, tokenizer files, executor state, and checkpoints stay in us-central1, with parent and child placement explicitly us-central1-a. Read no bulk raw data on the local workstation. Freeze regional source object generations, tokenizer/code hashes, exact quotas, sequence-index hashes, and final token-stream hashes before release.

Wikipedia uses the Dolma v1.7 raw wiki-0000 and wiki-0001 gzip shards in central1. FineMath uses only the finemath-3plus partition of the central1 raw FineMath download, corresponding to the existing catalog's HF revision 8f233cf; neither FineWeb-Edu nor the 4+ partition is substituted. FineMath's regional root is `gs://marin-us-central1/raw/finemath-7090a5/`; its partition/object inventory must be checked and frozen.

Proposed bounded sampling: equal token quotas across both Wikipedia shards; for FineMath, select 16 distinct parquet objects by the lowest seeded URI hashes across the complete 3+ object inventory and assign nearly equal packed-token quotas. Decode only the text field through the frozen native preprocessor, stopping at quotas. A short source fails rather than changing shards, partition, or region automatically. FineMath preparation needs a bounded parquet reader; the old StarCoder gzip reader is insufficient.

Materialize a global packed-sequence permutation of each finite parent, then select a uniform sample without replacement for the matched subset. Prespecify source selection seed 20260916, parent order seed 20260915 and matched subset seed 20260912; use the frozen StarCoder seed 20260912 subset for its comparison. These are samples conditional on chosen shards and within-shard prefixes, not a claim of uniform document sampling from the original corpus. Pool sizes count available packed token positions, not semantic uniqueness after deduplication.

The recorded original corpus sizes are 216,567,300,822 StarCoder tokens, 3,669,138,258 Wikipedia tokens and 34,001,855,255 FineMath-3+ tokens with the legacy Llama3/Marin tokenizer. These are provenance references, not exact counts under the new 32k tokenizer. Prior downsampling is specified by exact output token quotas, not by converting those legacy counts into an exact new-tokenizer fraction.

Reuse the identical six-component Nemotron background caches and fixed within-web token weights from StarCoder. Their target-sized pools have nonrepeat headroom. Preserve stable per-source shuffle keys and web stream order across focus-domain renaming and p=0/p=1 zero-weight removal. An existing p=0 checkpoint is reusable only when the model, training recipe, seeds, web support, allocation, and per-source order are identical; display it as a shared control, not as separate replicates.

## Initial grid, seeds and evaluation

Proposed common grid p = {0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00}. Corresponding matched epochs are {0, 0.79125, 1.5825, 3.165, 4.7475, 7.9125, 11.0775, 15.825}. This brackets the historical Wikipedia and math hints and stays on StarCoder's frozen 5%-spaced dense design.

Use one trainer seed 20260910 and one subset seed 20260912 per domain for this exploratory stage. The primary cross-domain figure uses that single-seed/single-subset StarCoder curve, not its six-replicate mean. Additional existing StarCoder replicates are secondary context. The seven-point StarCoder pilot supplies all proposed coordinates except 5% and20%; those two matched-proxy recipes need completion/reuse checks and separate release alongside the 14 new Wikipedia/FineMath nonzero points. Thus the common-grid proxy stage has 16 potentially new runs, rather than 14. Do not submit duplicate successful outputs, and do not alter existing frozen StarCoder recipe identities.

Primary selection/evaluation metric is the equal arithmetic mean of the seven frozen Uncheatable component BPBs: wikipedia_english, github_python, github_cpp, bbc_news, arxiv_physics, arxiv_computer_science, ao3_english. Use the same complete finite held-out caches/tokenizer/truncation rules as the current Figure 5 Uncheatable evaluation package; record each component as a secondary diagnostic. Do not reweight components by domain or select a different primary objective after viewing curves. Before cross-domain plotting, verify all three arms use exactly the same scoring population and aggregation.

The current Figure 5 Uncheatable evaluation is separate ongoing work; this design review should not wait for it or assume its results. The historical dataset-selection evidence below is exploratory context, not a prediction or independent confirmation of these new curves.

Choose the observed grid minimum, with exact ties going to smaller p. Show raw loss and excess above each curve's own observed grid minimum; mark boundary minima. No fitted curve chooses the primary optimum. A single trainer/subset pair supports descriptive conditional comparisons, not significance or domain-universal epoch optima. Any claim of separated optima or target selection gains needs prespecified confirmation; do not choose a more favorable subset after the first results. A refinement should add the adjacent unmeasured 5% grid points around each domain's observed minimum, applying the same rule to all domains, and preserve unfavorable or flat curves.

If target curves are later added, use the same p grid and frozen parents. Report target loss at each proxy-selected p and target-grid regret, with the target arm still using one prespecified trainer seed. An optional unmatched arm uses the identical parent at proxy scale, and is not part of the first-stage training budget.

## Prior evidence and cost

`evidence.json` extracts the focal-domain rows from the corrected 60M 39-bucket intervention archive. The best observed Uncheatable epochs are Wikipedia 3.621, FineMath 7.243 and synthetic math 7.243. The often-mentioned 14.486-epoch synthetic-math minimum is for OlmoBaseEval, not Uncheatable. FineMath's Uncheatable loss rises 0.003603 BPB from 7.243 to 14.486 epochs; synthetic math rises 0.001304. These are different mixtures, support sizes, models and possibly evaluation versions from the proposed experiment. They motivate candidate choice, not a transported optimum guarantee.

| Package | Target runs | Proxy runs | Estimated training FLOPs |
|---|---:|---:|---:|
| First-stage new Wikipedia/FineMath points |0|14|3.48724e17|
| Two missing StarCoder common-grid points |0|2|4.98177e16|
| First-stage maximum additional total |0|16|3.98542e17|
| Later Wikipedia/FineMath targets, shared p=0 reused |14|0|9.32946e19|
| Optional unmatched Wikipedia/FineMath curves, shared p=0 reused |0|14|3.48724e17|
| Figure 5 original seven-point experiment |7|50|4.78928e19|
| Figure 5 including five-point refinement budget |12|90|8.22086e19|

The target per-point cost is identical to Figure 5's target per-point cost; it is 267.53 times the tiny proxy's cost. These estimates exclude calibration, preparation, evaluation, compilation and hardware overhead. Two additional StarCoder target points at 5% and20% would cost 1.33278e19 if a matched common-grid target comparison across all three domains were requested; their existing frozen recipes are not released here.

## Review requested and submission boundary

Assess (1) whether the settings implement matched total TPP and matched actual epoch exposure with nonrepeating unmatched proxies, (2) whether range/grid/data/evaluation controls can answer the narrow scientific question, (3) whether the cap-related claim must be narrowed or needs an additional experiment, and (4) whether the later target expansion has hidden scientific or practical changes.

Separate scientific-design blockers, useful improvements and implementation gates. The new dataset builders and domain launcher do not exist yet; do not label this packet executable or submission-ready. Before any training submission, require source/cache receipts, actual allocator and named-shuffle audits including endpoints, complete seven-component evaluation parity, immutable run/output identities, code/runtime pin verification, region safety validation, and a review of the concrete launch plan. Reuse verified successful StarCoder outputs. The user has requested this CC review, not submission of the additional sweeps.
