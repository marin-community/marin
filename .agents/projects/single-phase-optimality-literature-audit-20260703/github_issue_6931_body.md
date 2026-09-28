## Summary

Audit whether nearby curriculum / specialized-pretraining papers rule out the possibility that a single-phase fixed-bucket mixture is optimal or near-optimal for our current data-mixing setting.

Current conclusion: they do not rule it out. The papers support the prior that phase/order effects can matter, but they do not imply that a two-phase schedule must substantially beat the best single-phase mixture for broad smooth objectives such as OLMoBaseEval Table-9 macro BPB or uncheatable BPB.

The key distinction is aggregate exposure versus phase placement. A clean phase-order test should hold aggregate exposure fixed,

$$
\bar w=\gamma_0 w^{(0)}+\gamma_1 w^{(1)},
$$

and vary only where buckets appear in training. Comparisons that change the amount of target-domain exposure cannot isolate the value of phase order.

## Papers Audited

### 1. Replaying pre-training data improves fine-tuning

**Setup in our terms.** This paper is closest to a two-bucket target-adaptation setup: generic data plus scarce target data. It studies target validation loss, using two-stage schedules parameterized by replay fraction \(\rho\) and target Stage-2 allocation \(\alpha\).

The paper's main schedule sweep is relevant because it explicitly varies phase placement. It finds that replay-only improves over a mid-training baseline, and the full schedule improves further for FineMath and Flan. StarCoder is a useful counterexample: replay-only and full schedule both report \(1.53\times\) data-efficiency improvement, so phase-placement gains are not universal even in their two-domain setting.

**Why the highlighted Appendix H.2 statement does not prove single-phase suboptimality.** Appendix H.2 fits a reference scaling law by changing how many target tokens the reference algorithm sees. The reference runs are uniform-throughout: the authors write that they "mix data uniformly throughout training." They then note that reference runs with "extra data outperform the best data orders" in the low-target-data setting.

That comparison changes aggregate target exposure. It says that a uniform-throughout reference algorithm with more target data can beat clever schedules that only receive the low target-data budget used in the main paper. That is evidence that aggregate exposure can dominate schedule effects, not evidence that single-phase mixtures are intrinsically suboptimal.

**Deviation from our current setting.**

- Their objective is target-domain validation loss, not a broad macro BPB objective.
- Their target data is scarce and repeated.
- Their schedule is low-dimensional and manually swept.
- Their data-efficiency metric is defined through a reference scaling law, not direct optimization of a fixed policy class.
- Learning-rate schedule and cooldown placement are part of the treatment.

The two-domain aspect is not itself a reason to discount the paper. We also have an internal StarCoder/Nemotron two-domain, two-phase landscape where the observed global optimum is near the single-phase diagonal. That makes the replay paper structurally relevant, but it also reinforces that phase asymmetry can be small for some objectives.

### 2. The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data

**Setup in our terms.** This paper studies domain adaptation with a general corpus and one specialized domain dataset. The main comparison is:

- NPT: pretrain on general data, then finetune on domain data.
- SPT: mix a small fraction \(\delta\) of domain data throughout pretraining, then finetune on domain data.

In our notation, the pretraining part of SPT is close to a one-phase mixture,

$$
w_{\mathrm{domain}}=\delta,\qquad w_{\mathrm{general}}=1-\delta.
$$

The paper then appends a separate finetuning policy and evaluates post-finetuning domain test loss.

**Does it rule out single-phase near-optimality?** No. If anything, it supports the idea that diffuse early exposure can be strong. SPT has domain data present throughout pretraining as a small fraction of tokens. Its success argues against the opposite extreme: all general data first, then all domain data only during finetuning.

The paper does show that timing matters: early domain exposure is not fully replaced by replay-based later training in their MusicPile setting. It also finds that the best \(\delta\) depends on compute budget and dataset size. This supports a budget- and repetition-dependent view of phase effects, not a universal claim that late two-phase schedules dominate.

**Deviation from our current setting.**

- It includes finetuning after pretraining; our current scaling validations evaluate trained checkpoints directly.
- \(\delta\) controls both mixture composition and repetition of a small domain corpus.
- The objective is post-finetuning specialized-domain test loss, not broad macro BPB.
- The question is when domain data should enter before finetuning, not whether two learned pretraining phases beat one fixed pretraining mixture.

The one-domain setup remains relevant. Our StarCoder landscape is also effectively a two-domain phase-order test. The more important difference is the downstream protocol and objective, not the number of domains.

## What Contradicts Our Findings?

The papers contradict any strong claim that phase/order never matters. Both provide regimes where when data appears during training changes performance.

They do not directly contradict our current observation that single-phase candidates are competitive for broad smooth objectives. The replay paper itself has a StarCoder case where full phase scheduling does not improve beyond replay-only, and the Finetuner's Fallacy paper is partly evidence for the strength of diffuse early exposure.

The careful synthesis is:

- The two-phase policy class is larger in principle.
- Our current two-phase solver has not reliably harvested a transferable phase-asymmetry advantage.
- Prior work says such advantages can exist, but their magnitude depends on objective, exposure, repetition, scale, and downstream protocol.
- The decisive next test is a matched-exposure phase-order experiment.

## Proposed Next Experiment

Run a matched-exposure phase-order DOE:

1. Choose strong aggregate mixtures \(\bar w\), including the best single-phase mixtures.
2. Generate two-phase variants with the same \(\bar w\) but different phase placement.
3. Validate whether Table-9 macro BPB or uncheatable BPB changes beyond the 3e18 noise floor.

This directly tests whether phase placement itself matters in our regime, without confounding it with aggregate exposure.

## Acceptance Criteria

- [ ] Record the literature conclusion: these papers motivate phase-order tests but do not rule out single-phase near-optimality.
- [ ] Define the matched-exposure phase-order experiment.
- [ ] If we run follow-up experiments, report whether phase-placement changes are larger than the measured 3e18 noise floor.
