## Background Research Brief

- Effort: medium
- Stop rule: stopped after the two requested papers, the pinned Marin implementation, local one-vs-two artifacts, and CC/Fable review no longer changed the conclusion.
- Date: 2026-07-03

### Question

Do `Replaying pre-training data improves fine-tuning` and `The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data` rule out the possibility that a single-phase fixed mixture is optimal or near-optimal for our multi-domain Marin data-mixing objectives?

### Current Marin Context

Local paired one-vs-two artifacts are under:

- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/one_vs_two_phase_swarm_debug_20260630/`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/one_vs_two_phase_best_mixture_comparison_20260701/`

On 240 paired qsplit rows, the single-phase average is slightly lower for both smooth objectives:

- `eval/uncheatable_eval/bpb`: two mean `0.997548`, single mean `0.996694`, single-better fraction `0.500`.
- OLMoBaseEval Table-9 macro BPB: two mean `1.060699`, single mean `1.058430`, single-better fraction `0.483`.

This supports "no obvious sampled-family dominance" more than it supports a true policy-class optimum claim. The two-phase class contains the single-phase class in principle; best-observed single-phase beating best-observed two-phase therefore mainly diagnoses search/sample/surrogate failure, not a mathematical optimum reversal.

### Internal Prior Work

The replay paper points to a pinned Marin implementation at commit `bfbc4492aefe50291829e2ceebf1b3b94186da9c`, `experiments/two_stage`.

Relevant implementation facts:

- `TwoStageConfig` defines a two-domain rare/common setting.
- The search variables are `replay_ratio`, `rare_stage2_allocation`, and implied `stage2_duration`.
- Stage weights are derived from a fixed rare-token constraint, not an unconstrained many-bucket macro objective.
- `mid_training.py` sweeps `replay_ratio in [0.0, 0.25, 0.5, 0.75, 0.875]` and `rare_stage2_allocation in [1.0, 0.5, 0.25, 0.125]` for FineMath, StarCoder, and Flan.

This setup is materially different from our 39/167-bucket mixture family and broad macro objectives.

### External Prior Art

#### Replaying pre-training data improves fine-tuning

Main claims from the paper source:

- The controlled setup is a 150M model, two data pools, 4M target tokens, up to 4B generic C4 tokens, and target validation loss as the objective.
- Generic replay during fine-tuning improves target data efficiency up to `1.87x`.
- In the mid-training setting, replay alone improves up to `2.06x`.
- Searching the full two-stage schedule gives `1.53x` StarCoder, `2.49x` FineMath, and `4.80x` Flan over the mid-training baseline.
- Introducing target data earlier gives additional gains for two of three domains; StarCoder is a counterexample where replay-only and full schedule both report `1.53x`.
- The paper explicitly lists simplifications: two distributions, validation loss, and no continuous annealing/sample-level ordering.

Interpretation: this shows phase/order effects can matter for scarce target-domain adaptation, but it does not imply two-phase schedules must dominate broad multi-domain macro BPB objectives.

#### The Finetuner's Fallacy

Main claims from the paper source:

- Specialized pretraining (SPT) mixes a small domain-specific fraction `delta` into general pretraining, then finetunes.
- SPT improves domain performance across MusicPile, ChemPile, and ProofPile.
- Benefits are largest when the target domain is underrepresented.
- The useful mixture fraction depends on dataset size and compute budget.
- Excessive repetition can make aggressive early mixing overfit; smaller datasets can prefer later specialized continued pretraining.
- During pretraining, domain data as a small batch fraction acts as a regularizer and can tolerate many repeats.

Interpretation: this paper actually strengthens the plausibility that diffuse early/single-mixture exposure can be strong. It is not evidence that late-stage concentration always wins.

### CC / Fable Review

Fable review was invoked through `env -u ANTHROPIC_API_KEY claude --model claude-fable-5`. The broad read-only review hung for several minutes and was stopped; a shorter evidence-in-prompt review returned.

Key corrections from Fable:

- Mean paired comparisons are not policy-class optimum comparisons.
- Best observed single-phase beating best observed two-phase proves search/sample/surrogate failure or insufficient coverage, not that the true two-phase optimum is worse.
- The current deltas are small enough that they should be interpreted against seed/eval noise before claiming "single-phase wins."
- Verify whether paired comparisons hold time-averaged aggregate exposure fixed; otherwise they conflate aggregate mixture differences with phase-order effects.
- Both papers move away from pure final-stage target concentration and toward more mixing, which is directionally compatible with the single-phase/null view.
- The strongest counter-prior is not these papers but the empirical success of OLMo-style second-phase annealing/Dolmino.

### Evidence Map

#### Claim: The requested papers do not rule out single-phase near-optimality in our setting.

- Support:
  - Replay paper: target-domain objective, two-domain rare/common setup, finite rare-token constraint.
  - Finetuner's Fallacy: diffuse early exposure can beat late-only specialization; optimal mixing depends on data size and budget.
  - Local qsplit paired rows: sampled-family means are very close, with near-coinflip pair win fractions.
- Contradictions:
  - Replay paper: two-stage schedules improve rare-target loss in several settings.
  - Finetuner's Fallacy: "when" domain data enters training can matter.
  - OLMo/Dolmino prior: quality annealing can matter for broad downstream metrics.
- Directness to Marin:
  - Replay implementation is in Marin but differs strongly from our objective and dimensionality.
  - Fallacy is external and domain-adaptation focused, but its repetition/early-exposure mechanism is relevant.
- Confidence: moderate.
- Action: frame as "not contradicted; order effects are objective/regime-dependent," then run matched-exposure interventions.

### Recommended Next Experiments

#### 1. Measure the 3e18 noise floor.

- Minimum experiment: repeat fixed proportional and/or best one-phase Table-9 checkpoints with native Table-9 and uncheatable eval.
- Baseline/control: same mixture, independent seeds/subsets.
- Expected signal: if single-vs-two differences are below `1-2 sigma`, do not chase phase-order claims from current data.
- Falsifier: low enough noise that observed gaps are clearly significant.
- Cost/risk: low/moderate.

#### 2. Re-analyze existing one-vs-two data with paired statistics.

- Minimum experiment: paired confidence intervals, top-decile comparisons, worst-tail signature analysis, aggregate-exposure matching audit.
- Baseline/control: qsplit pair mapping.
- Expected signal: distinguish "phase order useless" from "two-phase sampler creates bad tails."
- Falsifier: robust top-decile two-phase advantage after exposure matching.
- Cost/risk: local only.

#### 3. Matched-exposure phase-swap DOE.

- Minimum experiment: choose a few buckets with strong late-placement priors plus controls; hold total aggregate exposure fixed; compare all-early, uniform/single, and all-late placement under the same total tokens and LR schedule.
- Baseline/control: exposure-matched single-phase mixture.
- Expected signal: per-bucket and macro metrics reveal whether phase effects exist but cancel in macro.
- Falsifier: no effect beyond noise even for high-prior domains.
- Cost/risk: moderate.

### Handoff

Conclusion for paper framing: The cited papers are evidence that phase/order effects can be real, especially for scarce target-domain adaptation, but they do not contradict a near-single-phase optimum for broad multi-domain macro BPB at 300M/3e18. The defensible claim is weaker and cleaner: in our current sampled family and smooth macro objectives, phase asymmetry has not produced a reliable advantage beyond search/noise; the next decisive test is matched-exposure phase-order intervention.
