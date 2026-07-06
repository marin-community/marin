## Literature Audit: Does Prior Work Rule Out Single-Phase Near-Optimality?

- Effort: medium
- Date: 2026-07-03
- Scope: `Replaying pre-training data improves fine-tuning` and `The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data`
- Local sources:
  - Replay paper source: `/tmp/replay_pretraining_paper/`
  - Finetuner's Fallacy source: `/tmp/finetuner_fallacy_paper/`
  - Data mixing theory: `/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper/theory.md`

### TL;DR

These papers support the broad prior that data order and phase placement can matter. They do not rule out a single-phase fixed-bucket mixture being optimal or near-optimal for our current broad multi-domain objectives.

The main reason is that both papers study adaptation regimes that differ from ours in objective, budget constraints, and downstream protocol. They primarily ask how to use a scarce target-domain dataset relative to a generic corpus, often under fixed target-token budgets or followed by finetuning. Our current question is different: for broad objectives such as OLMoBaseEval Table-9 macro BPB and uncheatable BPB, does a two-phase policy materially improve over the best single-phase policy at the same budget?

The most important methodological point is that phase/order effects must be separated from aggregate exposure. In our notation, a two-phase policy \(\mathbf w=(w^{(0)},w^{(1)})\) has both an aggregate exposure profile and a phase placement profile. A clean phase-order test should hold something like

$$
\bar w=\gamma_0 w^{(0)}+\gamma_1 w^{(1)}
$$

fixed, and vary only the phase allocation. The highlighted appendix passage from the replay paper does not do this. It says that a uniform-throughout reference algorithm with more target data can beat the best low-target-data schedules. That is evidence that aggregate target exposure can dominate schedule improvements, not evidence that a single-phase schedule is suboptimal.

### Current Theoretical Frame

Our theory document defines data mixing as policy optimization. A policy \(\pi\) samples a selected multiset \(T\), the ordering kernel \(\kappa_B\) turns it into a stream \(S\), training produces \(\theta\), and an evaluation procedure returns \(Y_b\). The central object is the expected benchmark value \(\mu_{b,B}(\pi)\).

For fixed-bucket mixtures, a one-phase weight vector \(w\in\mathcal W_B\) induces \(\pi_{w,B}\) and response surface

$$
f_{b,B}(w)=\mu_{b,B}(\pi_{w,B}).
$$

For two phases, the control is

$$
\mathbf w=(w^{(0)},w^{(1)})\in\mathcal W_B^{(2)},
$$

with response \(f_{b,B}(\mathbf w)=\mu_{b,B}(\pi_{\mathbf w,B})\). In principle, the two-phase policy class contains single-phase policies, so the true optimum over the two-phase class cannot be worse than the true optimum over the single-phase class. If the best observed single-phase candidate beats the best observed two-phase candidate, that diagnoses sampling, optimization, surrogate, noise, or validation coverage; it does not prove the two-phase policy class is inferior.

### Paper 1: `Replaying pre-training data improves fine-tuning`

#### Setup in Our Notation

This paper studies a two-distribution target-adaptation problem. There is a generic distribution, usually C4, and a target distribution such as FineMath, StarCoder, or Flan. In our notation this is close to a \(k=2\) fixed-bucket setting:

- bucket 0: generic data,
- bucket 1: target data,
- benchmark \(b\): target validation loss,
- policy class: restricted two-stage schedules rather than arbitrary many-bucket mixtures.

The fine-tuning setup has Stage 1 mostly or entirely generic, and Stage 2 target plus optional generic replay. The mid-training/pretraining setup uses two schedule parameters:

- replay fraction \(\rho\): how much generic data appears in Stage 2,
- target Stage 2 allocation \(\alpha\): how much target data appears late versus early.

The paper's appendix makes the aggregate-exposure constraint explicit. It defines target step fraction \(\gamma\), Stage 2 target weight \(w_2\), Stage 1 target weight \(w_1\), and confirms that

$$
w_1(1-\delta)+w_2\delta=\gamma.
$$

This is close to the matched-exposure phase-order test we want: vary where target data appears while holding total target exposure fixed.

#### Where It Deviates From Our Setting

The deviations are material:

- The objective is target validation loss, not a broad macro objective across many unrelated benchmarks.
- The target data is scarce and repeated; our Table-9 and uncheatable objectives are broad smooth BPB objectives over a production-like mixture family.
- The schedule variables are low-dimensional and manually swept, while our two-phase mixture lives in a high-dimensional sparse design.
- Their main data-efficiency metric is defined relative to a reference scaling law for target tokens, not as direct optimization of \(f_{b,B}\) over a fixed policy class.
- Learning-rate schedule matters directly in their conclusions: WSD and cooldown placement are part of the treatment.

The two-distribution aspect is not by itself a reason to discount the paper. We also have an internal two-domain, two-phase StarCoder/Nemotron landscape where the observed global optimum is near the diagonal \(p^{(0)}\approx p^{(1)}\). That toy setting is structurally closer to the replay paper than the 39/167-bucket production setting, and it still suggests that phase asymmetry need not be large for every objective.

#### Does It Rule Out Single-Phase Near-Optimality For Us?

No. It rules out some simple schedules in their target-adaptation regime, but not single-phase near-optimality for our broad fixed-bucket objective.

The paper does show that schedule effects can be real. In the two-stage sweep, replay-only improves data efficiency over the mid-training baseline, and the full schedule search improves more for FineMath and Flan. But StarCoder is a useful counterexample: replay-only and full schedule both report \(1.53\times\). In the authors' discussion, StarCoder's optimal data schedule only requires adding replay data to Stage 2.

The highlighted appendix passage is especially easy to misread. The appendix is fitting a reference scaling law. For that reference, they vary the amount of target data and train uniform-throughout runs. The sentence says that the best such reference runs with extra target data outperform the best low-target-data data orders. This comparison does not hold aggregate target exposure fixed. It says: if we give the uniform reference algorithm more target tokens, it can beat clever schedules that only receive the low target-data amount used in the main paper. That is not evidence against single-phase mixtures; if anything, it is evidence that aggregate exposure can dominate schedule effects.

Short verbatim evidence:

- "mix data uniformly throughout training" (appendix line 166)
- "extra data outperform the best data orders" (appendix line 168)

#### What Contradicts Our Current Findings?

The FineMath and Flan results contradict a strong claim that phase/order never matters. They show two-stage schedules can beat replay-only when the objective is a scarce target-domain validation loss.

They do not directly contradict our current observation that the best single-phase candidates are competitive for broad Table-9 and uncheatable BPB. The paper itself lists limitations that matter here: two distributions, simple schedules, validation loss, and no continuous annealing or sample-level ordering. It also reports a StarCoder case where the full two-stage schedule does not improve beyond replay-only, which is directionally compatible with our finding that phase asymmetry may be weak or hard to identify for some objectives.

#### Takeaway For Our Experiment Design

The paper motivates a matched-exposure phase-order DOE:

1. Choose a strong aggregate mixture \(\bar w\).
2. Create schedules that keep \(\bar w\) fixed.
3. Move selected buckets earlier or later.
4. Evaluate whether \(f_{b,B}(\mathbf w)\) changes beyond the 3e18 noise floor.

This is the clean way to test whether phase placement itself matters in our setting.

### Paper 2: `The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data`

#### Setup in Our Notation

This paper studies domain adaptation with a general corpus and one specialized domain dataset. Its main comparison is:

- NPT: pretrain on general data, then finetune on domain data.
- SPT: mix a small fraction \(\delta\) of domain data throughout pretraining, then finetune on domain data.

In our notation, SPT is close to a one-phase pretraining mixture:

$$
w_{\mathrm{domain}}=\delta,\qquad w_{\mathrm{general}}=1-\delta.
$$

It then appends a separate finetuning policy. The final measured response is post-finetuning domain test loss, not the pretrained model's broad evaluation macro.

#### Where It Deviates From Our Setting

The deviations are also material:

- It includes a finetuning stage after pretraining; our current scaling validations evaluate the trained checkpoint directly on Table-9 and uncheatable BPB.
- \(\delta\) controls both mixture composition and repetition of a small domain corpus.
- The objective is specialized-domain post-finetuning test loss, not broad macro BPB.
- The paper's main question is whether domain data should enter before finetuning, not whether two learned pretraining phases beat one fixed pretraining mixture.

The one-domain setup is still relevant to us. Our StarCoder two-phase landscape is also effectively a two-domain setting, and its optimum is close to constant across phases. The more important difference is not the number of domains; it is that Finetuner's Fallacy evaluates post-finetuning domain test loss after a pretraining mixture, whereas our current validation evaluates pretrained checkpoints directly on smooth broad BPB metrics.

#### Does It Rule Out Single-Phase Near-Optimality For Us?

No. It actually provides evidence that diffuse early exposure can be strong.

SPT is itself a uniform pretraining mixture over time, with domain data present from the start as a small fraction of tokens. Its success does not imply that a late-heavy two-phase schedule is optimal. It argues against the opposite extreme: all general data first, then all domain data only during finetuning.

The paper does say that when domain data enters training matters. It also finds that the optimal \(\delta\) changes with compute budget and domain dataset size. That supports our budget-dependent theory, but it does not settle the one-phase versus two-phase question for broad objectives.

Short verbatim evidence:

- "repeated starting from pretraining" (main line 37)
- "optimal mixture fraction \(\delta\) also depends" (factors line 162)

#### What Contradicts Our Current Findings?

The strongest tension is that the paper reports lasting benefits from early domain exposure that replay-based later training does not fully replace. That challenges a pure aggregate-exposure-only model.

But this tension cuts toward early/single-mixture exposure, not necessarily toward late two-phase schedules. The paper also reports that for smaller datasets, early heavy repetition can overfit and later specialized continued pretraining can be better. In other words, the sign and magnitude of phase effects depend on dataset size, repetition, compute budget, and target objective.

#### Takeaway For Our Experiment Design

The paper suggests we should treat phase effects as budget- and repetition-dependent, not universal. It also suggests that a good one-phase mixture may be a strong baseline because diffuse exposure can regularize domain learning. For our setting, this makes it even more important to avoid interpreting "single-phase is competitive" as a bug by default. It may be true for broad BPB objectives at 300M/3e18, or it may indicate that our two-phase optimizer has not found the right phase-placement interventions.

### Synthesis

The two papers support three claims we should keep:

1. Phase/order effects can be real.
2. Aggregate exposure and repetition can dominate schedule details.
3. The sign and size of schedule effects are objective-, budget-, and dataset-dependent.

They do not support the stronger claim that a two-phase schedule must substantially beat a single-phase mixture for our current Table-9 or uncheatable objective. They also do not support the stronger claim that our best observed single-phase candidate is truly optimal. The current evidence is best framed as:

- The two-phase policy class is larger in principle.
- Our current two-phase solver has not reliably harvested a phase-asymmetry advantage.
- Prior work says such advantages can exist, but in regimes that differ from ours.
- The decisive next test is matched-exposure phase intervention, not another exposure-confounded comparison.

### Issue-Ready Comment

We audited two nearby papers to check whether they rule out the possibility that a single-phase fixed-bucket mixture is optimal or near-optimal for our current setting.

**TL;DR:** They do not rule it out. Both papers support the prior that phase/order effects can matter, but their setups differ materially from our current Table-9 and uncheatable BPB optimization problem. The key distinction is aggregate exposure versus phase placement. A clean phase-order test should hold aggregate exposure \(\bar w=\gamma_0 w^{(0)}+\gamma_1 w^{(1)}\) fixed and vary only where buckets appear in training. We should not dismiss these papers merely because they are low-dimensional: our own StarCoder two-domain/two-phase landscape has an observed optimum near the single-phase diagonal.

**Replaying pre-training data improves fine-tuning.** This paper is closest to a two-bucket target-adaptation setup: generic data plus scarce target data. It sweeps replay fraction \(\rho\) and target Stage-2 allocation \(\alpha\), mostly optimizing target validation loss. It does show that schedules matter for FineMath and Flan. But it does not show that two-phase schedules must beat single-phase mixtures for broad objectives. StarCoder is a counterexample where the full schedule does not improve beyond replay-only. The highlighted appendix sentence is also not evidence against single-phase mixtures: it compares uniform-throughout reference runs with more target data against low-target-data data-order schedules, so it changes aggregate exposure.

**The Finetuner's Fallacy.** This paper studies specialized pretraining: mix a small fraction \(\delta\) of domain data throughout pretraining, then finetune. In our notation, the pretraining part is close to a one-phase mixture \(w_{\mathrm{domain}}=\delta\), \(w_{\mathrm{general}}=1-\delta\), followed by a separate finetuning policy. The paper argues that early diffuse exposure can be better than keeping all domain data for finetuning. This supports the idea that timing matters, but it does not imply late two-phase schedules dominate. It also finds that the best \(\delta\) depends on compute budget and dataset size.

**Implication for us.** Prior work motivates a matched-exposure phase-order DOE:

1. Choose strong aggregate mixtures \(\bar w\), including the best one-phase mixtures.
2. Generate two-phase variants with the same \(\bar w\) but different phase placement.
3. Validate whether Table-9 macro BPB or uncheatable BPB moves beyond the 3e18 noise floor.

Until we run that, the most careful statement is: prior work suggests phase effects can exist, but it does not contradict our current finding that single-phase candidates are competitive for broad smooth objectives at this scale. Our current gap is likely about identifying or exploiting transferable phase asymmetry, not a literature contradiction.

### Source Ledger

| Source | Type | Location | Claim used |
|---|---|---|---|
| Data mixing theory | local theory note | `theory.md` lines 33-39, 79-99, 105-127 | Defines policy pipeline, fixed-bucket mixture, two-phase policy, and restricted policy class. |
| Replay paper | paper source | `/tmp/replay_pretraining_paper/sections/two_stage.tex` lines 19-25 | Defines two schedule variables and reports replay-only versus full-schedule improvements. |
| Replay paper appendix | paper source | `/tmp/replay_pretraining_paper/sections/appendix.tex` lines 1-17, 150-168 | Defines schedule equivalences and shows the highlighted reference-algorithm comparison changes target data amount. |
| Replay paper limitations | paper source | `/tmp/replay_pretraining_paper/sections/limitations.tex` lines 3-7 | Records limitations: two distributions, simple schedules, validation loss. |
| Finetuner's Fallacy | paper source | `/tmp/finetuner_fallacy_paper/sections/performance.tex` lines 18-33, 37-49 | Defines NPT/SPT, \(\delta\), repetition, and finetuning setup. |
| Finetuner's Fallacy | paper source | `/tmp/finetuner_fallacy_paper/sections/factors.tex` lines 117-142, 162-193 | Shows dataset-size and compute-budget dependence of optimal domain mixture. |
| Finetuner's Fallacy | paper source | `/tmp/finetuner_fallacy_paper/sections/replay.tex` lines 1-15 | Shows replay does not fully replace early domain exposure in their domain-adaptation setting. |
