# Marin 8B Retrospective

This is a retrospective on our first generation Marin 8B run.
We include details on the data mix, hyperparameters, and other details. We also include observations during the run and
document some of the mistakes we made and lessons learned.

This document was not created during the run, but rather after the fact. Some details may be missing or incorrect,
but we have tried to be as accurate as possible.

In addition, the [actual code](https://github.com/marin-community/marin/blob/852c53f9741b233549daf9f1649fe88c9c5a170c/experiments/tootsie/exp600_tootsie.py)
for the run should be treated as more authoritative than this document. See also the [main issue on GitHub](https://github.com/marin-community/marin/issues/600)
or the [WandB report](https://wandb.ai/stanford-mercury/marin/reports/Tootsie-8B---VmlldzoxMTY3MzU3OA)


## The "Tootsie Roll" process

A core premise of the Marin 8B run was that we didn't fully know the best recipe—
so we just started training with what we had and adapted along the way.
Internally, we referred to this as the "Tootsie" process, a reference to
[Tootsie Rolls, which use a "graining" process](https://en.wikipedia.org/wiki/Tootsie_Roll) where each day's batch
contains a bit of the previous day's, seeding crystallization or something. (We are not food scientists.)
This is admittedly a bit of a strained metaphor, but the idea was that we'd keep folding in new data and whatever
else as the training process went on.
(As it would turn out, dear reader, we would often change more than the data...)

## Model Basics

### Model Size

We decided to build an ~7-8B class model mostly out of pragmatism: we initially only had reserved capacity
to train a model of that size for long enough.

### Architecture

We settled on the "[Llama architecture](https://arxiv.org/abs/2302.13971)" for the usual reasons: it has been shown to work well,
easier to plug into existing inference stacks, no one ever got fired
for buying IBM, etc.

We used the same settings as Llama 3.1 8B. More specifically:

| **Parameter**         | **Value**   |
|-----------------------|-------------|
| `hidden_dim`          | 4096        |
| `intermediate_dim`    | 14336       |
| `num_heads`           | 32          |
| `num_kv_heads`        | 8           |
| `num_layers`          | 32          |
| `activation_function` | `silu`      |

We used [Levanter](https://github.com/stanford-crfm/levanter)'s implementation of the Llama architecture.
We trained with a sequence length of 4096 tokens per sample.
We used JAX's TPU Splash Attention kernel.

We used mixed float32/bfloat16 precision, with parameters and optimizer states in float32 and compute in bfloat16
(except for the final softmax over the vocabulary.)
We used the [AdamW](https://arxiv.org/abs/1711.05101) optimizer.

### Tokenizer

In Marin, we ultimately converged on using the Llama 3 tokenizer after [an experiment](https://github.com/marin-community/marin/issues/524) showing it to be superior
to both Llama 2 and NeoX in terms of bits-per-byte (bpb).

### Batch Schedule

We used a varying batch schedule, with between 1024 * 4096 = 4Mi tokens, 3072 * 4096 = 12Mi tokens, and 4096 * 4096 = 16Mi tokens.
As with most things in this run, we did not initially plan for this schedule; we changed the batch schedule to adapt to our changing infrastructure, data availability, and timelines.
(Note: we use `Mi` to mean Mebi-(https://simple.wikipedia.org/wiki/Mebibyte), not mega-.)

### Checkpointing Policy

We saved permanent full checkpoints every 20,000 steps (which, due to the varying batch schedule, could be a varying number of tokens)
To save on egress bandwidth charges, we have not made these available. However, if they are of interest, we should be
able to share them. ("Temporary" checkpoints were saved much more frequently.)

### Hardware

The TPU hardware varied between two different TPU clusters, generously provided by Google's [TPU Research Cloud]( https://sites.research.google/trc/about/):

- 2x v5e-256, configured using [multislice](https://cloud.google.com/tpu/docs/multislice-introduction)
- 1x v4-2048

The first phase was run on the 2x v5e-256, while subsequent phases were run on the v4-2048.

# Training Phases

Retrospectively, we can partition the 8b run into several different phases. We have given them animal names as monikers.  Here is a short summary:

- *[Kestrel](#phase-1-kestrel-dclm-wsd-s-phase) (DCLM WSD-S Phase)*: In the first phase, we used the "DCLM mix" and [WSD-S](https://arxiv.org/abs/2410.05192) for about 2.7T tokens. We used 2x TPU v5e-256 coordinated with multislice for this. (0->2.7T tokens)
- *[Ocelot](#phase-2-ocelot-dclm-ema-phase) (DCLM WSD Phase)*: We were given access to a v4-2048 slice and moved to that. To better utilize the hardware, we increased our batch size by 50%. We also switched from WSD-S to WSD. We kept the learning rate high until 3.78T tokens (2.7T->3.78T tokens).
- *[Jellyfish](#phase-3-jellyfish-first-cooldown) (First Cooldown)*: It was time to cooldown as we were starting to run low on DCLM. Following a recent work on midtraining (e.g. [Olmo 2](https://arxiv.org/abs/2501.00656)), we decided to fold in higher quality data during cooldown. (3.78T->4.78T tokens)
- *[Phoenix](#phase-4-phoenix-reheated) (Reheated)*: We had more time for training, so we rapidly rewarmed the model and transitioned our mixture to [Nemotron-CC](https://arxiv.org/abs/2412.02595) and added [Starcoder](https://huggingface.co/datasets/bigcode/starcoderdata). (4.78T->11.1T tokens)
- *[Starling](#phase-5-starling-second-cooldown) (Second Cooldown)*: Now we were running low on time, so we started another cooldown. We followed a similar process to the first cooldown, but added a few new datasets that we had created and also some that had dropped since our previous attempt. (11.1T->12.75T tokens)

We emphasize that these phases were not planned in advance. Decisions were made reactively based on changing time lines and data availability.

Moreover, while there is a single straight-line set of phases that produced the final Marin 8B artifact,
there were a number of short branches that did not make it into the final run. Many of these were by design: there were other trial
[LLama 3](https://arxiv.org/abs/2302.13971) micro-annealing runs (see also [Olmo 2](https://arxiv.org/abs/2501.00656), [Yi](https://arxiv.org/html/2403.04652v1), [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)) that helped decide our cooldown mix, as well as attempts at "deeper cooldowns.
We detail these [GH#784](https://github.com/marin-community/marin/issues/784), [GH#820](https://github.com/marin-community/marin/issues/820), and [GH#898](https://github.com/marin-community/marin/issues/898), though provide a summary later in this one.
We also ran some experiments investigating issues with "SFT-ability,"
documented in these issues:

- [#898 Raccoon](https://github.com/marin-community/marin/issues/898)
- [#916 Spoonbill](https://github.com/marin-community/marin/issues/916)

## Phase 1: Kestrel (DCLM WSD-S Phase)

In the first phase, we trained from scratch using the best publicly available dataset at the time:
[DCLM Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0).
We decided to use [their best mixture](https://arxiv.org/abs/2406.11794): DCLM Baseline, [Starcoder](https://huggingface.co/datasets/bigcode/starcoderdata), and [Proofpile 2](https://huggingface.co/datasets/EleutherAI/proof-pile-2), though we did not use a curriculum
as they did.

### Hardware

We started with a reserved 2x TPU v5e-256 slice, which was the largest TPU slice available to us at the time. We used [multislice](https://cloud.google.com/tpu/docs/multislice-introduction) to coordinate the two slices.

### Data Mix

At the beginning, we decided to use the DCLM 7B mix in
ratios roughly proportional to token count. (DCLM 7B was, at the time, the best open source model.)

Specifically, this meant:

| Dataset                                                                          | Percentage |
|----------------------------------------------------------------------------------|------------|
| [DCLM Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | 92.6%      |
| [Starcoder](https://huggingface.co/datasets/bigcode/starcoderdata)               | 6.1%       |
| [Proofpile 2](https://huggingface.co/datasets/EleutherAI/proof-pile-2)                        | 1.3%       |

We planned on adding new datasets as we (and others!) developed them.
For evaluation, we initially tracked a large subset of Paloma (with a particular focus on the `c4_en` subset) during training.

### WSD-S

Based on the success of [our work understanding cyclic warmup-stable-decay schedules (WSD-S)](https://arxiv.org/abs/2410.05192), we decided to use a
WSD-S learning rate schedule. WSD-S is essentially cyclic warmup-(stable-decay)`*`, where you do a warmup, then a long
plateau at peak learning rate, and then a decay phase, repeating the stable and decay phase. The decay phases allow you
to measure how the model is performing.

WSD-S has a number of appealing properties:

- You can run at a very high learning rate for as long as you want, without needing to pre-register cooldowns.
- We could periodically do a rapid cooldown to get a sense of model performance without (completely) "wasting" FLOPs.
- WSD-S was shown to converge just as well as the more standard cosine schedule at any given point, meaning we could potentially use the checkpoints as fodder for scaling law analysis.

For the first 200K steps, we did a decay cycle every 10k steps for 1k steps.
However, I (@dlwh) got worried that we weren't seeing significant improvement in evaluation losses.
We then moved to a decay cycle every 20k steps for 2k steps, which led to a significant improvement in some eval losses,
but not all. (See below.)

We ended up moving away from WSD-S after this phase, for reasons to be detailed later.

### Other hyperparameters

We used a sequence length of 4096 and a batch size of 1024 * 4096 = 4Mi tokens.

The [DCLM paper](https://arxiv.org/abs/2406.11794) also showed that you could run fairly "hot", and we followed their example.
At 1e-3, our LR was roughly 3x higher than [Olmo 2 7B](https://arxiv.org/abs/2501.00656)'s 3e-4, and, with WSD-S, we were running at peak LR for 90% of steps.
We used a weight decay of 0.05. These are roughly consistent with Table 12 of the [DCLM paper](https://arxiv.org/abs/2406.11794).
(They recommend 2e-3, but we encountered instability at that rate.)

We initially opted to not use Z-loss because we didn't have problems with LR=1e-3. In retrospect, we should have used it and have since made it the default for future Marin runs.

### Specification

The specification for the first phase is available on [Github here](https://github.com/marin-community/marin/blob/852c53f9741b233549daf9f1649fe88c9c5a170c/experiments/tootsie/exp600_tootsie.py#L51-L78).

### Notes

#### Stability

Training was fairly stable with very few spikes.

#### WSD Cycle Change

At step 200k, we had a hypothesis that longer cooldowns would show more progress in terms of eval losses.
We had been doing a decay every 10k steps for 1k steps, and we switched to a decay every 20k steps for 2k steps.
That is, we still spent 10% of our steps in a decay phase, but each decay phase was twice as long, giving the model more
time to consolidate its progress.

Visually, the schedule now looked like this:

![graph depicting the new spacing of decay phases](../images/tootsie-8b-retro-wsd-interval.png)

As expected, this led to a drop in eval losses that most looked like our training data. In the below, the orange line is
eval loss (Paloma's c4en), while the blue line is training loss.
There is a noticeable drop in both the eval loss and the training loss during the decay phase.

![graph depicting the drop in eval loss and training loss with the longer, less frequent decay phases](../images/tootsie-8b-wsd-s-loss-transition.png)

Interestingly, not all eval losses dropped.  In fact, for some domains, the eval loss increased.
We saw decreases in `mc4`, `c4en`, `m2d2 wikipedia`, `m2d2 s2orc`, and `refined web`, but marked increases in `100_subreddits`, `twitterAAE_HELM_fixed`, `manosphere`, `4chan`, among a fwe others.
Interestingly, after this initial increase, almost all domains started to decrease again.
Subsequent analysis revealed that this was due to structural differences in preprocessing between the domains: some Paloma domains had texts that obligatorily ended with a space character (which we did not strip), which was not how our training data was formatted. The deeper cooldown allowed the model to dislike these final spaces more. We did a more thorough analysis [here](https://github.com/marin-community/marin/issues/826#issuecomment-2696496271).

![graph depicting many eval losses after the transition to longer decay phases](../images/tootsie-8b-wsd-s-losses-post-transition.png)


## Phase 2: Ocelot (DCLM EMA Phase)

At around 2.7e12 tokens, we were given access to a v4-2048 reservation and immediately moved to that.
All subsequent phases were run on that.

### Adjusted Hyperparameters

To better utilize the hardware, we increased our batch size by 3x, to 12Mi tokens at around 2.77e12 tokens.
Following [this blog post by Sadhika Malladi](https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/), we increased the learning rate to 1.7e-3,
which is approximately the old learning rate multiplied by $`\sqrt{3}`$ (the square root of the batch size increase).

We also switched from WSD-S to WSD, using the exponential moving averaging (EMA) of weights for monitoring evaluation performance. We did this following the [Deepseek V3 paper](https://arxiv.org/pdf/2412.19437v1). We used a $`\beta`$ of 0.995.
Initially, I inadvertently got the direction of the EMA wrong, so early evals were not substantially different from the "hot" model. Oh well.

We did not reset the optimizer state or do a rewarmup in this or any transition.

To my continued embarrassment, I also realized that we were using Llama 2's settings for Rotary Embeddings rather than Llama 3's settings. We changed to Llama3-style at this point.

### Specification

The specification for the second phase of training is [here](https://github.com/marin-community/marin/blob/852c53f9741b233549daf9f1649fe88c9c5a170c/experiments/tootsie/exp600_tootsie.py#L81-L116).

### Notes

#### Stability

Despite the changes, training was still fairly stable with very few spikes.
We saw a brief spike near the beginning which we attribute to the change in rotary embeddings, though we did not investigate this further.

#### The EMA Gap

One of the most interesting things we saw during this phase was what we call the "EMA gap."
The EMA gap is the difference between the eval loss of the EMA model and the eval loss of the hot model. As expected, the EMA loss was better.
Surprisingly, the gap was fairly stable over time, changing only with changes to the learning rate (or different datasets). For c4en, with the hot learning rate, the gap was consistently around 0.015 bits-per-byte (bpb). A stable gap of 0.017 was observed in the subsequent Phoenix phase. The EMA gap would of course shrink as the learning rate was cooled down (since, in the limit, the model freezes).

This was not to say there was no deviation in the gap, but the fact that it did not seem to increase or decrease with time was surprising.

![embed](../images/tootsie-8b-ema-gap.png)


### Interlude: Micro-Annealing

Following recent work on midtraining (e.g. [Olmo 2](https://arxiv.org/abs/2501.00656), [Yi](https://arxiv.org/html/2403.04652v1), [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)), we knew we would need to mix in higher quality data during our cooldowns. What constitues higher quality data?

Llama 3, Olmo 2, and DBRX (XXX is this right?) used small experiments (what Olmo calls "microannealing") to test the effect of different data sources. The basic idea is to take a model that has already been mostly trained, and then do a short cooldown with ~70% original data and ~30% a test high quality data source. We ran a series of micro-annealing runs to test the effect of different data sources.

See [GH#784](https://github.com/marin-community/marin/issues/784) and [GH#820](https://github.com/marin-community/marin/issues/820) for all experiments and more details on our approach.

The most important takeaway was that naively oversampling "High Quality" (HQ) data does not lead to improved task performance in
microannealing experiments, though they do usually lead to improved loss on HQ validation sets (e.g. Paloma's various subsets).

We believe this is because typical "high quality" data sources (e.g. ArXiv, Wikipedia) don't have as much fewshot-learning-inducing data (e.g. mulitple choice questions) as the broader web does. When you replace such a large fraction of the PT mix with a HQ source, you lose out on the multiple choice data that is present in the web.

In fact, we found that nothing led to improved task performance in microannealing experiments compared to the control (of 100% PT mix)... until we mixed in FLAN into all microannealing runs. (XXX is this right)
Instead of the 70/30 recommended in the Llama 3 paper, we found that 70% PT/ 15% FLAN/15% HQ led to the best results for our experiment budget.

By doing this, we were able to improve both the loss and the task performance of most microannealing runs. Ironically, notwithstanding the above, only 70% PT/30% FLAN underperformed the 100% PT control.

We also found that a microannealing experiment with all HQ sources performed approximately average compared to the other microannealing runs: in this setting, there was no advantage to variety.

## Phase 3: Jellyfish (First Cooldown)

At around 3.7T tokens, we were running low on DCLM tokens, which meant we needed to change something. We decided to try a cooldown.

### Data Mix

Based on the above, we decided to mix in a mixture of 70% high quality "web" (i.e. Dolmino's DCLM HQ and [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata)), and 30% a modified Dolmino + [FineMath 3+](https://huggingface.co/datasets/HuggingFaceTB/finemath).

Specifically, we included all sources from our ablations that outperformed the 100% PT control, meaning everything we tried at this phase except Dolmino FLAN.

| Dataset               | Percentage |
|-----------------------|------------|
| Dolmino DCLM HQ       | 67.8% |
| Dolma peS2o           | 10.8% |
| FineMath 3+           | 6.3% |
| Dolma Arxiv           | 5.2% |
| Dolma StackExchange   | 3.2% |
| Starcoder             | 2.2% |
| Dolma Algebraic Stack | 2.1% |
| Dolma Open Web Math   | 0.9% |
| Dolma Megawika        | 0.8% |
| Dolma Wikipedia       | 0.7% |


The HQ sources were weighted roughly proportional to token count and then upweighted to be 30% of the total.

The main deviations from the Dolmino mixture were:

- We included datasets that [Olmo 2](https://arxiv.org/abs/2501.00656) used in its Phase 1 (e.g. wikipedia) that we did not.
- We did not include [FLAN](https://arxiv.org/abs/2109.01652).
- We did not include the other synthetic math datasets in Dolmino. (This was a mistake.)
- We added [FineMath 3+](https://huggingface.co/datasets/HuggingFaceTB/finemath).

### Learning rate schedule

We decayed the learning rate from 1.7e-3 to 1.7e-4 over 1e12 tokens (79500 steps at 12Mi tokens/step). We used a linear decay schedule.

### Results

Results for this model are pretty good for "base model" tasks, though predictably not great for math and instruction following.

| Task | Score (%) |
|------|-----------|
| MMLU (5-shot) | 65.3 |
| MMLU (0-shot) | 62.5 |
| BBH (few-shot CoT) | 57.0 |
| GSM8K (8-shot) | 50.9 |
| HumanEval (pass@1) | 24.4 |
| MATH (4-shot) | 18.5 |
| IFEval (prompt-level loose) | 9.2 |

A 5-shot MMLU of 65.3 was better than both Olmo 2 7B and DCLM, and not too far behind Llama 3.1 8B. But we can do better.

### Notes

#### C4 EN Perplexity

Interestingly, this mix led to a large increase in Paloma c4en loss:

![embed](../images/tootsie-8b-ema-gap.png)

We haven't investigated this, but our hypothesis is that there are more structural differences in the formatting between DCLM HQ and c4en than there were between DCLM Baseline and c4en.

It is worth noting that we did not observe this increase in our largely similar final cooldown (which used Nemotron CC instead of Dolmino's DCLM HQ).


## Interlude: "Dessert" Runs

While our "mainline" run continued in the Phoenix phase, we ran a number of short ablations
consisting of what we termed "dessert" runs.
Dessert coasted at the same already low LR, with the goal of patching a few gaps in capabilities: we added the synthetic math datasets from Dolmino that we had excluded from our main run and added FLAN. See [this comment in GH#600](https://github.com/marin-community/marin/issues/600#issuecomment-2704462502) as well as [this one](https://github.com/marin-community/marin/issues/600#issuecomment-2704521062) for more details.

- Adding the math datasets improved GSM8K and MATH somewhat. GSM8K went from 0.509 to 0.611, and MATH went from 0.184 to 0.211. Not enough to get excited about, but a positive sign for later.
- FLAN didn't help final MMLU 5-shot (.653->651) or other tasks, at least not in this regime.

So "dessert" was seemingly a failure, though we revisited this in the final phase.


## Phase 4: Phoenix (Reheated)

The specification for this phase is available [here](https://github.com/marin-community/marin/blob/852c53f9741b233549daf9f1649fe88c9c5a170c/experiments/tootsie/exp600_tootsie.py#L465-L522).

After the cooldown, at around 4.7T tokens, we had more time for training, so we decided to keep going. We rapidly rewarmed the model and transitioned our mixture to [Nemotron-CC](https://arxiv.org/abs/2412.02595) (plus some code and math).

### Data

Because we were running out of DCLM, we transitioned the data from our phase 1 and 2 mixture (DCLM+Starcoder+ProofPile 2) to a new mixture that was [Nemotron-CC](https://arxiv.org/abs/2412.02595) and [Starcoder](https://huggingface.co/datasets/bigcode/starcoderdata).
(We didn't worry too much about epoching StarCoder.)

The target mixture was Nemotron-CC (with each subset weighted approximately proportionally to token count) and Starcoder,
also weighted by token count. As a transition, we weighted all components
(DCLM, Starcoder, Proofpile, and Nemotron's subcomponents) approximately proportional to token count, which we used for 2000 steps (about 25.2e9 tokens), after which we switched to the target mixture.

### Learning Rate Schedule

We rewarmed the learning rate linearly from 1.7e-4 to 1.7e-3 over those same 2000 steps after which we
held the learning rate fixed for the duration of this phase.

### Notes

The most interesting thing about this phase was that the transition went very smoothly. We expected a substantial loss spike, but didn't see one. The loss did jump, but returns to a slighly lower level than where the run was before the cooldown.

![graph depicting the loss during the transition to the new data mix. The loss very briefly spiked but then returned to lower levels than before the firstcooldown.](../images/8b-phoenix-transition.png)

Also of small interest is that the steady state c4en loss for Phoenix was considerably lower
than DCLM. This need not mean anything other than structural similarities in the preprocessing.
We have not investigated this further.


## Interlude: Deeper Cooldowns

While Phoenix was running, we also ran a series of "deeper" cooldowns, trying to improve the model's SFT-ability. This led to the [Raccoon](https://github.com/marin-community/marin/issues/898) and [Spoonbill](https://github.com/marin-community/marin/issues/916) series of runs.

### Raccoon: Debugging SFT-ability

The genesis of Raccoon was that we were having trouble getting our model to SFT well. While our nascent SFT pipeline would exhibit an "epoch cliff" (i.e. loss would jump down on each repeated epoch) for Llama 3.1 8B and Olmo 2 7B, [it didn't do that for our model](https://github.com/marin-community/marin/issues/897), and indeed didn't perform that well. We hypothesized that this was because our "cooled down" LR was still quite high: 1.7e-4, which was closer to Olmo 2's peak LR than their final LR.
We thought this might have something to do with our struggles with SFT-ability.

Starting from the jellyfish cooldown, we further cooled down the model from 1.7e-4 to 1.7e-5 over about 100B tokens. ([WandB report here](https://wandb.ai/stanford-mercury/marin/reports/898-Tootsie-Soft-Raccoon--VmlldzoxMTk3NjUwNg?accessToken=06f87pmmvhdulczenkg3349jxk7e1pwbd4pdci2i8wvyxg9289122gfnckr9ymwc))
Things looked good at first, but as the cool down deepened the the training loss started slowly creeping up?!?
Divergence we could understand, but a slow increase in training loss we couldn't.

![graph depicting the slow increase in training loss during the cooldown](../images/8b-raccoon-loss-increase.png)

We tried a number of things to debug this:

- Resetting the optimizer state (maybe something was wonky?)
- Removing weight decay (maybe the gradient signal was too weak?)

Nothing solved the problem.

(That said, SFT performance did get better, so that was good!)

What?

### Spoonbill: Z-loss to the rescue

[WandB report here](https://wandb.ai/stanford-mercury/marin/reports/916-Tootsie-Hypnotic-Spoonbill--VmlldzoxMjA1NjU2Nw)

With [Spoonbill](https://github.com/marin-community/marin/issues/916), we tried decaying to 3e-5 rather than 1.7e-5. 3e-5 was Olmo 2's final LR, and in Raccoon we were still seeing decreases at 3e-5. We also threw in some [Tulu v3 data](https://arxiv.org/abs/2411.15124) (.3% of the data) and FLAN (1%) to adjust the data schedule and generally improve SFT-ability.

Alas, the training loss still crept up.

What?

![graph depicting the slow increase in training loss during the cooldown for spoonbil](../images/marin-8b-spoonbill-loss.png)

We reran with extensive norm tracking and finally isolated the problem: the `lm_head` was exploding:

![graph depicting the norm of the lm_head during training compared to some other params](../images/8b-spoonbill-norms.png)

We fixed this by adding a z-loss penalty of 1e-4 on the final logits.

And it worked!

![graph depicting the loss during the cooldown for spoonbill](../images/8b-spoonbill-zloss.png)

Interestingly, [subsequent experiments](https://wandb.ai/marin-community/marin/reports/ZLoss-vs-Not-1-4B--VmlldzoxMjEzMzA1NA) showed that, when starting from scratch, zloss shrinks the final layer norm, and typically **increases** the lm_head.

So, zloss, it's not just for avoiding explosions.


## Phase 5: Starling (Second Cooldown)

At around 11.1e12 tokens, we decided to start another cooldown. Building on lessons from our previous cooldowns, we made the following changes:

- We deepened the cooldown to 1.7e-5, rather than 1.7e-4.
- We added a small z-loss penalty of 1e-4.
- We increased the batch size to 16 Mi tokens rather than 12Mi.

### Data

We switched to a mix that was 70% [Nemotron-CC](https://arxiv.org/abs/2412.02595) and 30% high-quality sources, including some new sources we created.
Nemotron-CC's components were weighted according to compressed bytes (which is roughly proportional to token count).
Within the high quality datasets, we weighted them roughly proportional to token counts multiplied by an oversampling ratio we set more or less arbitrarily.

We included the following datasets:

| Dataset                           | Proportion | Oversampling |
|-----------------------------------|------------|--------------|
| NemoTron CC Medium                | 22.1% | 1x |
| NemoTron CC HQ Synth              | 17.8% | 1x |
| NemoTron CC Medium Low            | 10.1% | 1x |
| NemoTron CC HQ Actual             | 6.0% | 1x |
| NemoTron CC Medium High           | 5.4% | 1x |
| NemoTron CC Low Actual            | 4.6% | 1x |
| NemoTron CC Low Synth             | 4.1% | 1x |
| Marin Arxiv Markdownified         | 5.2% | 5x |
| Dolmino PES2O                     | 5.2% | 5x |
| Starcoder Data                    | 4.5% | 1x |
| Proofpile 2                       | 4.5% | 1x |
| Finemath-3-Plus                   | 3.0% | 5x |
| Dolmino FLAN                      | 3.0% | 10x |
| Dolmino StackExchange             | 1.5% | 5x |
| Marin StackExchange Markdownified | 1.5% | 5x |
| Dolmino Math                      | 0.8% | 10x |
| Marin Wikipedia Markdownified     | 0.3% | 5x |
| Dolmino Wiki                      | 0.3% | 5x |
| Marin Datashop Science QA         | 0.1% | 5x |

Here "Dolmino Math" refers to most of the math-y components of Dolmino:

- [CodeSearchNet](https://arxiv.org/abs/1909.09436) (with OWM Filter)
- [GSM8K](https://arxiv.org/pdf/2110.14168v1)
- [MetaMath](https://arxiv.org/abs/2309.12284)
- [Dolmino SynthMath](https://arxiv.org/abs/2501.00656)
- [MathCoder2 Synthetic](https://arxiv.org/abs/2310.03731)
- [Dolmino TinyGSM-MIND](https://arxiv.org/abs/2501.00656)
- [Dolmino Tulu Math](https://arxiv.org/abs/2501.00656)

### Learning Rate Schedule and Other Hyperparameters

We linearly cooled down the learning rate schedule from 1.7e-3 to 1.7e-5 (sic) over approximately 1.34e12 tokens
(80,000 steps).

Based on our findings in the "Racooon" and "Spoonbill" deep cooldowns (where we observed steady loss increases at low
learning rate), we imposed a z-loss penalty of 1e-4 on the final logits.

We also increased our batch size to 16Mi tokens. This change was to further reduce gradient variance and better utilize the hardware (i.e. slightly more MFU).


### Bonus: Starling Dessert

Because we were still seeing steady log-linear loss decreases (in c4en and elsewhere) even at this very low LR, we decided to
coast a little longer. We ran a dessert run from 12.4e12 to 12.7e12 tokens, with the learning rate fixed at 1.7e-5
and otherwise using the same hyperparameters as Starling.

### Notes

#### C4 EN Perplexity

![graph depicting the c4en perplexity during the starling cooldown phases](../images/marin-8b-starling-c4en.png)

Interestingly, c4en perplexity decreased a lot during this cooldown, when in the previous cooldown it had increased.
Again, we attribute this to structural differences in the preprocessing of Dolmino's DCLM HQ and Nemotron CC.

#### WSD-S Hill / River decomposition

Separately, the slope in loss decrease during the dessert phase slowed down dramatically.
This is consistent with the theory around [WSD-S](https://arxiv.org/abs/2410.05192), which decomposes the loss into a "river" and "hill" component, where the hill is the loss due to flucuations due to the learning rate (and stochasticity in SGD).

As the learning rate is decayed, the hill component diminishes. Once we stop decaying the learning rate, there is no further decrease in the hill component, so progress slows down.

![graph depicting the slowdown in loss decrease during the cooldown](../images/tootsie-8b-starling-loss-slowdown.png)


### Results

XXX

GSM8K 0.5094768764215315 -> 0.67854
MATH 0.1846 -> 0.3526
HumanEval 0.24390243902439024 -> 0.5731707317
IFEval 0.09242144177449169 -> 0.4953789279
BBH 0.5702657041929043 -> 0.003532483489479343


## Main Takeaways

- **Tootsie means never having to say you're sorry.** We made several dramatic changes to the data mix, optimizer, and other hyperparameters during the run. We cooled down, rewarmed up, changed the mixture, etc. without any major issues. Highly recommend.
- **Z-loss isn't just for avoiding explosions.** While we didn't need z-loss to stabilize the training, we found it actually pretty necessary during very deep cooldowns.
- **Format diversity is useful.**  While not necessarily showing up in non-loss evals, the high sensitivity of c4en perplexity (and that of other datasets) to the data mix suggests that formatting diversity is useful. We will pursue this further in future runs.
- **So-called "high quality" data doesn't have everything.** High quality data seems to often lack data that leads to good few-shot performance. Mixing in FLAN or other datasets seems to help.
- **Exponential moving averages are nice, but maybe not all that useful with WSD.** The stability of the EMA gap suggests that there isn't that much information in monitoring EMA model evals: we know from WSD that there is some easy performance to be had whenever we want it (by cooling down and removing the hill component). That said, even at very low LR there is still *some* gap, so that's a tiny bit of free performance. The harm is also small, since the EMA can be held in CPU memory rather than HBM.
- **Tootsie means having to say you're sorry kind of a lot.** We made many, many mistakes during the run. Some of the changes were actually unintentional, or were fixes to initial mistakes (e.g. the rotary embedding hyperparameters). Nevertheless, the end result was a model that performed well on a wide variety of tasks.


## Future Work

Future work will focus on:

- **Bigger models.** We have a 32B model that is performing quite well at 1T tokens.
- **SFT/RL.** We're in the middle of SFTing Marin 8B. (TL;DR is that it works pretty well at AlpacaEval, but there is an unacceptable degradation on tasks like MMLU.) RL is the obvious next step.
- **Format diversity.** We will continue to pursue formatting diversity as a way to improve performance on tasks like MMLU.
