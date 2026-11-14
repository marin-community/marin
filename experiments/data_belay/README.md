# Data Belay

The Data Belay tests whether a Hugging Face dataset improves Marin's held-out evals. It gives the dataset a share
`p` of the training tokens of the fast-track dense model (`experiments/grug/fast_track/launch.py --dense`, at its
baseline budget of 20 tokens per parameter) and compares the result with a control that trains on the same number of
tokens, all from the base cache.

| Arm | Training tokens at rung budget `B` |
|---|---|
| control | `B` tokens of the base cache `hero_tok/v16384_shuf/train` |
| treatment | `(1 - p)·B` base tokens and `p·B` tokens of the candidate |

The control depends only on the rung and the seed, so every candidate reuses the same control runs.

## Commands

Candidate caches live under the `MARIN_PREFIX` that the training cluster (`cw-us-east-02a`) sets in its Iris config,
so they sit beside the base cache. Commands that load the tokenizer on your machine (`check`, and `prepare`
without `--submit`) also need `MARIN_PREFIX` set to that prefix.

```bash
# 1. Intake: pin the head commit, check the license, estimate the size and epochs at each rung.
uv run data-belay check --dataset HuggingFaceTB/finemath --config finemath-4plus --share 0.2

# 2. Tokenize a seeded sample of the dataset's files into a cache (an Iris CPU job). The job prints the cache path
#    and the candidate's label, for example finemath-4plus-p20-<hash>.
uv run data-belay prepare --dataset HuggingFaceTB/finemath --config finemath-4plus --share 0.2 --submit
#    Or one candidate from the normalized output of one or more Datakit registry sources:
uv run data-belay prepare-datakit --label dk-<name> --source <registry source> --share 0.2 --submit

# 3. Train each seed of the treatment. --mix-cache is the cache path that prepare printed, not the Hugging Face
#    dataset. The run id is dbelay-main-<rung>-<label>-s<seed>, and the share is the one the cache was prepared for.
#    The control (no --mix-cache) runs once per rung and seed, for the seeds in results.CONTROL_SEEDS.
uv run data-belay train --rung d512 --seed 0 \
    --mix-cache s3://marin-us-east-02a/marin/data_belay/caches/<...> --submit

# 4. Once the runs finish, score their final checkpoints' choice loss (an Iris job on one H100).
uv run data-belay score --rung d512 --submit dbelay-main-d512-<label>-s0 dbelay-main-d512-<label>-s1 \
    dbelay-main-d512-<label>-s2

# 5. Verdict against the shared control, or against another candidate at the same seeds and share.
uv run data-belay report --rung d512 --candidate <label> --seeds 0,1,2
uv run data-belay report --rung d512 --candidate <filtered label> --against <unfiltered label> --seeds 0,1,2
```

`train` launches `experiments.grug.fast_track.launch --dense` on `cw-us-east-02a` at Iris priority `batch`, and it
refuses a run id that W&B already has. Runs write under `tmp/ttl=30d/data_belay/runs/` in the cluster's
`MARIN_PREFIX`. `score` refuses a run that has not finished training, and must run before the 30-day TTL deletes the
checkpoints. It writes one JSON per run under `data_belay/choice/` in
the training cluster's `MARIN_PREFIX` (`s3://marin-us-east-02a/marin/data_belay/choice/`), outside the TTL root, and
`report` reads it from there.

Under the verdict, `report` prints how much of each arm's data its runs read: the tokens, the epochs of its cache, and
the fraction of the dataset seen. It takes the share, the cache and the tokens trained from the runs' W&B config, and
the dataset's size from the cache's manifest, so it needs the manifest as well as the scores. The ladder line gives
the epochs at each rung, since a gain that falls off as the epochs pass 1 reads differently from one that falls off
below 1. The share is the runs' too: a candidate is judged against the dilution references whose runs trained at its
share, and `--against` refuses two arms at different shares. `--share` takes a whole percent, so that the label of the
cache it prepares (`p20`) names the share exactly.

`check` allows the licenses on the Blue Oak Council's [list of permissive licenses](https://blueoakcouncil.org/list)
(every rating, version 16) and OpenMDW. It rejects NC and ND licenses, and refuses any other license for a person to
review: a data license such as CC BY, CC BY-SA or ODC-By, which the list does not cover, `other`, a missing license,
or a second license that is not on the list. It also refuses a share that reads the dataset more than 4 times at the
largest rung. It sizes the head commit only, because the row count comes from the datasets-server, which sizes the head.
`prepare` checks the epochs again on the cache it built, and writes the manifest that `train` reads only when they
pass. `prepare-datakit` records the licenses on the sources' Hugging Face cards and does not refuse one off the
allowlist, since the Datakit registry has already admitted the sources.

## Metrics

Every run logs, at its end:

| Group | W&B key |
|---|---|
| Paloma | `eval/paloma/macro_bpb` |
| Uncheatable | `eval/uncheatable_eval/macro_bpb` |
| Eval Policy knowledge (MMLU-Pro, TriviaQA, TruthfulQA) | `eval/eval_policy/knowledge/macro_bpb` |
| Eval Policy commonsense (BoolQ, PIQA, WinoGrande) | `eval/eval_policy/commonsense/macro_bpb` |
| Eval Policy math (GSM8K, MATH-500) | `eval/eval_policy/math/macro_bpb` |
| Eval Policy code (HumanEval+, MBPP+, CRUXEval-O) | `eval/eval_policy/code/macro_bpb` |

The Eval Policy groups are the BPB of the gold answer after the prompt, on the non-agentic benchmarks of the Marin
Eval Policy (marin#10197): small ladder models score near chance on the benchmarks themselves. The caches are
validation steps in [`experiments/datasets/eval_policy_answer_bpb.py`](../datasets/eval_policy_answer_bpb.py), which
the launcher adds with `--eval-policy-answer-bpb` and builds on first use. GPQA is left out because its dataset is
gated. The verdict averages each Eval Policy group's
task BPBs (`eval/eval_policy/<group>/<task>/bpb`) and leaves out BoolQ, which is logged: its one-token yes/no answer
varied 10.7% between control seeds.

For the multiple-choice knowledge tasks, MMLU-Pro and TruthfulQA (MC1), the verdict uses choice loss instead: the gold
option's loss relative to the alternatives, `−log2 softmax(ll)[gold]`, where `ll` is each option's log-likelihood
summed over its tokens, averaged over questions. Answer BPB scores only the gold answer, so it misses a gain in the
gold answer's margin over the distractors (see the accuracy check). TriviaQA, a generative task, keeps answer BPB.
Knowledge averages each of its three tasks relative to the control's mean at the rung, since the units differ.
PIQA and WinoGrande keep answer BPB: in the pilot, which scored them with the Eval Policy prompts, their choice loss
varied about 4 times as much between control seeds.

## Verdict

For each group, `Δ` is the treatment's mean over a baseline's, minus 1, with a Welch t interval. The gain is `Δ`
against the control. The harm is `Δ` against the dilution references at the share that the candidate's runs trained
at (`results.DILUTION_REFERENCES`): four Datakit sources at 20% whose text no eval measures, in scripts the English
evals do not share (ClimbLab Japanese, and FinePDFs Chinese, Thai and Arabic). Giving up 20% of the base mix costs
every group 1.3% to 3.6% on its own at d512 and d768, past every group's harm limit, so the harm is what a candidate
costs beyond that. The reference arm holds one value per reference, the mean of its seeds, because the references
differ from each other by more than their seeds do. Only 20% has references so far. `report` refuses a candidate at
another share, unless `--against` is given, because its harm against the control would include what giving up the
share costs. The refusal prints the commands that build the references at that share.

Use `--against` when the question is A instead of B at one share: a filtered dataset against its source, or two
candidates for one place in the mix. Both the gain and the harm are then against B's runs, since both arms give up
the same share: `better` means A beats B, and `worse` means A costs more than B past the harm limit. Use the
default, against the control, for a dataset that has no B: the gain is what it adds over the base mix despite giving
up the share, and the harm is what it costs beyond the dilution references.

A group is `better` (gain interval below 0, and a gain of at least 1%), `worse` (harm interval above its harm limit),
`neutral` (harm interval below the limit) or `unresolved`. The harm limit is 1%, or 2.5 seed SDs of the d512 control
where the seeds vary more. `report` computes the limits from the current d512 control seeds, so they follow a change
of model or base mix, and `noise --rung d512` prints them. The 1% floor holds for gains too. With the
interval alone, a null arm's d512 rung reads yes 13.8% of the time whatever the seeds' spread (4,000 simulated arms);
with the floor, 1.2% at this model's seed spread and 5.7% at the vanilla Track's. This model's null arm had math at
−0.56% [−1.09%, −0.03%] at d512, `helps` without the floor. On the vanilla Track the floor changed one of the Datakit
sweep's 183 answers (Nemotron's wiki rewrite, from mixed to stop) and none of the validation arms' verdicts.

| Verdict | Rule |
|---|---|
| `helps` | a group is better, and every other group is better or neutral |
| `trade-off` | a group is better, and another group may be past its harm limit |
| `hurts` | a group is worse |
| `no gain` | no group is better and none is worse |

`uv run data-belay noise --rung d512` prints the control's seed spread. See [`judge.py`](judge.py).

## Ladder

A candidate climbs d512, d768, d1024, up to the largest rung its cache supplies within 4 epochs:

| Verdicts so far | Next step |
|---|---|
| any one rung | run the next rung |
| no at two consecutive rungs | stop: the answer is no |
| no at two consecutive rungs below the top, with some group's `Δ` against the control below 0 at the second | run the next rung |
| a yes at either of the last two rungs, below the top | run the next rung |
| yes at d768 and d1024 | the answer is yes |
| d768 and d1024 disagree | mixed: report both |

A yes is `helps` or `trade-off`, a gain on one of the evals; a no is `hurts` or `no gain`. `report` prints the step
when the candidate's runs at the rungs below share its label, and `train` refuses a rung that reads its cache more
than 4 times. One rung is not enough either way: at 20%, OpenCoder's web code was `helps` at d512 (Paloma −1.1%) and
`no gain` at d768 and d1024, while FineMath's math gain grew from −12.2% at d512 to −16.7% at d1024. A gain inside
its interval at one rung can clear it at the next: of 49 Datakit sources stopped at d768 that ran on to d1024, 6 were
trade-offs there, and each had its gaining group below 0 at d768. Running on costs runs only, since a yes still
needs both top rungs. A null arm, which gives up no share, has some group below 0 at d768 most of the time, so
it usually runs on. In 4,000 simulated null arms none ended yes at this model's seed spread, and 0.3% at the
vanilla Track's (1.1% at either without the 1% floor on a gain).

## Model

Runs train main's fast-track dense model. Until 2026-11-12 they trained the dense model of Belay's `vanilla` Track,
on the `marin_belay_v1_recipe` branch: the validation runs below and the Datakit sweep, with run ids
`dbelay-<rung>-…`. Both models read the same base cache with the same tokenizer, so the candidate caches and the
dilution references carry over. The fast-track model adds GatedNorm, short convolutions, and sliding-window attention
(one KV head at d512 and d768), and trains with MuonH. Its budget is 690 steps at d512 and 2,040 at d768, against
720 and 2,161.

Five validation arms reran on it at d512 and d768, and the null arm at d512 (`dbelay-main-…`). Each arm is judged
with seeds 0–2 on both models, so the 6-seed vanilla arms below count their first 3 here:

| Arm | d512, vanilla | d512, fast-track | d768, vanilla | d768, fast-track |
|---|---|---|---|---|
| FineMath-4plus, 20% | helps | helps | helps | helps |
| FineWeb-Edu, 20% | trade-off | helps | helps | helps |
| FineWeb, 20% | no gain | trade-off | helps | helps |
| Cosmopedia web samples, 20% | no gain | no gain | no gain | no gain |
| FineWeb, 50% | hurts | hurts | hurts | hurts |
| Null: control seeds 6–8 | no gain | no gain | no gain | not run |

FineWeb at 50% was judged with its harm against the control, which `report` now refuses at a share without
references. Its `hurts` comes from math and code, +3.5% and +8.6% against the control at d512, and part of that may
be what giving up half the base mix costs on its own.

The gains agree: across the 30 pairs of arm and group at each rung, the two models' `Δ`s against the control
correlate at 0.99, and where their signs differ both are within 1%. The one rung that turns from no to yes, FineWeb
at 20% at d512, has the gains that the vanilla model shows at d768 (Paloma −1.5%, commonsense −2.8%). The fast-track
model's control seeds vary less. At d512, over seeds 0–5, the SDs are Paloma 0.11% (vanilla 0.38%), Uncheatable
0.19% (0.46%), knowledge 0.31% (0.65%), commonsense 0.34% (0.39%), math 0.48% (0.84%) and code 0.75% (1.04%). The
limits from these SDs are 1% for every group but math (1.19%) and code (1.88%), against the vanilla Track's 1.0% to
3.0%, and change no verdict at d512 or d768. d1024 has no fast-track controls or dilution references yet; the first
candidate to reach d1024 needs them trained (3 control seeds, and 3 seeds of each reference).

## Validation runs on the vanilla Track (2026-11-06 to 2026-11-10)

| Candidate | Rungs | Answer |
|---|---|---|
| FineMath-4plus, 5% | helps, trade-off, trade-off | yes |
| FineMath-4plus, 20% | helps, helps, helps | yes |
| FineMath-4plus, 50% | trade-off, hurts, hurts | no |
| OpenWebMath, 20% | helps, helps, helps | yes |
| OpenCoder synthetic code, 20% | trade-off, trade-off, trade-off | yes |
| OpenCoder web code, 20% | helps, no gain, no gain | no |
| FineWeb-2 Chinese, 20% | no gain, no gain | no |
| FineWeb, 20% | no gain, helps, helps | yes |
| FineWeb, 50% | hurts, hurts, hurts | no |
| FineWeb-Edu, 20% | trade-off, helps, helps | yes |
| FineWeb-Edu, 50% | hurts, hurts, hurts | no |
| FineWeb-Edu against FineWeb, 50% | hurts, hurts, hurts | no |
| Cosmopedia web samples, 20% | no gain, no gain, no gain | no |
| Null: control seeds 6–8 | no gain, no gain, no gain | no |

The harm is against the dilution references for the 20% arms. It is against the control for the 5% and 50% arms,
whose shares have no references, for the null arm, which gives up no share, and for FineWeb-Edu against FineWeb.
Against the control alone, OpenCoder synthetic code was `hurts` at d768 and d1024 and its answer no: its Paloma and
Uncheatable costs at d768, +2.4% and +2.7%, are +0.8% and +0.6% against the references. Five arms stopped at d768
under the first ladder with a group below 0 there, and ran on to d1024: all five stay no, and the null arm stays
`no gain`. FineWeb at 20% ran after them, to separate FineWeb from the 50% share: it is a yes, and its
code cost at d1024, +1.95% against the control, is −0.3% against the references. At 50%, judged against the
control, code +7.2% made FineWeb `hurts`.

| Candidate, share | Rung, seeds | Paloma | Uncheatable | knowledge | commonsense | math | code | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FineMath-4plus, 5% | d512, 3 | −0.43% | −0.41% | −0.88% | −0.38% | −6.15% | −1.04% | helps |
| FineMath-4plus, 5% | d768, 3 | +0.14% | +0.13% | −0.78% | +0.51% | −7.89% | −0.27% | trade-off |
| FineMath-4plus, 5% | d1024, 3 | −0.08% | +0.01% | +0.37% | −0.16% | −8.30% | −1.61% | trade-off |
| FineMath-4plus, 20% | d512, 6 | +0.08% | +0.72% | +0.43% | +0.53% | −12.17% | −0.44% | helps |
| FineMath-4plus, 20% | d768, 3 | +0.13% | +0.46% | −0.29% | +0.58% | −14.79% | −1.86% | helps |
| FineMath-4plus, 20% | d1024, 3 | +0.33% | +0.73% | −0.54% | +0.29% | −16.70% | −4.01% | helps |
| FineMath-4plus, 50% | d512, 3 | +1.94% | +3.57% | +2.39% | +3.03% | −15.87% | +1.11% | trade-off |
| FineMath-4plus, 50% | d768, 3 | +1.43% | +2.71% | +0.89% | +1.87% | −21.18% | −1.44% | hurts |
| FineMath-4plus, 50% | d1024, 3 | +2.18% | +3.40% | +1.08% | +1.71% | −23.48% | −2.60% | hurts |
| OpenWebMath, 20% | d512, 3 | −0.51% | −0.57% | −0.09% | +0.83% | −6.48% | +0.89% | helps |
| OpenWebMath, 20% | d768, 3 | −0.25% | −0.57% | +0.49% | +0.85% | −7.68% | +1.15% | helps |
| OpenWebMath, 20% | d1024, 3 | −0.03% | −0.67% | −0.59% | +0.33% | −8.52% | −0.55% | helps |
| OpenCoder synthetic code, 20% | d512, 3 | +1.49% | +1.33% | +1.43% | +1.85% | +1.72% | −9.05% | trade-off |
| OpenCoder synthetic code, 20% | d768, 3 | +2.36% | +2.68% | +1.99% | +2.57% | +3.41% | −6.52% | trade-off |
| OpenCoder synthetic code, 20% | d1024, 3 | +2.15% | +2.44% | +2.75% | +1.72% | +2.03% | −5.31% | trade-off |
| OpenCoder web code, 20% | d512, 3 | −1.12% | −0.58% | −0.49% | −0.04% | +0.40% | +1.62% | helps |
| OpenCoder web code, 20% | d768, 3 | −0.82% | −0.66% | −0.76% | −0.65% | −1.01% | +1.74% | no gain |
| OpenCoder web code, 20% | d1024, 3 | −0.63% | −0.61% | −0.18% | −0.32% | −0.80% | −0.49% | no gain |
| FineWeb-2 Chinese, 20% | d512, 3 | +0.79% | +1.07% | +0.46% | +1.34% | +1.69% | +1.96% | no gain |
| FineWeb-2 Chinese, 20% | d768, 3 | +2.19% | +2.83% | +2.15% | +2.49% | +3.78% | +4.11% | no gain |
| FineWeb, 20% | d512, 3 | −0.91% | +0.31% | −0.46% | −1.58% | +2.34% | +4.78% | no gain |
| FineWeb, 20% | d768, 3 | −1.85% | −1.08% | −1.89% | −3.11% | +0.21% | +1.11% | helps |
| FineWeb, 20% | d1024, 3 | −1.29% | −0.63% | −1.32% | −2.30% | +0.45% | +1.95% | helps |
| FineWeb, 50% | d512, 6 | −2.43% | −0.01% | −2.71% | −5.13% | +3.69% | +7.78% | hurts |
| FineWeb, 50% | d768, 3 | −2.99% | −0.65% | −3.57% | −5.51% | +2.90% | +8.29% | hurts |
| FineWeb, 50% | d1024, 3 | −2.73% | −0.81% | −3.69% | −4.80% | +2.35% | +7.18% | hurts |
| FineWeb-Edu, 20% | d512, 3 | −0.53% | −0.21% | −2.56% | −1.28% | +0.56% | +4.02% | trade-off |
| FineWeb-Edu, 20% | d768, 3 | −0.60% | −0.60% | −3.24% | −1.86% | −0.24% | +2.13% | helps |
| FineWeb-Edu, 20% | d1024, 3 | −0.64% | −0.73% | −3.48% | −1.76% | −0.84% | +0.67% | helps |
| FineWeb-Edu, 50% | d512, 6 | −0.14% | +0.91% | −4.28% | −2.18% | +2.99% | +8.79% | hurts |
| FineWeb-Edu, 50% | d768, 3 | −1.06% | −0.46% | −5.71% | −3.66% | +0.77% | +7.98% | hurts |
| FineWeb-Edu, 50% | d1024, 3 | −0.74% | −0.47% | −6.33% | −3.44% | +0.77% | +7.36% | hurts |
| FineWeb-Edu against FineWeb, 50% | d512, 6 | +2.34% | +0.92% | −1.62% | +3.11% | −0.67% | +0.94% | hurts |
| FineWeb-Edu against FineWeb, 50% | d768, 3 | +2.00% | +0.19% | −2.22% | +1.96% | −2.06% | −0.29% | hurts |
| FineWeb-Edu against FineWeb, 50% | d1024, 3 | +2.05% | +0.34% | −2.73% | +1.43% | −1.54% | +0.17% | hurts |
| Cosmopedia web samples, 20% | d512, 3 | +1.20% | +1.21% | −0.44% | +0.27% | +3.61% | +4.23% | no gain |
| Cosmopedia web samples, 20% | d768, 3 | +0.71% | +0.84% | +0.85% | −0.09% | +2.34% | +4.00% | no gain |
| Cosmopedia web samples, 20% | d1024, 3 | +0.57% | +0.65% | −0.33% | −0.63% | +1.29% | +1.54% | no gain |
| Null: control seeds 6–8 | d512, 3 | +0.17% | +0.19% | +0.79% | +0.61% | +0.38% | −0.15% | no gain |
| Null: control seeds 6–8 | d768, 3 | −0.13% | −0.06% | +0.13% | +0.43% | −0.88% | −0.08% | no gain |
| Null: control seeds 6–8 | d1024, 3 | +0.09% | −0.07% | −0.06% | −0.04% | −1.02% | −0.16% | no gain |

Each `Δ` is against the control at the same rung (6 control seeds at d512, 3 at d768 and d1024), negative is better;
the knowledge column combines choice loss and TriviaQA's BPB. The
runs are `dbelay-<rung>-<candidate>-s<seed>` in W&B `marin-community/marin_moe`; the null arm is control seeds 6–8,
judged against the control seeds above.

Six arms tested the method, with expectations written down before their runs: the null arm, FineMath at 5% and 50%,
Cosmopedia, OpenWebMath, and FineWeb-Edu at 20%.

- **Null** (met): no group is `better` or `worse` at any of the three rungs. At d512 one of six group intervals (commonsense,
  +0.6%) lies above 0, about the rate a 95% interval allows, so one group's gain of about 0.5% at a single rung is noise.
- **FineMath dose response** (met): the math gain grows with the share at every rung (d1024: −8.3%, −16.7%, −23.5%),
  and so does the cost on Paloma and Uncheatable. 5% gets about half of 20%'s math gain at no cost elsewhere; 50% is
  `hurts`.
- **Cosmopedia** (missed): synthetic textbooks were expected to improve knowledge or commonsense, and no task in either
  group moved. Accuracy agrees (below), so this is a real no for this mix, at d1024 too.
- **OpenWebMath, 20%** (met): a second math source, expected to be a yes with a smaller math gain than FineMath's. It
  is a yes (math −6.5%, −7.7% and −8.5% at d512, d768 and d1024, about half of FineMath 20%'s).
- **FineWeb-Edu, 20%** (met): expected to be a yes with knowledge and commonsense `better`. It is a yes, `helps` at
  d1024, and the code cost of a 50% share (+8.0%) shrinks to +0.7%. Both groups are `better` at every rung. With answer
  BPB in place of choice loss, knowledge was `better` only at d1024 (−2.9%), with its interval crossing 0 at d512 and
  d768. These runs were the fresh test of choice loss, with the rule fixed beforehand.

## Accuracy check

`experiments.data_belay.accuracy` scores a run's final checkpoint on multiple-choice tasks, cloze style: `acc_norm`
picks the choice with the highest log-likelihood per byte. At d768, the control is above chance only on SciQ,
ARC-Easy and PIQA. Their mean `acc_norm`, with a Welch 95% interval against 6 control seeds (3 seeds per arm):

| Arm (d768) | SciQ, ARC-Easy, PIQA | Against the control (points) |
|---|---:|---|
| control | 51.3% | seed SD 0.7 |
| FineWeb-Edu, 50% | 53.9% | +2.6 [+1.8, +3.4] |
| FineWeb, 50% | 52.0% | +0.7 [−0.4, +1.7] |
| Cosmopedia web samples, 20% | 50.9% | −0.3 [−1.3, +0.6] |
| FineWeb-Edu against FineWeb | | +1.9 [+0.9, +3.0] |

Against the control, answer BPB sees FineWeb-Edu's gain too (knowledge −4.7%, commonsense −3.7%); its verdict is
`hurts` only because a 50% share displaces the base mix's code. Against FineWeb, answer BPB misses the Edu filter's
gain: accuracy is 1.9 points higher, while knowledge BPB is neutral and Paloma is 2.0% worse. Answer BPB measures the
gold answer's likelihood, not its margin over the distractors, so data that sharpens that margin without lowering the
gold answer's loss, as a quality filter on web text can, shows up in accuracy only.

Choice loss measures that margin, so knowledge now uses it for MMLU-Pro and TruthfulQA. Its rule was fixed before any
comparison. Against FineWeb it sees the Edu filter's gain (knowledge −1.6% at d512, −2.2% at d768, both `better`). Under
the first judge, which measured harm against the control, it changed none of the 33 rung verdicts, and the null arm
stayed `no gain`. Choice loss over per-byte
log-likelihoods failed: every task sat within 0.5% of chance.

Summed log-likelihood also penalizes long options. On MMLU-Pro, picking the most likely option scores 8.3% at every
rung, below the 10% chance rate, so the gold option tends to be longer than the distractors. Choice loss sits far
above chance there (26 bits at d512, against log2(10) = 3.3) and mixes the margin with answer length. Per task it
moves with answer BPB, but not in proportion. It discounts the math sources' gain on MMLU-Pro (FineMath 20% at d1024:
answer BPB −4.9%, choice loss −2.6%) and amplifies educational text's gain on TruthfulQA (FineWeb-Edu 20% at d1024:
−2.4% and −3.7%). The weighting alone does not explain the change: averaging answer BPB relative to the control, as
knowledge now does, calls FineWeb-Edu 20% `better` only at d1024, and Edu against FineWeb only at d768.

## Limits

- No contamination scan yet: a gain on one eval set alone needs a check for leakage.
- A verdict says whether a candidate helps at one share, not how much compute it saves. For a source under serious
  consideration, run a second share (as FineMath's dose response did) and read the compute saved at equal loss off the
  control's loss curve, by hand for now.
- Commonsense keeps answer BPB, which misses a gain in telling the gold answer from the distractors. `score` also
  writes accuracy on seven more multiple-choice tasks as a check; at d768 the control is above chance on SciQ,
  ARC-Easy and PIQA.
- The sample is whole files, so a dataset with few large files is sampled coarsely, and one file tokenizes on one
  worker.
- A seed sets both the model init and the data order. On the vanilla Track, FineWeb-Edu runs varied 1.0% between seeds
  on Paloma, almost three times as much as the control (0.38%).
- Runs train and score on `cw-us-east-02a` only, the region of the base cache.
- Text datasets only: the text field must be a string.
