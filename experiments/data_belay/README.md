# Data Belay

The Data Belay tests whether a Hugging Face dataset improves Marin's held-out evals. It gives the dataset a share
`p` of the training tokens of the Belay `vanilla` Track and compares the result with a control that trains on the
same number of tokens, all from the base cache.

| Arm | Training tokens at rung budget `B` |
|---|---|
| control | `B` tokens of the base cache `hero_tok/v16384_shuf/train` |
| treatment | `(1 - p)·B` base tokens and `p·B` tokens of the candidate |

The control depends only on the rung and the seed, so every candidate reuses the same control runs.

## Commands

Candidate caches live under the `MARIN_PREFIX` that the training cluster (`cw-us-east-02a`) sets in its Iris config,
so they sit beside the Track's base cache. Commands that load the tokenizer on your machine (`check`, and `prepare`
without `--submit`) also need `MARIN_PREFIX` set to that prefix.

```bash
# 1. Intake: pin the head commit, check the license, estimate the size and epochs at each rung.
uv run data-belay check --dataset HuggingFaceTB/finemath --config finemath-4plus --share 0.2

# 2. Tokenize a seeded sample of the dataset's files into a cache (an Iris CPU job). The job prints the cache path
#    and the candidate's label, for example finemath-finemath-4plus-p20-<hash>.
uv run data-belay prepare --dataset HuggingFaceTB/finemath --config finemath-4plus --share 0.2 --submit

# 3. Train each seed of the treatment. --mix-cache is the cache path that prepare printed, not the Hugging Face
#    dataset. The run id is dbelay-<rung>-<label>-s<seed>, and the share is the one the cache was prepared for.
#    The control (no --mix-cache) runs once per rung and seed, for the seeds in results.CONTROL_SEEDS.
uv run data-belay train --rung d512 --seed 0 \
    --mix-cache s3://marin-us-east-02a/marin/data_belay/caches/<...> --submit

# 4. Verdict against the shared control, or against another candidate at the same seeds.
uv run data-belay report --rung d512 --candidate <label> --seeds 0,1,2
uv run data-belay report --rung d512 --candidate <filtered label> --against <unfiltered label> --seeds 0,1,2
```

`train` launches with the Belay Track's own command (`experiments.belay.submit.track_command`), so runs write under
the Belay's 30-day TTL root, and it refuses a run id that W&B already has.

`check` refuses a license that is not on the allowlist (NC and ND licenses, `other`, a missing license, or a second
license that is not on the list) and a share that reads the dataset more than 4 times at the largest rung. It sizes
the head commit only, because the row count comes from the datasets-server, which sizes the head.
`prepare` checks the epochs again on the cache it built, and writes the manifest that `train` reads only when they
pass.

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

## Verdict

For each group, `Δ` is the treatment's mean BPB over the control's, minus 1, with a Welch t interval over seeds. A
group is `better` (interval below 0), `worse` (interval above its harm limit), `neutral` (interval below the limit) or
`unresolved`. The harm limit is 1%, or 2.5 seed SDs of the d512 control where the seeds vary more.

| Verdict | Rule |
|---|---|
| `helps` | a group is better, and every other group is better or neutral |
| `trade-off` | a group is better, and another group may be past its harm limit |
| `hurts` | a group is worse |
| `no gain` | no group is better and none is worse |

`uv run data-belay noise --rung d512` prints the control's seed spread. See [`judge.py`](judge.py).

## Ladder

A candidate climbs d512, d768, d1024 until two consecutive rungs say no:

| Verdicts so far | Next step |
|---|---|
| any one rung | run the next rung |
| no at two consecutive rungs | stop: the answer is no |
| a yes at either of the last two rungs, below the top | run the next rung |
| yes at d768 and d1024 | the answer is yes |
| d768 and d1024 disagree | mixed: report both |

A yes is `helps` or `trade-off`, a gain on one of the evals; a no is `hurts` or `no gain`. `report` prints the step
when the candidate's runs at the rungs below share its label. One rung is not enough either way: at 20%, OpenCoder's
synthetic code was a `trade-off` at d512 (code −9.1%) and `hurts` at d768 (Paloma +2.4%, Uncheatable +2.7%), while
FineMath's math gain grew from −12.2% at d512 to −16.7% at d1024.

## Validation runs (2026-11-06 to 2026-11-07)

| Candidate | Rungs | Answer |
|---|---|---|
| FineMath-4plus, 5% | helps, trade-off, trade-off | yes |
| FineMath-4plus, 20% | trade-off, trade-off, trade-off | yes |
| FineMath-4plus, 50% | trade-off, hurts, hurts | no |
| OpenCoder synthetic code, 20% | trade-off, hurts, hurts | no |
| OpenCoder web code, 20% | trade-off, no gain, no gain | no |
| FineWeb-2 Chinese, 20% | no gain, hurts | no |
| FineWeb, 50% | hurts, hurts | no |
| FineWeb-Edu, 50% | hurts, hurts | no |
| FineWeb-Edu against FineWeb, 50% | hurts, hurts | no |
| Cosmopedia web samples, 20% | no gain, no gain | no |
| Null: control seeds 6–8 | no gain, no gain | no |

| Candidate, share | Rung, seeds | Paloma | Uncheatable | knowledge | commonsense | math | code | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FineMath-4plus, 5% | d512, 3 | −0.43% | −0.41% | −0.60% | −0.38% | −6.15% | −1.04% | helps |
| FineMath-4plus, 5% | d768, 3 | +0.14% | +0.13% | −1.04% | +0.51% | −7.89% | −0.27% | trade-off |
| FineMath-4plus, 5% | d1024, 3 | −0.08% | +0.01% | +0.12% | −0.16% | −8.30% | −1.61% | trade-off |
| FineMath-4plus, 20% | d512, 6 | +0.08% | +0.72% | +0.02% | +0.53% | −12.17% | −0.44% | trade-off |
| FineMath-4plus, 20% | d768, 3 | +0.13% | +0.46% | −0.70% | +0.58% | −14.79% | −1.86% | trade-off |
| FineMath-4plus, 20% | d1024, 3 | +0.33% | +0.73% | −1.15% | +0.29% | −16.70% | −4.01% | trade-off |
| FineMath-4plus, 50% | d512, 3 | +1.94% | +3.57% | +2.06% | +3.03% | −15.87% | +1.11% | trade-off |
| FineMath-4plus, 50% | d768, 3 | +1.43% | +2.71% | +0.32% | +1.87% | −21.18% | −1.44% | hurts |
| FineMath-4plus, 50% | d1024, 3 | +2.18% | +3.40% | +1.46% | +1.71% | −23.48% | −2.60% | hurts |
| OpenCoder synthetic code, 20% | d512, 3 | +1.49% | +1.33% | +0.78% | +1.85% | +1.72% | −9.05% | trade-off |
| OpenCoder synthetic code, 20% | d768, 3 | +2.36% | +2.68% | +2.12% | +2.57% | +3.41% | −6.52% | hurts |
| OpenCoder synthetic code, 20% | d1024, 3 | +2.15% | +2.44% | +2.99% | +1.72% | +2.03% | −5.31% | hurts |
| OpenCoder web code, 20% | d512, 3 | −1.12% | −0.58% | −0.42% | −0.04% | +0.40% | +1.62% | trade-off |
| OpenCoder web code, 20% | d768, 3 | −0.82% | −0.66% | −0.12% | −0.65% | −1.01% | +1.74% | no gain |
| OpenCoder web code, 20% | d1024, 3 | −0.63% | −0.61% | +0.25% | −0.32% | −0.80% | −0.49% | no gain |
| FineWeb-2 Chinese, 20% | d512, 3 | +0.79% | +1.07% | +0.84% | +1.34% | +1.69% | +1.96% | no gain |
| FineWeb-2 Chinese, 20% | d768, 3 | +2.19% | +2.83% | +2.82% | +2.49% | +3.78% | +4.11% | hurts |
| FineWeb, 50% | d512, 6 | −2.43% | −0.01% | −2.68% | −5.13% | +3.69% | +7.78% | hurts |
| FineWeb, 50% | d768, 3 | −2.99% | −0.65% | −3.76% | −5.51% | +2.90% | +8.29% | hurts |
| FineWeb-Edu, 50% | d512, 6 | −0.14% | +0.91% | −2.68% | −2.18% | +2.99% | +8.79% | hurts |
| FineWeb-Edu, 50% | d768, 3 | −1.06% | −0.46% | −4.72% | −3.66% | +0.77% | +7.98% | hurts |
| FineWeb-Edu against FineWeb, 50% | d512, 6 | +2.34% | +0.92% | +0.00% | +3.11% | −0.67% | +0.94% | hurts |
| FineWeb-Edu against FineWeb, 50% | d768, 3 | +2.00% | +0.19% | −1.00% | +1.96% | −2.06% | −0.29% | hurts |
| Cosmopedia web samples, 20% | d512, 3 | +1.20% | +1.21% | +0.11% | +0.27% | +3.61% | +4.23% | no gain |
| Cosmopedia web samples, 20% | d768, 3 | +0.71% | +0.84% | +1.04% | −0.09% | +2.34% | +4.00% | no gain |
| Null: control seeds 6–8 | d512, 3 | +0.17% | +0.19% | +0.44% | +0.61% | +0.38% | −0.15% | no gain |
| Null: control seeds 6–8 | d768, 3 | −0.13% | −0.06% | −0.08% | +0.43% | −0.88% | −0.08% | no gain |

Each `Δ` is BPB against the control at the same rung (6 control seeds at d512, 3 at d768 and d1024), negative is better. The
runs are `dbelay-<rung>-<candidate>-s<seed>` in W&B `marin-community/marin_moe`; the null arm is control seeds 6–8,
judged against the control seeds above.

Four arms tested the method, with expectations written down before their runs: the null arm, FineMath at 5% and 50%,
and Cosmopedia.

- **Null** (met): no group is `better` or `worse` at either rung. At d512 one of six group intervals (commonsense,
  +0.6%) lies above 0, about the rate a 95% interval allows, so one group's gain of about 0.5% at a single rung is noise.
- **FineMath dose response** (met): the math gain grows with the share at every rung (d1024: −8.3%, −16.7%, −23.5%),
  and so does the cost on Paloma and Uncheatable. 5% gets about half of 20%'s math gain at no cost elsewhere; 50% is
  `hurts`.
- **Cosmopedia** (missed): synthetic textbooks were expected to improve knowledge or commonsense, and no task in either
  group moved. Accuracy agrees (below), so this is a real no for this mix at d768.
- **OpenWebMath, 20%** (met): a second math source, expected to be a yes with a smaller math gain than FineMath's. It
  is a `trade-off` at d512 and d768 (math −6.5% and −7.7%, about half of FineMath 20%'s).

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

## Limits

- No contamination scan yet: a gain on one eval set alone needs a check for leakage.
- Answer BPB misses a gain in telling the gold answer from the distractors (see the accuracy check), so a quality
  filter compared against its unfiltered source needs `experiments.data_belay.accuracy` too.
- The sample is whole files, so a dataset with few large files is sampled coarsely, and one file tokenizes on one
  worker.
- A seed sets both the model init and the data order. FineWeb-Edu runs varied 1.0% between seeds on Paloma, almost
  three times as much as the control (0.38%).
- Text datasets only: the text field must be a string.
