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

Caches and the Eval Policy manifest live under the `MARIN_PREFIX` that the training cluster (`cw-us-east-02a`) sets
in its Iris config, so they sit beside the Track's base cache. Commands that load the tokenizer on your machine
(`check`, `build-answer-bpb`, `prepare` without `--submit`) also need `MARIN_PREFIX` set to that prefix.

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
Eval Policy (marin#10197): small ladder models score near chance on the benchmarks themselves. `uv run data-belay build-answer-bpb`
writes these caches; GPQA is left out because its dataset is gated. The verdict averages each Eval Policy group's
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

`uv run data-belay noise --rung d512` prints the control's seed spread. Confirm a d512 result at d768: at 20%,
OpenCoder's synthetic code was a `trade-off` at d512 (code −9.1%) and `hurts` at d768 (Paloma +2.4%, Uncheatable
+2.7%), while FineMath's math gain grew from −12.2% at d512 to −16.7% at d1024. See [`judge.py`](judge.py).

## Validation runs (2026-11-06)

| Candidate, share | Rung, seeds | Paloma | Uncheatable | knowledge | commonsense | math | code | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FineMath-4plus, 20% | d512, 6 | +0.08% | +0.72% | +0.02% | +0.53% | −12.17% | −0.44% | trade-off |
| FineMath-4plus, 20% | d768, 3 | +0.13% | +0.46% | −0.70% | +0.58% | −14.79% | −1.86% | trade-off |
| FineMath-4plus, 20% | d1024, 3 | +0.33% | +0.73% | −1.15% | +0.29% | −16.70% | −4.01% | trade-off |
| OpenCoder synthetic code, 20% | d512, 3 | +1.49% | +1.33% | +0.78% | +1.85% | +1.72% | −9.05% | trade-off |
| OpenCoder synthetic code, 20% | d768, 3 | +2.36% | +2.68% | +2.12% | +2.57% | +3.41% | −6.52% | hurts |
| FineWeb-2 Chinese, 20% | d512, 3 | +0.79% | +1.07% | +0.84% | +1.34% | +1.69% | +1.96% | no gain |
| FineWeb, 50% | d512, 6 | −2.43% | −0.01% | −2.68% | −5.13% | +3.69% | +7.78% | hurts |
| FineWeb-Edu against FineWeb, 50% | d512, 6 | +2.34% | +0.92% | +0.00% | +3.11% | −0.67% | +0.94% | hurts |

Each `Δ` is BPB against the control at the same rung (6 control seeds at d512, 3 at d768 and d1024), negative is better. The
runs are `dbelay-<rung>-<candidate>-s<seed>` in W&B `marin-community/marin_moe`.

## Limits

- No contamination scan yet: a gain on one eval set alone needs a check for leakage.
- The sample is whole files, so a dataset with few large files is sampled coarsely, and one file tokenizes on one
  worker.
- A seed sets both the model init and the data order. FineWeb-Edu runs varied 1.0% between seeds on Paloma, almost
  three times as much as the control (0.38%).
- Text datasets only: the text field must be a string.
