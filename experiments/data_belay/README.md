# Data Belay

The Data Belay tests whether a Hugging Face dataset improves Marin's held-out evals. It gives the dataset a share
`p` of the training tokens of the Belay `vanilla` Track and compares the result with a control that trains on the
same number of base tokens.

| Arm | Training tokens at rung budget `B` |
|---|---|
| control | `B` tokens of the base cache `hero_tok/v16384_shuf/train` |
| treatment | `(1 - p)·B` base tokens and `p·B` tokens of the candidate |

The control depends only on the rung and the seed, so every candidate reuses the same control runs.

## Commands

`B` is `uv run python -m experiments.data_belay`. Point `MARIN_PREFIX` at `s3://marin-us-east-02a/marin` for local
commands that read the tokenizer or write caches.

```bash
# 1. Intake: pin the revision, check the license, estimate the size and epochs at each rung.
B check --dataset HuggingFaceTB/finemath --config finemath-4plus --share 0.2

# 2. Tokenize a seeded sample of the dataset's files into a cache (an Iris CPU job).
B prepare --dataset HuggingFaceTB/finemath --config finemath-4plus --share 0.2 --submit

# 3. Train each seed of the treatment (the control runs once per rung and seed).
B train --rung d512 --seed 0 --run-id dbelay-d512-finemath-p20-s0 \
    --mix-cache s3://marin-us-east-02a/marin/data_belay/caches/<...>/train --mix-share 0.2 --submit

# 4. Verdict against the shared control.
B report --rung d512 --candidate finemath-p20 --seeds 0,1,2
```

`check` refuses a license that is not on the allowlist (NC and ND licenses, `other`, a missing license, or a second
license that is not on the list) and a share that reads the dataset more than 4 times at the largest rung.

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
Eval Policy (marin#10197): small ladder models score near chance on the benchmarks themselves. `B build-answer-bpb`
writes these caches; GPQA is left out because its dataset is gated.

## Verdict

For each group, `Δ` is the treatment's mean BPB over the control's, minus 1, with a Welch t interval over seeds.
`helps` needs one group wholly below 0 and no group that can reach its harm limit; `hurts` is a group wholly above
its harm limit. See [`judge.py`](judge.py).

## Limits

- No contamination scan yet: a gain on one eval set alone needs a check for leakage.
- The sample is whole files, so a dataset with few large files is sampled coarsely.
- Text datasets only: the text field must be a string.
