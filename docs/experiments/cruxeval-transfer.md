# Single-expert CruxEval transfer (#9920)

The [final experiment record](https://github.com/marin-community/marin/issues/9920#issuecomment-6076348127)
compares plain supervised fine-tuning (SFT), rejection-sampled SFT (RS-SFT), and
on-policy distillation (OPD). All arms start from Snowball B and transfer the
CruxEval expert E's behavior on the same 1,600 exposed evaluation prompts.
Scores measure transfer on seen prompts. The record contains results, evaluator
record IDs, checkpoint selection, and compute accounting.

The implementation is in `experiments/post_training/cruxeval_transfer/`.
It preserves the final expert and SFT-arm recipes from tasks 133/144/145.
E+ was a separate reference, never the comparison teacher; its continuation
recipe and the abandoned MATH-500 pilot are excluded. OPD runs in MarinSkyRL;
its [configuration](https://github.com/marin-community/MarinSkyRL/blob/76fc4be980aff1b14a42bfa549b5fe7479744701/cloud/iris/configs/snowball_opd_cruxeval_e.yaml)
is maintained there.

## Frozen inputs and recipes

B is `open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92` at
`fdc3c3c5fff489d3fb04dcbc2387c179b1bf3fab`. E is the task144 export
`task133-cruxeval/checkpoints/karan/cruxeval-step92-lr1e-05/2026.10.07.5/hf/step-272`
under `s3://marin-us-east-02a/marin/experiments/`. `teacher.yaml` pins that
export and B's tokenizer revision.

| Run | Demonstrations | Optimizer | Updates | Selected export |
| --- | --- | --- | --- | --- |
| E | 1,454 correct, completed, reasoning-bearing GLM 5.3 traces | AdamH, LR 1e-5, betas .9/.95, 3% warmup and cosine decay | 273, three epochs | 272 |
| Plain SFT | E sample 0 for all 1,600 prompts, unfiltered | AdamW, constant LR 1e-6, betas .9/.999 | 400, four epochs | 399 |
| RS-SFT | First execution-correct E sample among eight; 1,503 prompts | Same AdamW recipe | 376, four epochs | 282 |

All use global batch 16, zero weight decay, gradient clip 1, frozen router biases,
unpacked 32,768-token rows, and 32 H100s: expert parallelism 8 within each node,
context parallelism 4 across nodes. The AdamH recipe imports the canonical
`experiments.grug.moe` implementation with the zero-norm guard used by the
historical recipe. Export indices are zero-based: `step-272` follows 273 updates.
Selected exports can precede the final update. The 97 RS-uncovered prompts remain
in evaluation.

Regional inputs are historical artifact identities. Their current availability
must be checked before a rerun; this consolidation does not recreate or move them.

| Input | Prefix under `s3://marin-us-east-02a/marin/teacher-traces/` |
| --- | --- |
| B regional copy | `cruxeval-test-exposed/step92-fdc3c3c5/` |
| GLM captured requests/responses | `cruxeval-test-exposed/task133-20261007-glm/` |
| Prepared E data | `cruxeval-test-exposed/task133-20261007-prepared-v2/` |
| Plain transfer data | `cruxeval-transfer/task145-plain-prepared-20261008/` |
| RS transfer data | `cruxeval-transfer/task145-rs-prepared-20261009/` |

`crux-report.json` is the E training manifest, including source hashes and token
counts. The GLM prefix contains `requests.jsonl` (including each exact captured
baseline request in `b_request`) and `raw-responses.jsonl`. Raw traces, token caches, model weights, grading bundles, and selection
outputs are not checked into Git.

## Data path

1. `collect_cruxeval` reconciles the exact captured baseline prompts with the
   checksum-pinned dataset and collects GLM responses without executing them.
2. `grade_cruxeval` uses the pinned Evalchemy execution module. It executes
   generated Python: run it only in a credential-free, network-disabled,
   resource-limited container, never in the collection or training process.
3. `prepare_cruxeval` selects the expert's GLM traces and uses `rendering` to
   combine reasoning, `<|end_think|>`, and the answer under B's native template.
   Prompt tokens receive no loss; overlong rows are rejected without truncation.
4. `collect_teacher` collects eight E completions per captured prompt with
   native token IDs, temperature/top-p 1, and a 30,720-token cap. Per-sample
   deterministic seeds belong to collection; evaluation was unseeded.
5. `export_grading` and `unpack_grading` transport the bounded text bundle to
   isolated execution grading. `select_rejection` chooses the first correct
   sample without a reasoning or finish-reason filter. `prepare_plain` uses
   sample 0, or `--selection` for RS. Both preserve completion token IDs and
   shift the loss mask for next-token prediction, including emitted EOS.

The collector fetches `eval/chat_benchmarks/CruxEval/data/cruxeval.jsonl` from
`marin-community/evalchemy` at dataset pin `c131e5ab`; the evaluator that actually ran was
`ae362881cf53f869a6ad52c8a59fa17343c0b5f8`. `constants.py` distinguishes these
pins. The [frozen evaluator configurations](https://gist.github.com/karansag/49de943771b98a7a23d9348b58933f71)
and final record describe vLLM H100x8, thinking enabled, the Grug reasoning
parser, and unseeded sampling at the server's temperature-1 default.

## Inspect or run the training graph

This command builds E's graph locally without requesting accelerators:

```bash
uv run python -m experiments.post_training.cruxeval_transfer.train_cruxeval \
  --report experiments/post_training/cruxeval_transfer/crux-report.json \
  --learning-rate 1e-5 --wandb-entity marin-community --wandb-mode offline --only train
```

For the plain SFT arm, this reads the historical S3 manifest and prints the graph:

```bash
uv run python -m experiments.post_training.cruxeval_transfer.train_plain \
  --arm plain \
  --report s3://marin-us-east-02a/marin/teacher-traces/cruxeval-transfer/task145-plain-prepared-20261008/manifest.json \
  --learning-rate 1e-6 --wandb-entity marin-community --wandb-mode offline --only train
```

For RS-SFT, use `--arm rs` and `task145-rs-prepared-20261009/manifest.json`.
Run storage-reading commands on a worker local to the CoreWeave bucket.
Historical output roots were `MARIN_PREFIX=s3://marin-us-east-02a/marin/experiments/task133-cruxeval`
for E and `.../task145-cruxeval` for both transfer arms.

Actual execution requires an Iris CPU coordinator with the regional storage
credentials and `WANDB_API_KEY` for online logging; see the [Iris operations
manual](https://github.com/marin-community/marin/blob/6b135b3e9db669f8d03df4bcf9b089f43f1fe78c/lib/iris/OPS.md). The parent copy must retain its historical
`task133-readiness.json` receipt (87 objects, 134,175,117,778 bytes). The recipes
consume that verified input; they do not recreate the old TTL staging source. Build `--only cache --run`
first, then `--only train --run`. The coordinator validates the cache and parent
receipt before requesting GPUs. Obtain resource approval before launching;
choose a new output root and change the `VERSION` constants and E's `SFTSpec.version`
for any changed experiment.

The HF export path sets `hf_save_reference_code=False` while retaining weights,
configuration, tokenizer, and generation metadata. E disables native checkpoint
saving; the SFT arms keep per-epoch native checkpoints. Before serving an export,
run the range-read audit from the same storage region:

```bash
uv run python -m experiments.post_training.cruxeval_transfer.verify_export \
  --export <export-prefix>
```

It checks index/header agreement, payload bounds and object sizes,
tokenizer/template equality with B, and exact router-bias equality after the
export dtype cast. The Snowball exports should report 26 router-bias tensors.
It does not checksum all weights or run model inference. This consolidates the
surviving external `task144_verify_export.py` and `task144_verify_router_bias.py`
audit scripts, with guarded storage access and no monitoring sleep. Their
identity against historical Iris workspace bundles was not independently
established. The final record reports the historical audit results; local tests
do not revalidate the stored 67B models.

Local checks:

```bash
uv run --package marin-core --extra cpu --extra dedup --group test pytest \
  experiments/post_training/cruxeval_transfer -q
```
