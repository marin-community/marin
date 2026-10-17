# Two-expert Snowball transfer (#9984)

The [final report](https://github.com/marin-community/marin/issues/9984#issuecomment-6144923850)
records the experiment, comparisons, costs and every evaluation ID. These recipes
preserve the E₂ MMLU-Pro expert and mixed plain/RS-SFT arms from tasks
157/162/163/165/166. They depend on [#9920's shared recipes](cruxeval-transfer.md).
No traces or weights are checked in. The consolidation was validated offline;
training, collection and cloud audits were not rerun.

B is the frozen `open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92`
baseline. E is #9920's CruxEval expert; E₂ is the MMLU-Pro expert below.
MOPD denotes the two-expert on-policy distillation arm; MOPD-100 is its
step-100 export. E₂ step254 passed the subset-improvement gate and was selected
by the checkpoint rule in the final report, not by held-out accuracy.

## Entry points

Run modules from the repository root with `uv run --no-sync python -m`.
Outputs of preparation and audits require fresh prefixes.

| Module under `experiments.post_training` | Inputs and result |
| --- | --- |
| `second_expert.verify_candidate` | Pinned canonical labels, captured GLM requests/responses and TIGER-Lab/MMLU-Pro test revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`; `--output` writes independent identity, grouped-selection and native-mask verification. |
| `second_expert.prepare` | Verified GLM traces; `--output` writes 1,359 real rows plus one zero-loss row, prebuilt token IDs and shifted loss masks. Uses the shared `cruxeval_transfer.rendering`. |
| `second_expert.train` | `--report`, `--learning-rate 1e-5`, `--wandb-entity`, `--wandb-mode`; three epochs, AdamH, 32 H100. |
| `second_expert.collect_teacher` | `--model-config`, `--prompts`, `--output`, `--samples 8`, `--concurrency`; `--run` serves the gated E₂ step254 and captures 12,848 native completions. |
| `second_expert.audit_collection` | `--source` collection and `--output`; writes immutable first-correct selection sidecars and response hashes. |
| `mixed_transfer.prepare_domain` | `--source` E₂ collection and `--output`; sample zero, or `--selection <audit>/first-correct.jsonl` for historical RS. |
| `mixed_transfer.combine` | Two repeated `--source` prepared prefixes (CruxEval first, MMLU-Pro second), `--output`; removes old padding, preserves real-row order and adds new batch padding. |
| `mixed_transfer.train` | `--report`, its exact `--report-sha256`, `--arm plain\|rs`, `--learning-rate 1e-6`, `--wandb-entity`, `--wandb-mode`; four epochs, AdamW, 32 H100. |
| `mixed_transfer.audit_records` | `--output <fresh JSON path>`; audits selected records, requires the final-report HEAD commits, reconciles native writers from active source blobs, and reports invalid merge MATH500 rows. |

Training entry points default to graph inspection, with `--only cache|train`.
`--run` executes the selected graph and can request GPUs. A local report permits
offline inspection; passing a remote report reads cloud storage even without
`--run`. Verification, preparation and record audits read pinned cloud inputs.
Live runs require regional CoreWeave credentials (`CW_KEY_ID`, `CW_KEY_SECRET`),
an Iris coordinator, a regional `MARIN_PREFIX`, and `WANDB_API_KEY` for online
logging. Use the same storage region and obtain approval for the experiment's
compute budget; see the [Iris operations manual](https://github.com/marin-community/marin/blob/6b135b3e9db669f8d03df4bcf9b089f43f1fe78c/lib/iris/OPS.md).

Offline E₂ graph inspection (no tokenizer download or artifact read):

```bash
uv run --no-sync python -m experiments.post_training.second_expert.train \
  --report experiments/post_training/second_expert/training_manifest.json \
  --learning-rate 1e-5 --wandb-entity marin-community --wandb-mode offline
```

The checked-in training manifest is the original hashed artifact. Its
`selection_status` is historical preparation metadata, not a current launch
permission. Collection manifest/grade strings are also retained byte-for-byte
in meaning to reconcile existing sidecars.

For E₂ collection, `--model-config` is
`experiments/post_training/second_expert/teacher-step254.yaml`; `--prompts` is
`s3://marin-us-east-02a/marin/scratch/karan/glm-bench-traces/glm/MMLUPro/requests.jsonl`.
Each preparation module writes `manifest.json` beneath its `--output` prefix.
Pass the two domain-preparation output prefixes to `combine --source`, then its
output's `manifest.json` to mixed training. The CruxEval preparation inputs are
listed in the [single-expert recipe](cruxeval-transfer.md).

Successful candidate verification establishes that all 12,032 full rendered
prompts match the pinned source identities, raw GLM grading reproduces canonical
labels, the seeded whole-group draw reproduces 1,606 selected documents, and
usable examples fit the native template/mask and context constraints. It is not
a model-quality evaluation.

## Frozen artifacts

All S3 paths below share `s3://marin-us-east-02a/marin/`.
Student B and its tokenizer use `fdc3c3c5fff489d3fb04dcbc2387c179b1bf3fab`, shared
with #9920. Native completion token IDs are never decoded and retokenized or
truncated. MMLU-Pro domain padding uses ID 128489; mixed batch padding uses
128009. Both have zero loss.

| Input/output | Path or identity |
| --- | --- |
| Canonical labels and subset | `scratch/karan/mopd2/mmlupro-canonical-v2/`; canonical SHA-256 `ac5acdc984fc2698947e6c818c3595bc5d81d6eb494532a557d3734e787c424f`, subset SHA-256 `c12af58d8b7d75acdc421c281f9f0df7c438b9293415761fed07d8a63812f27b` |
| GLM captures | `scratch/karan/glm-bench-traces/glm/MMLUPro/` |
| Independent verification | `scratch/karan/mopd2/task162-candidate-verification-v4/` |
| E₂ prepared data | `teacher-traces/mopd2/task162-e2-prepared-20261009-v1/`; manifest SHA-256 `1dd3888bd655234c6f50837a8f503d4a2cfa79fdb2310320f6d8ae0f08badfdb` |
| Mixed plain prepared data | `teacher-traces/mopd2/task165-mixed-plain-prepared-20261010/` |
| Mixed RS prepared data | `teacher-traces/mopd2/task166-mixed-rs-prepared-20261010/` |
| E₂ k=8 collection / selection audit | `teacher-traces/mopd2/task163-e2-step254-k8-20261010/` / `teacher-traces/mopd2/task163-e2-step254-k8-audit-20261010/` |
| E₂ checkpoints | `experiments/task157-e2/checkpoints/karan/e2-mmlupro-step92-lr1e-05/2026.10.09.1/hf/step-{85,170,254}` |
| Mixed plain checkpoints | `experiments/task165-mixed/checkpoints/karan/mixed-plain-e-from-b-lr1e-06/2026.10.10.2/hf/step-{201,402,603,803}`; final relaunch trainer suffix `-20261010-r2` |
| Mixed RS checkpoints | `experiments/task166-mixed/checkpoints/karan/mixed-rs-e-from-b-lr1e-06/2026.10.10.1/hf/step-{182,364,546,727}`; trainer suffix `-20261010` |

The exact mixed `manifest.json` SHA-256 values for `--report-sha256` are:

- Plain: `aa6affd1568511cae23e5f84728b6b0932d1fbf122e8955e33f66601ce03c3e8`.
- RS: `3dc8737d11bfc653ecdb50e8e696fd9dba65f889f2334191dabff0f67d77a8e0`.

Dataset/cache versions remain `2026.10.10.1` for both mixed arms. Only plain's
training version changes to `.2`; it restarted from B after eviction.
Canonical grouping retains 1,606 subset docs (1,589 question-text groups) and
10,426 held-out docs. Plain trains 3,206 demos; RS trains 2,903 covered prompts
(1,503 CruxEval + 1,400 MMLU-Pro), without changing evaluation denominators.

## Selector and evaluation provenance

`second_expert.grading` deliberately preserves the historical first-match
extractor on the entire native completion, including reasoning. **It is not
the final evaluator.** The evaluator uses final content when available and
parser-aware reasoning fallback, then uppercases extracted answers. Fourteen
of the 1,400 selected MMLU-Pro RS demos are wrong under that evaluator;
regrading would gain 15 prompts, lose one and change 45 selected samples.
No corrected run was made. The grade metadata's old equivalence claim and
preparation's "canonical-answer-correct" label remain historical strings;
they must not be interpreted as verified evaluator agreement.

Frozen evaluation runtime: Evalchemy
`ae362881cf53f869a6ad52c8a59fa17343c0b5f8`. Collection dataset revision
`c131e5ab...` from #9920 is distinct. The full active-snapshot manifest is
`scratch/karan/mopd2/review/records-manifest-20261013.json`.
Here HEAD means the FineStore archive's selected commit, not a Git commit or
an Evalchemy version. Each archive has its own commit sequence.
B MMLU-Pro record `20261007-172014-...-220c` uses commit
`55b16b9998c3428babdc2d0031d0a387` (sequence 26), native writer
`evalchemy-034dd62a51134a1c87a9367f9cc2b71c`.
MOPD-100 MMLU-Pro record `20261011-075144-...-f728` uses commit
`946025cec9964374b833da37b12ba541` (sequence 24), writer
`evalchemy-f5b3cbfdd2c447af8e6dc1f34b6d0e33`.
Both had superseded full streams. Do not combine physical files across
attempts or treat those streams as independent repeats. `audit_records`
refuses a different HEAD; it does not update the published historical audit.
Its normalized-output re-extraction is diagnostic, not a replay of the native
Evalchemy scorer. Reading historical archives requires a compatible FineStore
format; no cloud or archive-format compatibility check was made here.

MOPD runs in [MarinSkyRL at `76fc4be9`](https://github.com/marin-community/MarinSkyRL/blob/76fc4be980aff1b14a42bfa549b5fe7479744701/cloud/iris/configs/snowball_mopd2_cruxeval_mmlupro.yaml).
Merge recipes remain in [#9974](https://github.com/marin-community/marin/pull/9974).

## Local checks

```bash
uv run --no-sync pytest experiments/post_training/second_expert \
  experiments/post_training/mixed_transfer \
  experiments/post_training/cruxeval_transfer/test_prepare.py -q
```
