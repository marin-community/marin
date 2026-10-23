# Campaign 4: constrained routed-expert transfer

Campaign 4 transfers capabilities from `open-athena/Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04` into `open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02`. A candidate may change at most 30% of Base's indexed scalar weights, may edit only rows of routed-expert projection tensors, and must preserve every other weight and non-weight checkpoint file byte-for-byte.

The campaign evaluates HumanEval+, MBPP+, OlympiadBench, and BFCL-Parity/Pi as transfer targets and MATH500 as a regression canary. It includes bounded DARE-linear and serial DARE-TIES/isospectral recipes, exact and fractional serial expert transfer, task-paired expert selection, task-union transfer, layer-adaptive donor deltas, behavioral-residual grafting, and wide-support serial transfer.

## Result

All 71 candidates completed the five fixed-policy evaluations. The three highest target sums came from layer-adaptive donor deltas:

| Candidate | Changed weights (%) | HumanEval+ | MBPP+ | OlympiadBench | BFCL-Parity | Target Sum | MATH500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [`adaptive-delta-08`](https://huggingface.co/open-athena/Grug-67B-A2B-Merge-Campaign4-Adaptive-Delta-08-2026.10.14/tree/2fb31748eed37435c7472e80f81f34e306b2fa7a) | 9.47 | 83.54 | 74.07 | 34.33 | 28.46 | 220.40 | 93.60 |
| [`adaptive-delta-05`](https://huggingface.co/open-athena/Grug-67B-A2B-Merge-Campaign4-Adaptive-Delta-05-2026.10.14/tree/4a19865c0ed69efab0bf457804d2d698fb3f72f5) | 8.18 | 76.83 | 73.02 | 36.00 | 27.64 | 213.49 | 94.00 |
| [`adaptive-delta-06`](https://huggingface.co/open-athena/Grug-67B-A2B-Merge-Campaign4-Adaptive-Delta-06-2026.10.14/tree/337eb566795323f93ee16bbf294da6efd5e48539) | 9.47 | 76.22 | 73.81 | 35.00 | 28.46 | 213.48 | 93.00 |

Base scored 174.90 on the four-target sum and 95.00 on MATH500. These are single-run development results from a search over the four reported targets; MATH500 was the only regression canary.

## Pinned inputs

| Input | Revision |
| --- | --- |
| Base | `open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02@50fe2f488aee7730bbccc721024b35229075602e` |
| Extraction target | `open-athena/Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04@fd07ef0f2b78a83099a0042e59c00de986e7b647` |
| Marin evaluation release | `marin-community/marin@aaf1878ce65a1a05b386aef257b952fa7205283a` |
| Evalchemy | `marin-community/evalchemy@958cdb8019b7a4c8432fe85eb9572538ceb75950` |
| Harbor | `marin-community/harbor@2203bacd03d82e9d4f2a9422f0b3f7e9bb7952cc` |

The published winning model repositories contain the exact merge recipe, selected tensor rows, row coefficients, readback audit, final tracker snapshot, and sanitized sealed evaluation records under `provenance/campaign4/`. Use Marin [revision `14a407e045`](https://github.com/marin-community/marin/commit/14a407e045f83f38e4e94a2fb21869707a256997) for the merge commands below and the immutable model revision recorded in the campaign issue; the recipe's source revisions are provenance fields and do not download a parent automatically.

## Reproduce a merge

Use Python 3.12 and run entry points through `uv`. Acquire the two parent checkpoints at the revisions above, choosing paths accessible to the merge worker:

```bash
uv run --package marin-core --extra cpu python -m experiments.weight_merging.prepare_parent \
  --repo open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02 \
  --revision 50fe2f488aee7730bbccc721024b35229075602e \
  --output /path/to/base

uv run --package marin-core --extra cpu python -m experiments.weight_merging.prepare_parent \
  --repo open-athena/Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04 \
  --revision fd07ef0f2b78a83099a0042e59c00de986e7b647 \
  --output /path/to/extraction-target
```

Download a published `provenance/campaign4/<experiment>.json` recipe. In a copy of that recipe, replace `base.path`, each `donors[].path`, `output`, and `audit` with fresh paths under your control. For a serial recipe, also replace every `stages[].output` and `stages[].audit`; keep the selected support, coefficients, source revisions, support cap, and change cap unchanged.

Run the entry point selected by the recipe's method:

```bash
# DARE, exact transfer, task union, or row-scaled adaptive delta
uv run --package marin-core --extra cpu python -m experiments.weight_merging.constrained_merge \
  --recipe /path/to/recipe.json

# Two or more CPU materialization stages
uv run --package marin-core --extra cpu python -m experiments.weight_merging.serial_merge \
  --recipe /path/to/recipe.json

# Serial DARE-TIES followed by isospectral transfer; requires one CUDA GPU
uv run --package marin-core --extra gpu python -m experiments.weight_merging.serial_iso \
  --recipe /path/to/recipe.json
```

Do not expose the output until the command exits successfully and writes the audit outside the model directory. `export_constrained` reads every stored object back, verifies tensor names, shapes, dtypes, shard layout, unchanged bytes outside the declared support, unchanged non-weight files, the support cap, and the independently measured changed-weight cap. Every serial stage is compared with the original Base checkpoint, so an intermediate model cannot reset the budget.

Behavioral-residual experiments have one calibration step before materialization. Run `experiments.weight_merging.calibration_runtime` with `--module experiments.weight_merging.campaign_calibrate`, a pinned MarinSkyRL revision, a calibration recipe, and a source revision. That recipe must pin the examples, source checkpoints, task rankings, round count, rows per round, residual weights, and step size; calibration examples must remain disjoint from scored evaluation prompts. The calibration recipe and selected-row artifact for this family are not public, so its eight Campaign 4 candidates cannot be replayed exactly from this repository. The published winning adaptive-delta checkpoints do not use this calibration path.

## Reproduce the evaluations

Serve the completed checkpoint with the model YAML and vLLM settings from the pinned Marin evaluation release. From a checkout at `aaf1878ce65a1a05b386aef257b952fa7205283a`, copy `model-configs/open-athena-Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02.yaml` to `model-configs/campaign4-candidate.yaml`. Change only `name`, `location`, and `revision` to the published candidate and its immutable Hugging Face revision. From the repository root, launch all five evaluations on one shared serving group:

```bash
export MARIN_DIR="$PWD"
export RECORDS_PREFIX="s3://your-bucket/campaign4-replay/evals"
source experiments/evaluation/campaigns/eval-policy/common.sh
prepare_judge_environment
resolve_coreweave_credentials
run_campaign_launch 1 \
  --model-config experiments/evaluation/campaigns/eval-policy/model-configs/campaign4-candidate.yaml \
  --records-prefix "$RECORDS_PREFIX" \
  --federated_cluster cw-rno2a \
  --priority interactive \
  --evalchemy-config experiments/evaluation/campaigns/eval-policy/evalchemy-configs/humanevalplus.yaml \
  --evalchemy-config experiments/evaluation/campaigns/eval-policy/evalchemy-configs/mbppplus.yaml \
  --evalchemy-config experiments/evaluation/campaigns/eval-policy/evalchemy-configs/olympiadbench.yaml \
  --evalchemy-config experiments/evaluation/campaigns/eval-policy/evalchemy-configs/math500.yaml \
  --harbor-config experiments/evaluation/campaigns/eval-policy/harbor-configs/bfclparity-pi.yaml
```

The pinned campaign [README](https://github.com/marin-community/marin/blob/aaf1878ce65a1a05b386aef257b952fa7205283a/experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/README.md) documents credentials, durable records, and policy-conformance limits. The command above matches the original one-server topology and selects these exact configurations:

- `evalchemy-configs/math500.yaml`
- `evalchemy-configs/humanevalplus.yaml`
- `evalchemy-configs/mbppplus.yaml`
- `evalchemy-configs/olympiadbench.yaml`
- `harbor-configs/bfclparity-pi.yaml`

Seal a score only when at least 90% of declared trials are scoreable, as required by the release policy. Compute Target Sum from the unrounded HumanEval+, MBPP+, OlympiadBench, and BFCL-Parity/Pi scores; MATH500 remains separate.

## Validation

Run the behavior tests that cover constrained exports and serial composition:

```bash
uv run --package marin-core --extra cpu --group test pytest \
  tests/test_constrained_merging.py \
  experiments/weight_merging/test_constrained_recipes.py
```
