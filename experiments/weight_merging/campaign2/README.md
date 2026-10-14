# Reproduce the Base-preserving merging campaign

Campaign 2 evaluated 48 merges, four variants of each of 12 methods, on NUPA200 and BFCL-Parity. Every export changes at most 25% of the original Base scalar weights and preserves Base's non-weight files, tensor names/shapes/dtypes, safetensors headers and shard layout. [TRACKER.md](TRACKER.md) contains all results and construction-time estimates; [results-evidence.json](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/blob/1ef199578aeec9f08e142a8dd760d44658264be4/campaign2-recipes/results-evidence.json) contains unrounded scores, coverage, evaluation pins and timing sources. The completed study is [issue #9975](https://github.com/marin-community/marin/issues/9975).

The best observed sum was DARE-TIES-02: NUPA200 0.00% + BFCL-Parity 45.53% = 45.53, with 6.04% of Base weights changed. Its [public checkpoint](https://huggingface.co/open-athena/Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10/tree/153f5369f69a0c0a7203779099ec94e98c66748f) is pinned to an immutable revision. These are single-run optimization scores. Non-target capabilities were not evaluated for these candidates. Bit-identical activation/serial control pairs produced different NUPA scores, so small differences are not established effects.

## Inputs and environment

Run commands from this PR's repository root, using Python 3.12 or newer. CPU exports used 16 CPUs and 128 GB RAM. ISO requires one CUDA H100; calibration requires eight H100s and 768 GB host RAM. Provide enough storage for three approximately 134 GB parents and each output; serial merges write two full outputs. Local and configured fsspec storage paths are supported. Keep object-storage compute in the source region.

Download these public parents at the exact revisions:

| Role | Hugging Face repository | Revision |
| --- | --- | --- |
| Base | [Grug Antidoom Step12](https://huggingface.co/open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02) | `50fe2f488aee7730bbccc721024b35229075602e` |
| NUPA donor | [Snowball Mixed-RLVR Step92](https://huggingface.co/open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92) | `fdc3c3c5fff489d3fb04dcbc2387c179b1bf3fab` |
| BFCL donor | [Grug Reference-KL Step20](https://huggingface.co/open-athena/Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04) | `fd07ef0f2b78a83099a0042e59c00de986e7b647` |

`revision` in a recipe is provenance, not a download instruction or remote revision check. Verify the downloaded revisions yourself. The original Base file names, sizes and SHA256 hashes are in [base-files.json](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/blob/1ef199578aeec9f08e142a8dd760d44658264be4/campaign2-recipes/base-files.json). For the same file-set audit, retain exactly those model files, excluding HF download-cache files and repository documentation from the input model directory. If a parent download lacks the two serving-cache files `.cache_metadata.json` and `.marinskyrl-model-manifest.json`, the public DARE-TIES-02 repository contains the exact Base copies; verify their hashes against `base-files.json` before copying those two files into Base. Do not copy the merged weight shards into Base. The published output adds a model card, license and provenance outside that original construction comparison.

The BFCL donor is documented as a continuation of Base. NUPA has aligned tensor coordinates, but its exact common ancestor with Base is unestablished. No expert-permutation matching is performed.

## Prepare fresh recipes

The public [JSON recipes](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/tree/1ef199578aeec9f08e142a8dd760d44658264be4/campaign2-recipes) preserve historical source paths and exact support selections. They must be copied with fresh paths before execution; the historical paths are not public download locations. Replace the example paths below with your downloaded parents and a new output directory or regional storage prefix:

```sh
uv run --package marin-core --extra cpu python -m experiments.weight_merging.prepare_campaign2 \
  --base /data/parents/base --nupa /data/parents/nupa --bfcl /data/parents/bfcl \
  --output-prefix /data/campaign2-replay --recipes /data/campaign2-recipes
```

The command downloads the recipes from the public dataset at a pinned revision and copies all 54 executable recipes: 48 candidates, one Base-copy control and five calibration jobs. It remaps source paths, calibration outputs, serial intermediate paths and audit destinations. It also restores the exact 48-row synthetic calibration data to `experiments/weight_merging/campaign2/calibration.jsonl` when that file is absent. Run subsequent commands from the repository root so the synthetic calibration data and SkyRL patch resolve. Do not reuse nonempty output directories or an existing audit. A failed export must not be served or used as a serial input.

## Construct checkpoints

For DARE-TIES-02, which needs no calibration:

```sh
uv run --package marin-core --extra cpu python -m experiments.weight_merging.constrained_merge \
  --recipe /data/campaign2-recipes/dare-ties-02.json
```

The same entry point handles `avg-*`, `task-*`, `ties-*`, `dare-linear-*`, `dare-ties-*`, `ram-*`, `structured-*`, `activation-*` and `base-copy`. Activation recipes already contain the exact selected expert IDs; rerunning routing is optional when replaying those frozen selections. R12/R25 select whole routed experts up to 12.50%/25.00% of Base scalars. A12/A25 select experts by domain-normalized Base routing on the synthetic calibration examples, split equally between numeric and tool domains; the random control matches layer/domain counts. Structured variants select the component blocks specified in each recipe.

Run ISO on a machine with its CUDA device visible, using the GPU extra:

```sh
uv run --package marin-core --extra gpu python -m experiments.weight_merging.constrained_merge \
  --recipe /data/campaign2-recipes/iso-03.json
```

ISO adapts [ISO-Merger](https://iso-rlvr.github.io/) to selected complete MoE expert projection matrices. It preserves Base singular values before BF16 storage; it does not preserve the spectrum exactly after quantization. The audit includes sampled stored-spectrum diagnostics.

Serial recipes run both stages and audit each against original Base:

```sh
uv run --package marin-core --extra cpu python -m experiments.weight_merging.serial_merge \
  --recipe /data/campaign2-recipes/serial-03.json
```

Serial-01/02 are disjoint-support ordering controls and produce the same checkpoint bytes as activation-01/02. Serial-03/04 blend overlapping support in opposite donor orders. The saved final audit covers stage 2; stage 1 has a separate audit. The tracker sums both construction timers.

## Calibrated methods

The 48 synthetic examples contain 16 numeric, 16 tool-call and 16 retention examples. They are independent of the two evaluation sets. A byte-identical copy is mirrored in the [public recipe folder](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/blob/1ef199578aeec9f08e142a8dd760d44658264be4/campaign2-recipes/calibration.jsonl); `prepare_campaign2.py` restores it automatically. Recipes enforce data SHA256 `a01451c472fb23d5212e3ac3e408dbcb8b99129ae812e64070e756814066002e`. Calibration uses Base token IDs for all three source models.

Run calibration in a disposable GPU environment: the runtime installs Transformers 5.16.1 and Loguru 0.7.3, fetches MarinSkyRL `33b47f2975b304bfc3417c0eb62abcdd3ce4264b`, and applies the included hash-checked standalone import patch. All eight GPUs must be visible.

```sh
uv run --package marin-core --extra gpu python -m experiments.weight_merging.campaign_calibration_runtime \
  --skyrl-revision 33b47f2975b304bfc3417c0eb62abcdd3ce4264b \
  --recipe /data/campaign2-recipes/calibration-foundation.json \
  --code-revision "$(git rev-parse HEAD)"
```

This produces routing counts and the two donors' squared per-sequence response-NLL gradients. These are calibration second moments, not original RL optimizer states. Run this shared job once before exporting `ota-ffg-01` through `ota-ffg-04`:

```sh
uv run --package marin-core --extra cpu python -m experiments.weight_merging.constrained_calibrated \
  --recipe /data/campaign2-recipes/ota-ffg-02.json
```

For each Expert Merging++ variant, run its independent fit, then export:

```sh
uv run --package marin-core --extra gpu python -m experiments.weight_merging.campaign_calibration_runtime \
  --skyrl-revision 33b47f2975b304bfc3417c0eb62abcdd3ce4264b \
  --recipe /data/campaign2-recipes/fit-expertpp-02.json \
  --code-revision "$(git rev-parse HEAD)"
uv run --package marin-core --extra cpu python -m experiments.weight_merging.constrained_calibrated \
  --recipe /data/campaign2-recipes/expertpp-02.json
```

The fit optimizes convex Base/NUPA/BFCL coefficients on selected routed experts, with 24 optimizer steps with layer-level coefficients followed by 96 optimizer steps with 104 chunks (120 steps total per fit). Frozen components stay Base. Recipes specify the hidden-state/logit objectives and retention-example weighting. This is a native-MoE adaptation, not a claim to reproduce the paper's model-specialist setting. GPU reduction order can change refitted coefficients; historical coefficients and curvature tensors remain in private regional storage, so exact byte replay of the calibrated outputs is not available from this PR alone.

Repeat the corresponding command for suffixes `01` through `04` in each family. Construction jobs can run independently once their calibration inputs exist; cap concurrency according to available memory and storage throughput. Read the successful audit outside each model directory before evaluation. It verifies all stored objects, counts bitwise scalar changes, and rejects non-weight file or layout changes.

## Evaluate with the campaign pins

Evaluation uses separate pinned Marin checkouts, not the PR checkout. See the [public evaluation release and commands](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/blob/c4fb17eb44005e01cbb3d5b38c672a2ffa473880/artifacts/inputs/RELEASE.md) for authentication and backend setup. An Iris/CoreWeave account and BFCL's Harbor/Daytona environment are needed for these commands. The raw evaluation transcripts and private regional model paths are not public artifacts.

```sh
git worktree add --detach /data/marin-bfcl f9990d362cd9b4ff97ae2ef0194d4bc0cc283050
git worktree add --detach /data/marin-nupa d2369021770463d4282126ddb0b37eab25aa8429
```

Download [eval-model.yaml](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/blob/1ef199578aeec9f08e142a8dd760d44658264be4/campaign2-recipes/eval-model.yaml) into each checkout's `experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs/` as `wm2-replay.yaml`. Set `name` to `wm2-replay` and `location` to the new model's serving-accessible location. The historical identity was a SHA1 of the sorted JSON construction recipe. Generate the replacement fields from your remapped recipe:

```sh
uv run --package marin-core --extra cpu python - <<'PYTHON'
import hashlib, json
recipe = json.load(open('/data/campaign2-recipes/dare-ties-02.json'))
revision = hashlib.sha1(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
print('revision:', revision)
print('identity: wm2-replay-' + revision)
PYTHON
```

Use those two printed fields in the YAML; they are provenance identifiers, not HF commit IDs. Keep Base tokenizer revision and all serving settings unchanged. The provided file is the exact historical DARE-TIES-02 serving configuration. Other candidates used the same settings with their own name, path and recipe identity.

BFCL, from `/data/marin-bfcl`:

```sh
./experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/launch-campaign.sh \
  --model wm2-replay --suite agentic --harbor bfclparity-pi --submit
```

For NUPA, download [nupa200.yaml](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/blob/1ef199578aeec9f08e142a8dd760d44658264be4/campaign2-recipes/nupa200.yaml) to `/data/marin-nupa/experiments/weight_merging/nupa200.yaml`. From `/data/marin-nupa`, run in Bash:

```sh
source experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/common.sh
resolve_hf_token
resolve_coreweave_credentials
run_campaign_launch 1 \
  --model-config "$MODEL_CONFIG_DIR/wm2-replay.yaml" \
  --evalchemy-config "$MARIN_DIR/experiments/weight_merging/nupa200.yaml" \
  --federated_cluster "$FEDERATED_CLUSTER" --priority "$PRIORITY"
```

NUPA200 tests numerical understanding and arithmetic across number representations on a fixed 200-item subset of the policy’s 5,000-item panel. BFCL-Parity is Harbor’s `bfcl_parity@1.0` adaptation with the Pi agent, not the full BFCL leaderboard protocol.

NUPA uses Evalchemy `2c102ddc321a1f9280c094e5996792eae8b8e413`, its `nupa200-loose` extra, seed 20222943, max length 73664 and max output 72945. [The manifest](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/blob/1ef199578aeec9f08e142a8dd760d44658264be4/campaign2-recipes/nupa200_manifest.jsonl) contains only task/digit/hash identities, and [selection metadata](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/blob/1ef199578aeec9f08e142a8dd760d44658264be4/campaign2-recipes/nupa200-selection.json) records the subset construction. BFCL uses Harbor `2203bacd03d82e9d4f2a9422f0b3f7e9bb7952cc`; each candidate attempts 123 trials. Check the saved record's model identity against your new recipe/YAML; check its tokenizer, runtime and evaluation source digest against [results-evidence.json](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/blob/1ef199578aeec9f08e142a8dd760d44658264be4/campaign2-recipes/results-evidence.json). Require at least 90% scored coverage, apply the release's [coverage and zero-score trace audit rules](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/blob/c4fb17eb44005e01cbb3d5b38c672a2ffa473880/artifacts/inputs/RELEASE.md#zero-score-trace-audit), and review ten traces for a zero score. No OOD evaluation is required by this completed campaign.

## CPU checks

```sh
uv run --package marin-core --extra cpu --group test pytest \
  tests/test_weight_merging.py tests/test_constrained_merging.py \
  experiments/weight_merging/test_select_experts.py \
  experiments/weight_merging/test_constrained_recipes.py
```
