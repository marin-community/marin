# Getting started with TaskTrove

TaskTrove is the Marin program for reinforcement learning (RL) task data. It has three parts:

- The upstream input: [`open-thoughts/TaskTrove`](https://huggingface.co/datasets/open-thoughts/TaskTrove), a public Hugging Face dataset of archived [Harbor](https://github.com/marin-community/harbor) tasks.
- The [RL data curation pipeline](README.md), which converts these tasks and admits the tasks that pass its checks.
- The curated cohorts that Marin publishes back as releases in [`open-athena/task-trove`](https://huggingface.co/datasets/open-athena/task-trove).

This guide is for a first-time contributor. It tells you what each stage makes, where each stage writes its data, and how to add your first source. When you complete it, you have a declared source, a passing conversion test, a QUICK output and a SAMPLE report. Training use needs a separate change to a selection recipe.

For field-level detail, read the [curation reference](../../../docs/references/task-curation.md). For declaration mechanics, read the [datasets guide](datasets/README.md). The [Glossary](#glossary) at the end defines the terms that this guide uses.

## Before you start

Each stage needs more access than the stage before it. Get the access for the stages that you will run.

1. **For all stages:** install Python 3.12 or higher, `uv` and Git. Clone the repository and run `uv sync --all-packages`. Follow the [installation guide](../../../docs/tutorials/installation.md).
2. **For QUICK:** you need only a local computer. `open-thoughts/TaskTrove` is public, so QUICK needs no Hugging Face token.
3. **For SAMPLE with review:** you need a review model, an endpoint for it and an API key.
    - Marin's review model is [`glm-5.3`](../glm.py). Marin's `GLM_BULK_TOKEN` and its relay job [`/muchanem/glm53-relay`](../glm.py) are for Marin team members only.
    - External contributors must arrange their own access to a GLM-compatible review model on an OpenAI-compatible endpoint.
    - Give your endpoint with `--base-url URL`, for example `https://HOST/v1`. Without `--base-url`, the driver looks up the Marin relay job, which works only inside Marin's Iris cluster.
    - Give your model name with `--model NAME` (default `glm-5.3`) and its revision with `--model-revision REVISION`.
    - Put the API key of your endpoint in the `GLM_BULK_TOKEN` environment variable. The driver reads the key only from this variable. Do not put the key in a file in the repository.
4. **For FULL and production:** you need Iris access and personal CoreWeave object-storage credentials.
    - For Iris login, refer to [Iris authentication](../../../lib/iris/OPS.md#authentication-headless--ci).
    - Export `CW_KEY_ID` and `CW_KEY_SECRET` before you submit a job. Iris supplies them to the job as files. Refer to [Keep data on CoreWeave](../../../docs/tutorials/cloud-gpu.md#keep-data-on-coreweave).
5. **For executable-environment cohorts:** you also need a local Docker daemon.

Marin team members can check all access in one pass with the [`onboard-marin`](../../../.agents/skills/onboard-marin/SKILL.md) agent skill. Never print a token or a key in a log, a report or a commit.

## How a source becomes training data

```text
declare (datasets/<family>)                 convert                review               admit           select
open-thoughts/TaskTrove  ──pin──▶  normalize/*.parquet  ──panel──▶  GLM review +  ──▶  final/  ──▶  RECIPE.json  ──▶  RL run inputs
   or any HF/source repo                    manifest.json           controls           admitted                      train/validation Parquet
```

A source goes through five stages:

1. **Declare.** Each dataset is one [`RlDataSource`](source.py), the Python object that declares one source. The declaration is in a module under [`datasets/`](datasets). It pins the source to an immutable revision, for example a Hugging Face commit.
2. **Convert.** The [curation driver](driver.py) in `--mode quick` (QUICK mode) downloads the pinned inputs one time. Then it converts each selected row to a [`TaskSpec`](../../../lib/taskcompendium/src/taskcompendium/models.py) row in Parquet. A `TaskSpec` is the Marin record for one RL task. QUICK keeps each row that the converter rejects. QUICK runs on your local computer, with no cluster, review model or container image.
3. **Review.** `--mode sample` (SAMPLE mode) curates a panel for each source. The GLM model reviews the quality of the panel, and the pipeline executes [grader controls](../../../docs/references/task-curation.md#graders-and-controls) on it. `--mode full` (FULL mode) expands each source that its panel accepted. Then assessment grades a sample of the admitted tasks and of the excluded rows. Assessment never changes admission.
4. **Admit.** FULL writes the admitted tasks to `final/`. Then it publishes the run into the [`CurationCatalog`](../../../lib/finestore/src/finestore/rl/curation.py), the shared record of sources, campaigns and grades. The [RL Data Atlas](../../../docs/references/rl-data-atlas.md) does not read the catalog directly. The Atlas is a Postgres copy. It changes only when an operator runs the [Atlas loader](../../../docs/references/rl-data-atlas.md#updating-atlas) on a published snapshot.
5. **Select.** A [selection recipe](inventory.py) (`--selection RECIPE.json`) is a JSON file that names the sources for training. The selection step does decontamination: it removes tasks that overlap the heldout set, and duplicates. It also applies the quota of each family. Then it writes the canonical `task_spec` Parquet that training and evaluation read. The [quick RL loop](../../../docs/recipes/quick-rl-loop.md) shows the current recipe.

Declaring, converting or assessing a source does **not** approve it for training. A cohort is training data only when a selection recipe names it.

**Inventory-only sources.** Some releases already have their admission evidence in a pinned Hugging Face revision. You can declare such a release without a converter, as in [`campaign_releases.py`](datasets/campaign_releases.py). No campaign converts an inventory-only source. A FULL campaign only publishes its declaration. The source gets to the Atlas only through a published snapshot and a loader run. Each Atlas column gets its data from a different feed:

- **Quality** comes from a curation assessment. An inventory-only source has no assessment, so it always shows *Not assessed*.
- **Examples** comes from the curriculum release. The curriculum release counts the declared row count of an inventory-only source.
- **Curriculum area** shows only after the [embedding pass](../../../docs/references/rl-data-atlas.md#curriculum-coverage) places the tasks of the release.
- **Independent QC** shows the verdict of the release itself, from [`SourceInfo.independent_qc`](source.py).

## Ground rules for new contributors

- New source data needs a license review before it can go into a published release. Record the terms in the `SourceInfo` notes of the source.
- Use `--source` to limit each campaign pass to its sources. Use a new output root for each pass.
- Do not edit a published cohort in place. Add a versioned release, and record the provenance file that proves it.
- Keep private verifier bytes, gold patches and credentials in `private/` directories. Keep them out of all task exports.
- If you are not sure that data is training-approved, examine the selection recipe and the campaign issue. Do not trust the release title.
- Get explicit approval from a Marin maintainer before a `mirror://` copy of more than 10 GB. Refer to the [cross-region rules](../../AGENTS.md).

## Your first source

1. **Declare the source.** Follow the [steps to add a dataset](datasets/README.md#adding-a-dataset):
    1. Find the family module in [`datasets/tasktrove/`](datasets/tasktrove) that is nearest to your tasks. These modules cover competitive code, pytest-backed Python tasks, NL2Bash, math, SWE repositories and more.
    2. Add one `RlDataSource`. Give it a stable `SourceInfo.id` and a pinned [`SourceReference`](source.py).
    3. Set `config=CurationRecipe(...)` and `pipeline=`[`process_rows`](pipeline.py), or write a family-specific converter.
    4. Record the license terms in `SourceInfo.notes`.
    5. Return the source from the `sources()` function of the module.
    6. If the module is new, add its `sources()` to [`all_sources()`](sources.py). If you do not do this step, the source does not exist for the driver.
    7. For an inventory-only entry, use [`unconverted.py`](datasets/unconverted.py) or [`campaign_releases.py`](datasets/campaign_releases.py) instead.
2. **Test the conversion.** Add a behavioral conversion test to the test module of the family in [`experiments/post_training/task_curation/tests/`](tests), for example `test_tasktrove_code.py`. Run your test and the catalog checks:

   ```bash
   uv run --with-editable './lib/taskcompendium[pipeline]' pytest \
     experiments/post_training/task_curation/tests/test_<family>.py \
     experiments/post_training/task_curation/tests/test_catalog.py
   ```

3. **Find the invocation name of your source.** `--source` takes the invocation name (`config.name`), not the stable ID. For example, the source with the ID `MarinSkyRL:eurus2_code` has the invocation name `eurus2_code`. To list all invocation names, run [`runnable_sources()`](sources.py):

   ```bash
   uv run python -c 'from experiments.post_training.task_curation.sources import runnable_sources; print("\n".join(runnable_sources()))'
   ```

4. **Run QUICK on your computer.** Use the invocation name for `YOUR_SOURCE`. The output shows which rows convert and which rows the converter rejects:

   ```bash
   uv run --with-editable './lib/taskcompendium[pipeline]' python \
     -m experiments.post_training.task_curation.driver --mode quick --run \
     --source YOUR_SOURCE --output-root /tmp/task-curation-first
   ```

5. **Examine the QUICK result.** Open `manifest.json` for your source. Make sure that `input_rows` is equal to `converted_rows` plus the sum of `rejections`. Make sure that you expect each rejection reason.
6. **Run SAMPLE with review.** Set `GLM_BULK_TOKEN` first (refer to Before you start). This example uses a panel of 5 rows for a fast test. The default panel has 100 rows.

   ```bash
   uv run --with-editable './lib/taskcompendium[pipeline]' python \
     -m experiments.post_training.task_curation.driver \
     --mode sample --cache-dir ~/.cache/marin/rl-curation \
     --source YOUR_SOURCE --quality-sample-size 5 --assess \
     --review-mode chat --model-revision GLM_REVISION --base-url GLM_ENDPOINT --run
   ```

7. **Check your production access.** A production FULL campaign writes the shared catalog in `s3://marin-us-east-02a`. You need Iris access and CoreWeave credentials that can write to that bucket (refer to Before you start). Keep the same `--quality-sample-size` and `--seed` as in SAMPLE, so that FULL uses the same panel.
8. **Run the production FULL campaign** in an Iris job. `--publish-catalog` requires FULL with assessment and no `--cache-dir`. Remove `--run` to print the plan first. The plan shows each resolved default:

   ```bash
   uv run iris --cluster=marin job run --target-cluster YOUR-CW-REGION \
     --extra cpu --cpu 0.5 --memory 2GB --timeout 43200 \
     -e GLM_BULK_TOKEN "$GLM_BULK_TOKEN" -e HF_HUB_DISABLE_XET 1 \
     -- python -m experiments.post_training.task_curation.driver \
     --mode full --source YOUR_SOURCE \
     --review-mode chat --model-revision GLM_REVISION \
     --report-path s3://marin-us-east-02a/marin/rl-data/campaigns/YYYY-MM-DD-N/campaign.json \
     --publish-catalog s3://marin-us-east-02a/marin/rl-data/catalog/production --run
   ```

   The flags below are optional overrides.

    - `--review-cache PREFIX` is the object-store prefix where the driver caches the GLM review requests and judgments. Review needs one ([`pipeline.py`](pipeline.py)). The default is `<artifact-root>/rl-data/review-cache`. Each cache key hashes the model revision and the complete request, so a shared cache cannot return a judgment for a different prompt or model. To reuse the judgments of an earlier campaign, give its prefix.
    - `--worker-image IMAGE` is the container image that the Iris workers run. The default is the cluster's `default_task_image`, `ghcr.io/marin-community/iris-task:latest` ([`marin.yaml`](../../../lib/iris/config/marin.yaml)). The [Ops - Docker Images](../../../.github/workflows/ops-docker-images.yaml) workflow publishes this image every week. To record an exact image in the artifact identity, give a digest. To find the current digest, run `docker buildx imagetools inspect ghcr.io/marin-community/iris-task:latest`.
    - `--max-workers` (default 64), `--coordinator-memory` (default `16g`) and `--normalized-shards` (default 32) set the scale of the Iris job.

   For grader images and other flags, refer to [Run a campaign](../../../docs/references/task-curation.md#run-a-campaign) and the [reviewed local loop](README.md#reviewed-local-loop-on-real-sources).
9. **Read the campaign report.** Make sure that the campaign is successful before anyone loads the Atlas. A failed FULL campaign also publishes a snapshot.
10. **Load the Atlas.** Run the [Atlas loader](../../../docs/references/rl-data-atlas.md#updating-atlas):

    ```bash
    uv run python infra/marina/apps/rl_data_catalog/load.py \
      --app-id fb11c931-5861-4878-8bb5-a964d652b45f
    ```

    The loader needs a Marina login and read access to the catalog bucket. Any authenticated internal Marina user can also release the Atlas app ([Updating Atlas](../../../docs/references/rl-data-atlas.md#updating-atlas)).
11. **Add the source to training data.** To use the source for training, add it (or its cohorts) to a selection recipe.

## Authoring executable-environment cohorts

Read this section only if your tasks need a live container environment.

Some TaskTrove cohorts are not simple row conversions. SWE-Smith staged tasks and CalibForge fixtures need live environments and hidden-test separation. They also need positive and negative controls before you can trust a task. The pull request `[data] Add CalibForge fixture and staged SWE-Smith preparation` ([#10330](https://github.com/marin-community/marin/pull/10330)) added the entrypoints below. These entrypoints are campaign authoring tools, not launchers. They do not register a cohort as training-approved. Each one writes all its results under the `--output` directory that you choose.

**Example: pilot one staged SWE-Smith parent.** This procedure uses [`swesmith_staged`](swesmith_staged.py). It needs a local Docker daemon. The data directory of the original campaign was approximately 137 MB before the runtime logs.

1. Make a data directory that contains `swesmith-release-v1-20261016/` and `swesmith-split-v1-20261016/`. The tool checks the SHA-256 of each file against its `manifest.json`.
    - For `swesmith-release-v1-20261016/`, download the files of the public [SWE-Smith release](https://huggingface.co/datasets/open-athena/task-trove/tree/0e8137a6a1eca1e2c8c12eeb90a16c5d7f1f750c/releases/smartie-continuation-20261016/swesmith): `manifest.json`, `tasks.parquet` and the decision files.
    - For `swesmith-split-v1-20261016/`, download the folder of the same name from the public [split dataset](https://huggingface.co/datasets/open-athena/Snowball-Smartie-RL-SWE-Smith-Split-v1-20261016/tree/624ea3eba74037615085bcac4370a04e0b27bec9). It has `manifest.json`, `train.parquet` and `validation.parquet`.
    - To download both folders, run:

      ```bash
      uv run hf download open-athena/task-trove --repo-type dataset \
        --revision 0e8137a6a1eca1e2c8c12eeb90a16c5d7f1f750c \
        --include 'releases/smartie-continuation-20261016/swesmith/*' --local-dir /tmp/swesmith-download
      mkdir -p DATA_DIR && mv /tmp/swesmith-download/releases/smartie-continuation-20261016/swesmith DATA_DIR/swesmith-release-v1-20261016
      uv run hf download open-athena/Snowball-Smartie-RL-SWE-Smith-Split-v1-20261016 --repo-type dataset \
        --revision 624ea3eba74037615085bcac4370a04e0b27bec9 --local-dir DATA_DIR
      ```
2. Print the source inventory. Without `--pilot`, the tool builds nothing:

   ```bash
   uv run python -m experiments.post_training.task_curation.swesmith_staged \
     --data DATA_DIR --output /tmp/swesmith-staged-pilot
   ```

3. Pilot one parent. The `--pilot` value is `TASK_NAME:IMPORT_NAME:DEPENDENCY_COMMAND`. `TASK_NAME` is the parent ID after the prefix. `IMPORT_NAME` is the Python module that the setup must make importable. `DEPENDENCY_COMMAND` is the extra install command, and can be empty. The campaign validated parent `swesmith-23040` (repository glom) with this value:

   ```bash
   uv run python -m experiments.post_training.task_curation.swesmith_staged \
     --data DATA_DIR --output /tmp/swesmith-staged-pilot \
     --pilot swesmith-23040:glom:
   ```

4. Read `pilot-summary.json` in the output directory. Each parent has one status: `controls_validated`, `excluded` (task defect) or `infrastructure_failed` (inconclusive run). The campaign's own [`pilot-summary.json`](https://huggingface.co/datasets/open-athena/Snowball-Smartie-RL-Campaign-Artifacts-2026/tree/cbaf7daa55670c81eaf12aabbedc8b08c2ecf553/artifacts/data/swesmith-staged-v1-20261017) is a reference.

**Staged SWE-Smith** makes setup and repair-handoff variants from the qualified SWE-Smith source parents. These parents have the ID prefix `open-athena/task-trove:laion__swesmith-oracle-filtered-v2/` ([`SOURCE_PREFIX`](swesmith_staged.py)). Each variant stays in the split of its parent. Stage 1 tasks test repository setup. Stage 3 tasks start from a partial or complete repair and ask the agent to verify and submit.

1. [`swesmith_staged`](swesmith_staged.py) pilots one parent at a time (`--pilot`). It builds the shared base recipe and checks repository readiness with the Stage 1 contract. Then it runs the complete and partial Stage 3 controls. The command logs stay next to each attempt.
2. [`swesmith_staged_cohort`](swesmith_staged_cohort.py) extends the repositories of the validated pilots to all setup branches. It uses a shared private dependency cache. `cohort-stage1-decisions.json` gives each eligible parent an explicit status.
3. [`swesmith_staged_v2`](swesmith_staged_v2.py) builds shared-image Stage 3 handoffs. Each repository gets one flattened, test-free actor image and one fixed-reference verifier image. Each parent gets its own correctness evidence. Gold patches, mutation and gold overlays, and contracts stay under `private/`. They never leave the authoring host.
4. [`swesmith_staged_v2_glm`](swesmith_staged_v2_glm.py) drafts unverified handoff notes for qualified partials through the GLM bulk transport. Model text never replaces verifier evidence.
5. [`swesmith_handoff`](swesmith_handoff.py) validates the submitted workspace of an actor against the private contract, outside the rollout harness. It is a library module with no command-line entrypoint of its own.

**CalibForge** converts fixture recipes that have no dependencies into tasks that share one environment. It has these modules:

- [`calibforge_environment`](calibforge_environment.py): source inventory through the pinned Dockerfile parser.
- [`calibforge_fixtures`](calibforge_fixtures.py): public fixture bundles.
- [`calibforge_native_references`](calibforge_native_references.py): offline reference builds.
- [`calibforge_captured_fixtures`](calibforge_captured_fixtures.py): frozen `/app` fixtures.
- [`calibforge_equivalence`](calibforge_equivalence.py): full-filesystem comparison.
- [`calibforge_release`](calibforge_release.py): Harbor task packaging, pending admission.

All outputs stay `compiled_pending_equivalence_and_admission` or `not_run` until an operator executes admission.

The Docker images that these tools commit are host-local. Exports record the image IDs in `local-images.json`. These IDs have no meaning on a different computer until an operator publishes the images.

## Where the data lives

| Stage | Location | Notes |
| --- | --- | --- |
| Pinned source bytes | `open-thoughts/TaskTrove` at the declared revision | Downloads go to a cache under `~/.cache/marin`. To change the cache location, use `--cache-dir`. On CoreWeave, disable Xet downloads. In a shell, use `export HF_HUB_DISABLE_XET=1`. In an Iris job, add `-e HF_HUB_DISABLE_XET 1` before `--`. |
| Local QUICK outputs | The directory that you give in `--output-root` | Each source writes `normalize/*.parquet` and `manifest.json`. The run writes `campaign.json`. It writes `corpus.json` only with `--embed` or a remote output root. QUICK publishes nothing. |
| Production campaign artifacts | `s3://marin-us-east-02a/marin/rl-data/campaigns/<date>-<n>`, under the default `--artifact-root` ([`s3://marin-us-east-02a/marin`](driver.py)) | Keep each retained campaign in a new versioned directory. Artifact reuse follows explicit versions. |
| Published catalog | `.../rl-data/catalog/production`, written by `--publish-catalog` | Read it through [`finestore.rl.curation.CurationCatalog`](../../../lib/finestore/src/finestore/rl/curation.py). A failed FULL campaign also publishes a snapshot. Because of this, read the campaign report before you load the Atlas. |
| RL Data Atlas | [Marina](../../../infra/marina) app `fb11c931-5861-4878-8bb5-a964d652b45f` ([applets.marina.oa.dev](https://applets.marina.oa.dev/a/fb11c931-5861-4878-8bb5-a964d652b45f/)) | A Postgres copy of one catalog snapshot and the curriculum release. Only the [loader](../../../infra/marina/apps/rl_data_catalog/load.py) refreshes it. To query it, use `uv run marina apps sql fb11c931-5861-4878-8bb5-a964d652b45f "<SQL>"`. Merged app code goes live only after an explicit `marina release` (refer to the [loader guide](../../../docs/references/rl-data-atlas.md#updating-atlas)). |
| Admitted cohorts (public) | [`open-athena/task-trove`](https://huggingface.co/datasets/open-athena/task-trove) under `releases/<name>` and `campaigns/<campaign>` | Release files record download hashes and screening records. |
| Restricted cohorts | [`open-athena/task-trove-smartie-continuation-private`](https://huggingface.co/datasets/open-athena/task-trove-smartie-continuation-private) (access is restricted) | Source access restrictions keep some mixtures private, for example [TaskForge](../../../lib/taskforge/README.md) payloads. Training and evaluation Parquet for an exact run gets its own public data repository when it passes release screening. |
| Container-backed task images | The Docker daemon of the authoring host, recorded by `local-images.json` next to the export | Image artifacts are not published with the Parquet. Treat `local-artifact/...` locators as host-local. |

## Reference: Snowball Smartie campaign artifacts

The Smartie campaign ([issue #10305](https://github.com/marin-community/marin/issues/10305)) made the current staged SWE-Smith and CalibForge cohorts. Its working data was in the experiment directory `experiments/complete/snowball-smartie-RL/artifacts/data/`, which is not in the Git checkout. A screened public copy is in the [campaign artifact archive](https://huggingface.co/datasets/open-athena/Snowball-Smartie-RL-Campaign-Artifacts-2026/tree/cbaf7daa55670c81eaf12aabbedc8b08c2ecf553/artifacts/data), under `artifacts/data/`. The archive withholds raw traces and restricted task payloads. To reproduce the campaign, follow the archive's [`REPRODUCING.md`](https://huggingface.co/datasets/open-athena/Snowball-Smartie-RL-Campaign-Artifacts-2026/blob/cbaf7daa55670c81eaf12aabbedc8b08c2ecf553/REPRODUCING.md).

- `swesmith-staged-v1-20261017/`: pilots, cohort decisions and sealed variants for the setup stage.
- `swesmith-staged-v2-20261017/`: six validated repositories, 600 parents, and the Stage 3 export with 762 train and 16 validation tasks.
- `calibforge-*-v1-20261016/` and `calibforge-shared-v2-20261017/`: inventory, fixtures, native references and releases.
- `stage1-structural-v4-20261019/`: the assembled Stage 1 mixture manifests.

These copies are durably preserved:

- The exact training and validation Parquet of run `.32` (run `2026.10.15.32`): 69,487 train and 910 validation tasks, in the public [run data repository](https://huggingface.co/datasets/open-athena/Snowball-Smartie-RL-2026.10.15.32-Data/tree/cb728a7636caad3fc6fa6b6ba85a49194d9d69a3).
- The admitted cohorts in the [`open-athena/task-trove`](https://huggingface.co/datasets/open-athena/task-trove) releases: SWE-Smith (8,045), CalibForge (2,205), shared-environment CalibForge (324) and NL2Bash (996).
- In [`open-athena/task-trove-smartie-continuation-private`](https://huggingface.co/datasets/open-athena/task-trove-smartie-continuation-private): the private continuation mixtures and the `campaign-archive/2026.10.20` snapshot. The snapshot has 84 archives, including the Stage 3 exports. It does not include the runtime images.

The campaign reports and links are in issue #10305 and in the archive's [`LINKS.md`](https://huggingface.co/datasets/open-athena/Snowball-Smartie-RL-Campaign-Artifacts-2026/blob/cbaf7daa55670c81eaf12aabbedc8b08c2ecf553/LINKS.md).

## Case study: putting five existing releases in the Atlas

This case study is reference material for operators. In November 2026 the goal was small. Five Smartie releases were already on Hugging Face, with their counts and admission evidence published:

- SWE-Smith admitted: 8,045
- CalibForge admitted: 2,205
- CalibForge shared-environment: 324
- staged SWE-Smith v1: 1,699
- staged SWE-Smith v2: 778

The releases had to show in the RL Data Atlas with their example counts, curriculum areas and QC. No task data changed. The first declaration landed on Nov 3 at 21:18 UTC. The first Atlas load that listed the five rows finished on Nov 4 at 18:59 UTC, approximately 22 hours later. Those rows had no example count, no curriculum area and no QC.

The work had these phases:

- **First Atlas load (Nov 4):** 7 FULL submissions (campaigns `.1`–`.7`, none successful), 6 read-only catalog inventory jobs and 2 loader runs. The load used the snapshot of failed campaign `.7`.
- **Example counts and curriculum areas (Nov 5):** a curriculum release that the team loaded on Nov 5 at 13:27 UTC. It used the code of [#10924](https://github.com/marin-community/marin/pull/10924) before that PR merged on Nov 9: its manifest has the per-source `basis` field that #10924 added.
- **Clean load (after Nov 6):** 4 more FULL submissions (campaigns `.8` and `.9`, 1 successful) and 1 more loader run.
- **Independent QC (Nov 11):** a loader run on Nov 11 at 04:12 UTC, after [#11167](https://github.com/marin-community/marin/pull/11167), showed Independent QC for all five rows.

The fixes were in [Iris](../../../lib/iris) job submission, [rigging](../../../lib/rigging) credential resolution, the [Zephyr](../../../lib/zephyr) Parquet reader, the campaign driver, assessment grading and catalog publication. The effort made five PRs, merged from Nov 4 to Nov 10: [#10891](https://github.com/marin-community/marin/pull/10891), [#10899](https://github.com/marin-community/marin/pull/10899), [#10919](https://github.com/marin-community/marin/pull/10919), #10924 and #11167.

The FULL campaign had only one function: to publish a catalog snapshot that contains the new declarations. The five inventory-only sources have nothing to convert. For this reason, the campaign ran on the smallest runnable source in the catalog, `snowball-v01-structure-closure-validation` (243 rows).

| # | Step or attempt | What happened | Resolution |
| --- | --- | --- | --- |
| 1 | Declare the five releases | Five inventory-only `RlDataSource` entries, pinned to exact Hugging Face revisions. | [`datasets/campaign_releases.py`](datasets/campaign_releases.py), [#10891](https://github.com/marin-community/marin/pull/10891) (merged Nov 4) |
| 2 | Campaigns `.1`–`.5` (first two attempts) | The attempts stopped in the `stage-workdir` init stage. `--egress-policy egress_policy_internet` cut the path from the pod to the code bundle of the controller. | Removed the flag. The documented submission does not use it. |
| 3 | Campaigns `.1`–`.5` (other attempts) | The attempts stopped approximately 50 s after pool start. FULL starts Zephyr workers with the privileged container profile, because local [bubblewrap](https://github.com/containers/bubblewrap) graders need it. Iris gives that profile only when the parent task has it. The driver created the coordinator without it. | The driver gives the worker profile to `coordinator_resources` (in #10891). |
| 4 | Campaigns `.6`, `.7` | The attempts reached execution. Then each scatter-merge read in the worker pods failed with an instance-metadata 404. Since [#10474](https://github.com/marin-community/marin/pull/10474), Iris supplies storage keys as files. The Rust object-store client of Polars reads only environment variables, so it used instance metadata. `-e CW_KEY_ID`/`-e CW_KEY_SECRET` did not help. The values got to the driver but not to the worker jobs that the coordinator created. | The rigging library resolves the credentials of the bucket from the file first, and [`zephyr.parquet_scan`](../../../lib/zephyr/src/zephyr/parquet_scan.py) gives them to Polars: [#10899](https://github.com/marin-community/marin/pull/10899) (merged Nov 5). |
| 5 | Side effect of `.6`, `.7` | A failed FULL campaign also publishes its snapshot. Production `CURRENT` pointed at the snapshot of `.7`. That snapshot had the five new rows correct, and one failed-attempt row for the validation source. | Loaded it, with approval. A clean campaign replaced the row later. |
| 6 | Read the production catalog | No in-cluster inventory job finished cleanly. Two of six were killed and four failed. Some failed in approximately 40 s. The last two read the catalog and then stopped on script bugs. CoreWeave rejected a local `s3fs` read that did not have the path-style bucket configuration of rigging. | Used the output that the last two jobs printed before they stopped. |
| 7 | First Atlas load | A transient Marina 502 occurred before the atomic pointer swap. The loader published no partial data. | Tried again. The load was successful. |
| 8 | Campaign `.8`, first two submissions | Two CLI failures occurred before any work started. First, `iris job run -e` now takes `KEY VALUE` pairs, so `-e A -e B` parsed the second `-e` as a value. Second, `KeyError: IRIS_TERMINATION_GRACE_MS` occurred. The newer client from latest main expected a controller field that the deployed cluster did not have. | Corrected the `-e` usage. Submitted from a worktree that matched the deployed controller, with the open fixes cherry-picked in. |
| 9 | Campaign `.8`, third submission | The campaign went through the Parquet stages for the first time. Then it failed at review grading. 2 of 243 GLM judgments were unparseable. One returned `evidence_ids` as a stringified JSON array. One was the reviewer reply "WRONG TASK ID — STOP" to a deliberate trap task. One unavailable judgment caused the whole campaign to fail. The shared review cache would return the same two outputs on a retry. | The grader now coerces stringified `evidence_ids`. It tolerates unavailable judgments up to [`MAX_UNAVAILABLE_JUDGMENT_SHARE`](../../../lib/finestore/src/finestore/rl/curation.py) (2%): [#10919](https://github.com/marin-community/marin/pull/10919) (merged Nov 6). |
| 10 | Campaign `.9` | Successful. It published a clean snapshot to production (report `campaign-releases-2026.11-04-9`). New GLM attempts graded the trap task cleanly, so the campaign did not use the new tolerance. | — |
| 11 | Second Atlas load | 214 sources, 147 assessments and 12,212 items, all reviewed. The five rows are present, and the validation row is `complete`. | Verified with `marina apps sql`. |
| 12 | The rows in the Atlas | **Quality: Not assessed**, no curriculum area and no example count. The rows have no curation assessment, and the curriculum release had no entries for them. | [#10924](https://github.com/marin-community/marin/pull/10924) (merged Nov 9) counts declared rows in the curriculum release and adds the **Independent QC** column. The area needed rows 13–14. |
| 13 | Place the five releases in the curriculum | The team staged the public task text from the pinned Hugging Face revisions. Then the team ran the embedding pass on it, CPU-only, with no GLM inference. A calibration with 200 rows for each source projected 8.4 h for all 13,051 tasks. The full run took 34.5 min. The projection scaled the fixed model-load time with the corpus size. | [`load_curriculum.py`](../../../infra/marina/apps/rl_data_catalog/load_curriculum.py) selected the new release (13,051 tasks, 310 sources) with one pointer swap. |
| 14 | Placement quality | Each row has a subject-level area: Computer & Information Sciences. Most SWE-Smith tasks are outside the curriculum. 6,245 of 8,045 (78%) are out of scope, because their instructions are repository-setup boilerplate. The other four rows have 0.3% (staged v1) to 16% (staged v2) out of scope. Only the CalibForge rows have capability-level placements (67 and 15). | The area for SWE-Smith comes from its 1,800 placed tasks. |
| 15 | Independent QC | The declarations of #10924 have the QC of each release. But the production catalog still had the Nov 4 declarations, where `independent_qc` is empty. Catalog publication kept the previous declaration of each source that had no assessment attempt in the campaign. An inventory-only source can never have an attempt, so a FULL campaign could not refresh it. On Nov 10 the Atlas app was released at version 71 with the new column. The column was empty for these rows. | [#11167](https://github.com/marin-community/marin/pull/11167) (merged Nov 10) changes [`publish_catalog`](catalog_snapshot.py). Publication now takes the current code declaration of each inventory-only source. The Atlas load of Nov 11, 04:12 UTC, shows Independent QC for all five rows. |

Each failure above occurred after the data was already correct and public. None of the failures had a relation to the five releases. The fixes were open PRs during the campaign runs. For this reason, those runs used a worktree that combined the branches.

### What to expect if you do this

- **Plan time for infrastructure.** Expect most attempts to fail on job submission, credentials or grading before they touch your data. Use a new report directory for each attempt (`campaign-releases-<date>-<n>`).
- **Inventory-only sources also need a FULL campaign.** The campaign publishes the declaration. An inventory-only source in code has no effect until a FULL campaign publishes a snapshot that contains it.
- **An edit to an inventory-only source needs a new publication.** Before #11167, a later edit, such as adding `independent_qc`, did not get to the production catalog (row 15). Now the edit gets there at the next FULL campaign publication.
- **Read the campaign report before you load.** A failed campaign can also update production `CURRENT`.
- **Check credentials from end to end.** Keys that you give to the driver pod do not always get to the worker pods that its coordinator starts. Make sure that the worker jobs receive them.
- **Use a client that matches the cluster.** An `iris` client from latest main can expect controller fields that the deployed cluster does not have.
- **Plan each Atlas column for an inventory-only source.** The declaration gives a listed row. The example count needs a curriculum release that includes the declared count. The curriculum area needs the embedding pass over the task text of the release. Independent QC needs `independent_qc` in the published declaration. Quality stays *Not assessed*.
- **Expect weak placement for boilerplate instructions.** Most tasks whose prompts describe repository setup are out of scope (78% for SWE-Smith).
- **Measure throughput after the fixed costs.** Model loading takes most of the time in small calibration runs. Calculate the projection from the rate for each task.
- **Release the app after its code merges.** A merge of Atlas code changes nothing in production. Run `marina release` for the app from a clean checkout at current main.
- **A retry of a failed review step is not neutral.** The shared review cache returns the same judgments. A retry fails in the same way, unless the grader or the cache key changes.

## Glossary

| Term | Meaning |
| --- | --- |
| RL source | One population of RL tasks, declared as one [`RlDataSource`](source.py) with a stable ID. [`all_sources()`](sources.py) collects all sources. |
| Stable ID | The `SourceInfo.id` of a source, for example `MarinSkyRL:eurus2_code`. It links the Atlas history of the source. |
| Invocation name | The `config.name` of a runnable source, for example `eurus2_code`. `--source` takes this name. |
| Task | One RL problem, stored as one [`TaskSpec`](../../../lib/taskcompendium/src/taskcompendium/models.py) row: its context, environment requirements, answer format and grader. |
| Family | The task-type group of a source (`SourceInfo.family`). A selection recipe sets a quota for each family. |
| Cohort | A named set of admitted tasks that Marin publishes together. A release is one published, versioned folder of a cohort, for example in [`open-athena/task-trove`](https://huggingface.co/datasets/open-athena/task-trove). |
| Campaign | One run of the curation driver in a [mode](../../../lib/taskcompendium/src/taskcompendium/pipeline/source_processing.py): QUICK (convert only), SAMPLE (review a panel) or FULL (expand, verify, admit and publish). Informally, also a program of work, for example the Snowball Smartie RL campaign. |
| Smartie | The Snowball Smartie RL campaign: multi-domain RL on the Snowball 67B-A2B model ([issue #10305](https://github.com/marin-community/marin/issues/10305)). |
| Inventory-only source | A source with no converter or pipeline. No campaign converts it. A FULL campaign only publishes its declaration. Refer to [sources without conversion recipes](datasets/README.md#sources-without-conversion-recipes). |
| Campaign Release | A type of inventory-only source whose ID starts with `Campaign Release:`. It points at a cohort that a campaign published outside the curation driver ([`campaign_releases.py`](datasets/campaign_releases.py)). |
| Panel | A sample of rows from each source that SAMPLE and FULL review (default 100; `--quality-sample-size`). |
| Admission | The decision to put a row in `final/`. The values are `admitted`, `rejected`, `deferred`, `no_grader` and `unverified` ([outputs](../../../docs/references/task-curation.md#outputs)). |
| Controls | Grader checks that run before admission. A golden (positive) control must score. A negative or empty control must not ([graders and controls](../../../docs/references/task-curation.md#graders-and-controls)). |
| Oracle | The reference solution of a task. Archived oracle scripts run in a sandbox to make the golden control answers. |
| Assessment | A GLM grade for a sample of admitted tasks and excluded rows. It never changes admission. |
| Trap task | A task whose text is deliberately adversarial, for example a task ID that does not match ([#10919](https://github.com/marin-community/marin/pull/10919)). The reviewer must still return a normal judgment. |
| QC | Quality control. *Quality* in the Atlas is the curation assessment. *Independent QC* is the verdict that a release recorded itself, in [`IndependentQC`](../../../lib/finestore/src/finestore/rl/curation.py). |
| Snapshot | One published state of the [`CurationCatalog`](../../../lib/finestore/src/finestore/rl/curation.py). Each terminal campaign publishes one snapshot, also when the campaign fails. |
| `CURRENT` | The catalog pointer to the latest published snapshot. The Atlas loader reads the snapshot that `CURRENT` names. |
| Atlas | The [RL Data Atlas](../../../docs/references/rl-data-atlas.md), a Marina app that shows a Postgres copy of one snapshot and the curriculum release. |
| Curriculum release | The output of the [embedding pass](../../../docs/references/rl-data-atlas.md#curriculum-coverage). It places tasks in curriculum areas and gives the Atlas its example counts. |
| Selection recipe | A [JSON file](recipes/snowball-100k.json) that names the sources for training, sets the family quotas and pins the heldout set ([final task selection](../../../docs/references/task-curation.md#final-task-selection)). |
| Heldout set | The pinned evaluation tasks that a selection recipe names. Training data must not overlap them. |
| Decontamination | The selection step that removes tasks which overlap the heldout set, and duplicates across sources. |
| Stage 1, Stage 3 | Staged SWE-Smith task types. A Stage 1 task tests repository and workspace setup. A Stage 3 task starts from a partial or complete repair, and the agent must verify and submit. |
| TaskForge | A Marin package that makes `TaskSpec` tasks from task ideas with model help ([`lib/taskforge`](../../../lib/taskforge/README.md)). Some TaskForge inputs have access restrictions. |
| Operator | The person who runs a production campaign or the Atlas loader. A campaign needs Iris access and CoreWeave credentials. The loader needs a Marina login and read access to the catalog bucket. |
| Marina | The Marin internal app platform that hosts the Atlas ([`infra/marina`](../../../infra/marina)). |
| Iris, Zephyr | Iris runs jobs on the clusters ([`lib/iris`](../../../lib/iris)). Zephyr runs the dataset processing workers ([`lib/zephyr`](../../../lib/zephyr)). |
