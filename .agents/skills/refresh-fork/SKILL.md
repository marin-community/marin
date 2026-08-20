---
name: refresh-fork
description: Rebase one Marin fork onto newer upstream per its config/external/migration.toml descriptor, run the fork's e2e, and open the Marin PR or file a blocker issue.
---

# Refresh a Fork

Read `AGENTS.md`. Each pin is a `marin-community` fork: rebase its overlay onto
the selected upstream base on `<branch>-next`, validate that staged tip with the
fork e2e, re-pin Marin, and prepare protected-branch promotion. Never
force-move a stable branch. A `group` refreshes all its sections in one PR; the
only current group is `vllm`/`tpu-inference`, while `vllm-gpu` is independent.

In local/manual mode ask before pushing fork branches, opening a Marin PR, or
filing an issue; the required e2e needs no extra confirmation. The weekly
`ops-fork-ferry` request supplies this skill and target pin/group.

## Descriptor and outcome

Read every target section in `config/external/migration.toml` before acting:
`upstream`, optional `group`, `base_select`/`derived_from`, `pin`, `branch`,
`e2e`, `blocker_assignee`, and `nuances`. The named pin source holds the actual
revision.

- If the selected base and metadata are current, exit with a no-op.
- On success, after e2e passes, create rollback/date tags from
  `docs/promotion-protocol.md` and exactly one draft Marin PR for the pin/group.
  Name the pending `<branch>-next` → `<branch>` admin promotion, request
  `blocker_assignee` as reviewer, and monitor via `commit`.
- On an unresolved blocker, do not open a PR. Create/update one issue titled
  `Fork refresh blocked: <fork> — <reason>` assigned to `blocker_assignee`, with
  current pins, selected base, SHAs/branches, fixes attempted, failure, and
  artifacts.

Use `/tmp/marin-fork-refresh/<run-id>` for scratch. Clone the repository named
by the pin source and configure remotes exactly once:

```sh
git clone <repository> <fork>
git -C <fork> remote add upstream <upstream>
git -C <fork> fetch --tags --multiple origin upstream
git -C <fork> remote set-head upstream -a
```

Keep notes of base evidence, SHAs, validation, compatibility traps, and
carry/drop/fix decisions for the PR or blocker.

## Select the base

- `upstream_main` (`evalchemy`, `harbor`, `MarinSkyRL`, `vllm-gpu`):
  `upstream/HEAD` (vLLM GPU tracks upstream head independently of TPU).
- `latest_release` (`tpu-inference`): newest GitHub Release with
  `draft == false`, `prerelease == false`, and exact `vMAJOR.MINOR.PATCH`; use
  its resolved SHA, not a raw tag/branch.
- `derived` (`vllm`): use the SHA at
  `tpu-inference:.buildkite/vllm_lkg.version` from that selected release and
  verify it resolves in the fork's upstream; inspect TPU dependency metadata.

Do not fall back to an older release when the latest eligible one fails. If an
isolated fork's upstream base has not moved, it is a no-op even if Marin's pin
lags the fork's `main`; that lag belongs to the daily external-dependency bump.

## Rebase the overlay

Stage `<branch>-next` (`tpu-next` for the vLLM TPU pin, `main-next` otherwise)
from `new_base`; it is disposable and may be force-updated on rerun. Keep the
protected stable `<branch>` unchanged. Set `old_base` to descriptor
`upstream_base`, or `git merge-base <fork>/<branch> upstream/HEAD` for isolated
and release pins. Set `old_tip` to fork `main` for isolated pins (cover all
patches even if Marin lags), or the stable pin tip for descriptor/release pins.

Inventory and classify in order:

```sh
git log --reverse --no-merges old_base..old_tip
git range-diff old_base..old_tip new_base..<new_tip>
```

- `carry`: still needed and not upstreamed;
- `drop`: upstream absorbed, obsolete, temporary, or a stale version ceiling;
- `fix`: intent remains but needs re-authoring against the new layout.

Check the new base for each patch's symbols/APIs before carrying it. Replay
only carry/fix commits in logical order; omit merge commits. Separate cascade
conflicts in fork-new files from real conflicts in paths shared by both bases.
Audit every overlay call site (signatures, constructors, attributes, removed
helpers, and tests) against `new_base`; textual cherry-pick success is not API
validation. Keep retained commit rationale and drop conditions, remove conflict
artifacts/unrelated refactors, and collapse superseded fork-infra churn.

Stop and file the blocker instead of forcing a PR when the overlay is non-linear,
upstream refactored touched files requiring substantial re-authoring, or many
core files conflict. Never reconstruct an overlay from a net diff; replay the
fork's real commits.

## Re-pin the staged tip

Point Marin at `<branch>-next`, run `uv run config/update-external.py`, and
confirm only intended pins change. Respect `nuances`; fixed-base overlay edits
are a separate workflow (`docs/overlay-only-pr.md`).

- `descriptor:<path>#<section>` (`vllm`, `tpu-inference`): push `<branch>-next`,
  set `commit` and `upstream_base` in `vllm/tpu.toml`. Do not touch
  `marin-core`, `marin-levanter`, or `marin-fray`; dependencies resolve in the
  fork's `uvx` environment.
- `release:<path>` (`vllm-gpu`): push `main-next`; build and promote the exact
  wheel through the fork, then let the helper write the manifest pin:

  ```sh
  gh workflow run marin-gpu-candidate.yaml --repo marin-community/vllm --ref main-next
  gh workflow run marin-gpu-release.yaml --repo marin-community/vllm --ref main-next -f candidate_tag=<tag>
  gh release download <release_tag> --repo marin-community/vllm --pattern marin-vllm-gpu-manifest.json
  uv run config/update-external.py --promote-gpu-release marin-vllm-gpu-manifest.json
  ```

  Wait for the candidate prerelease before promotion; the release workflow
  validates exact wheel bytes on real GPUs and publishes the immutable manifest.
  Do not hand-edit `gpu.toml`; the helper writes release tag, source SHA,
  version, torch backend, URLs, hashes, and `external_dependencies.py`.
  Re-audit the wheel verifier/release gate across CUDA/torch or stable-ABI
  boundaries (CUDA 13 extension: `vllm._C_stable_libtorch`).
- `isolated_project` (`evalchemy`, `harbor`, `MarinSkyRL`): point the uv source
  at `main-next`, run `uv run config/update-external.py <fork>` to lock the
  staged SHA, and keep the draft PR on `main-next` until admin promotion. Then
  restore source to `main`, relock, and verify the lock still records the tested
  SHA before ready/merge.

## Validate

Run the fork's own suite from its CI configuration verbatim; do not invent a
marker subset. Run every CI step in order. If local Docker bind mounts or
compiled CUDA/TPU dependencies prevent a behavioral run, use fork CI and label
local `py_compile`/conflict-marker checks structural only. Never downgrade an
upstream dependency floor to make a golden pass; inspect the deciding code path.

Then run the descriptor e2e before creating promotion tags or the Marin PR:

- `vllm`/`tpu-inference`: bounded brokered Qwen3 TPU smoke on Iris `marin`,
  interactive, `v6e-4`, `europe-west4`:

  ```sh
  uv run iris --config lib/iris/config/marin.yaml job run \
    --job-name served-qwen3-<run-id> --cpu 1 --memory 2G --extra cpu \
    --priority interactive --no-wait -- python -c \
    "from dataclasses import replace; from fray.types import ResourceConfig; from marin.execution.lazy import lower; from marin.execution.step_runner import StepRunner; from experiments.evals.lm_eval_suite import lm_eval_suite; from experiments.evals.served_qwen3 import QWEN3_TPU_INFERENCE; inference = replace(QWEN3_TPU_INFERENCE, iris=replace(QWEN3_TPU_INFERENCE.iris, worker_resources=ResourceConfig.with_tpu('v6e-4', ram='96g', regions=['europe-west4']))); StepRunner().run([lower(lm_eval_suite(inference, model_name='qwen3-0.6b-refresh-smoke', version='<run-id>-dev', limit=8))])"
  ```

- `vllm-gpu`: run the CoreWeave H100 Snowball parity gate with exact flags:

  ```sh
  uv run pytest tests/cluster/vllm/test_snowball_backend_parity.py \
    -m cluster -o addopts= --import-mode=importlib -vv -s
  ```

  Also run a bounded Qwen3 0.6B GPU serve-and-eval through
  `QWEN3_GPU_INFERENCE` on the Marin federation hub. Require successful broker
  and proxy startup, served completions, metrics, and sample output; Snowball
  parity alone does not validate the consuming serving path.

- `evalchemy` / `harbor`: bounded live evals using their respective config:

  ```sh
  uv run python -m experiments.evaluation.cli launch --model qwen3-0.6b --limit 8 \
    --evalchemy-config experiments/evaluation/configs/evalchemy/gsm8k-smoke.yaml
  uv run python -m experiments.evaluation.cli launch --model qwen3-0.6b --limit 8 \
    --harbor-config experiments/evaluation/configs/harbor/aime-smoke.yaml
  ```

  `--dry-run` is wiring only and does not validate the fork.
- `MarinSkyRL`: `uv run python -m experiments.post_training.iceball_micro
  --stage evaluation --version dev --run`; `--version` and `--run` are
  required to execute the terminal stage.

When an e2e fails, rerun the same workload on current Marin pins with the same
target/priority. Only a refreshed-only failure is a refresh defect. If the old
stack is already broken, hold the PR and file/link a blocker.

## Promote and publish

After e2e success, create and verify the rollback tag for the stable tip and a
date tag for the staged tip using `docs/promotion-protocol.md`; leave stable
branches unchanged. Keep the Marin PR draft until each admin hard swap is done.
For isolated projects, restore `main` and relock after promotion. Run
`review-pr`, then `./infra/pre-commit.py --review`, resolving every finding.

Open one draft PR through `commit`, request `blocker_assignee`, and follow its
monitoring loop. Include fork/base/staged SHA, rollback/date tags, pending
promotion (and GPU release tag), e2e result, risks, and carry/drop/fix evidence.

Done means the named pin and generated dependencies carry the validated tip,
staging/stable tags have the required meanings, retained/dropped patches are
documented, e2e passed (or a blocker exists), and the opened PR reaches the
`commit` monitoring exit condition.
