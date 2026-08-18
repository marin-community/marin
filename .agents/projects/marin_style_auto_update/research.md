# Marin-style auto-update research

## Background Research Brief

- Effort: medium
- Stop rule: implementation, live repository policy, and GitHub's documented App/ruleset model converged on one design.
- Date: 2026-08-18

### Question

How can `marin-style` remain exactly pinned and vendored in Marin-community repositories without requiring a person to prepare, approve, and merge the same mechanical update in every consumer?

### Current Marin context

The initial Echo rollout landed in `marin-style`, Harbor, TPU inference, vLLM, Evalchemy, Axolotl, and MarinSkyRL. Each consumer pins one exact `marin-style` commit in `infra/pre-commit.py` and one or more workflows. MarinSkyRL also pins it in `pyproject.toml` and `uv.lock`. All consumers vendor the same 20 files under `.agents/marin-style/` and nine named `.agents/skills/` subtrees.

The source package intentionally uses immutable revisions. The [consumption contract](https://github.com/marin-community/marin-style/blob/67f567027056ba465468e67dfb0ee59c07c1d0ce/README.md#consumption-model) says every contributor and CI run should use the same revision. Switching consumers to `@main` would remove the update PRs but would also make a consumer's behavior change without a commit in that repository.

### Internal prior work

Marin already runs two unattended dependency PR workflows with a dedicated GitHub App. The [external dependency workflow](https://github.com/marin-community/marin/blob/954a5251a2bee3cd712efcd463eaee85505f8783/.github/workflows/ops-external-dependencies.yaml#L1-L100) mints a short-lived installation token, resets a stable automation branch, publishes an allowlisted change, waits for fixed checks, and merges it.

The reusable lifecycle in [`dependency_update.py`](https://github.com/marin-community/marin/blob/954a5251a2bee3cd712efcd463eaee85505f8783/scripts/ci/dependency_update.py#L91-L153) validates PR author, base, branch, title, head SHA, changed files, and required checks before merging. It uses force-with-lease when refreshing an existing automation branch and passes the expected head commit to `gh pr merge`. The current policy type is repository-local: [`PullRequestPolicy`](https://github.com/marin-community/marin/blob/954a5251a2bee3cd712efcd463eaee85505f8783/scripts/ci/dependency_update_policy.py#L16-L58) has a fixed file set and one global set of Marin checks.

The App's IAC separates review bypass from CI. [`dependency_updater.py`](https://github.com/marin-community/marin/blob/954a5251a2bee3cd712efcd463eaee85505f8783/infra/pulumi/src/iac/github/dependency_updater.py#L116-L145) grants the App a pull-request-only bypass in the review ruleset while the required-CI ruleset keeps only the organization-admin bypass. The private key is stored in a main-only Actions environment. Echo incident [wiki:107](https://echo.oa.dev/wiki/107) records an earlier rollout failure where ruleset bypass alone was insufficient because classic branch protection independently required review.

`marin-style sync` already computes the authoritative packaged destinations in [`vendor.py`](https://github.com/marin-community/marin-style/blob/67f567027056ba465468e67dfb0ee59c07c1d0ce/src/marin_style/vendor.py#L54-L76), then renders and writes them. It does not record a manifest or delete assets removed from a later release. Its check mode reports missing or changed current assets, but cannot report obsolete generated files.

### Consumer inventory

| Consumer | Explicit pin files | Required checks | Special handling |
| --- | --- | --- | --- |
| Harbor | `infra/pre-commit.py`, `marin-ci.yaml`, `marin-nightly.yaml` | `harbor-config`, `marin-precommit`, `marin-style-sync`, `tests` | None |
| TPU inference | `infra/pre-commit.py`, `marin-ci.yaml`, `marin-e2e-nightly.yaml` | `cpu-tests`, `lint` | Preserve all upstream workflows |
| vLLM | `infra/pre-commit.py`, `marin-ci.yaml`, `marin-nightly.yaml` | `delta-smoke`, `marin-precommit` | `.claude/skills` is upstream-owned |
| Evalchemy | `infra/pre-commit.py`, `marin-ci.yaml`, `e2e-nightly.yaml` | `harness`, `marin-precommit`, `marin-style-sync` | Preserve `.agents/ops` |
| Axolotl | `infra/pre-commit.py`, `marin-ci.yaml` | `marin-style`, `tests` | No nightly pin |
| MarinSkyRL | `infra/pre-commit.py`, `cpu_ci.yaml`, `marin-nightly.yaml`, `pyproject.toml`, `uv.lock` | `lint`, `skyrl_gym_tests`, `skyrl_train_tests` | Run `uv lock --upgrade-package marin-style` |

The generated allowlist must be narrower than `.agents/**`: several consumers have local operations, project, or skill files. The updater can allow the exact old and new `marin-style` manifests plus tracked references in `infra/pre-commit.py`, workflow YAML, and the root `pyproject.toml`. A root `uv.lock` is generated only when it contains the pinned `marin-style` package. It must not touch `AGENTS.md`, `.claude/skills`, unrelated workflows, or TPU inference's repaired nightly test.

The initial rollout pinned PR head `5094279…`; the resulting source squash commit is `67f5670…`. Their trees are equal, but future updates should resolve and pin the reachable `marin-style` default-branch SHA after merge.

### External prior art

GitHub's [`create-github-app-token`](https://github.com/actions/create-github-app-token/blob/main/README.md) supports installation tokens limited to an explicit owner and repository list. Tokens expire after one hour and should request only the permissions used by the workflow. Every consumer pins `marin-style` under `.github/workflows/`, so the publication token needs Workflows write in addition to Contents and Pull requests write.

GitHub [repository rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets) support GitHub App bypass actors and a pull-request-only bypass mode. App registration and installation repository selection remain owner-managed in Marin because GitHub's selection endpoint needs user-scoped authority unsuitable for an unattended Pulumi run.

### Negative and failed leads

- Tracking `marin-style@main` removes repository churn at the cost of reproducibility and local audit history.
- Putting an updater workflow in every consumer duplicates credentials and policy and recreates the maintenance problem.
- A second Marin registry for these repositories duplicates `config/external` for five forks and still needs an Axolotl exception. Pin paths and CI names then drift in two places.
- Broadly replacing a SHA across a checkout can modify upstream-owned workflows in fork repositories.
- Allowlisting `.agents/**` can absorb changes to consumer-owned operations, projects, and custom skills.
- Letting the updater App bypass required CI turns a compromised generator or credential into an unchecked merge path.
- Current `changed_worktree_files()` uses `git diff --name-only`, which misses newly generated untracked files.

### Evidence map

#### Claim: the existing PR lifecycle is reusable

- Support: the helper already validates immutable PR identity, an explicit changed-file boundary, required checks, and the merge head SHA.
- Caveat: the current helper names Marin's checks explicitly.
- Confidence: high.
- Action: retain dynamic file policies and read the consumer's required rows through `gh pr checks --required`.

#### Claim: no consumer registry is needed

- Support: the App installation already limits repository access. Every consumer has one canonical pin in `infra/pre-commit.py`, and the remaining direct references use recognizable workflow or project syntax.
- Caveat: App installation selection and protection audits remain owner-managed.
- Confidence: high.
- Action: build the matrix from installation repositories, exclude Marin itself, and reject a checkout without the canonical pin.

#### Claim: exact pins should remain

- Support: the source README makes exact revision pinning part of the consumer contract; generated Echo clients also embed the installed revision.
- Contradictions: trusted shared composite actions already track `main`, but they are a narrower delivery channel and do not determine local agent discovery or lint behavior.
- Confidence: high.
- Action: automate the PR lifecycle instead of making the source reference mutable.

#### Claim: a generated manifest is needed

- Support: current sync knows new destinations but cannot identify assets deleted by a release; consumers also own adjacent `.agents` paths.
- Confidence: high.
- Action: have `marin-style sync` persist exact managed paths and hashes, prune only unchanged stale outputs, and fail on modified stale outputs.

### Source ledger

| Source | Type | Claim used for | Confidence |
| --- | --- | --- | --- |
| Marin dependency update workflow and helper at `954a5251…` | Marin code | Existing fail-closed PR lifecycle | High |
| Marin dependency updater IAC at `954a5251…` | Marin code | App credential and bypass separation | High |
| `marin-style` vendor implementation and README at `67f5670…` | Marin code | Exact pins, generated destinations, pruning gap | High |
| Echo wiki 107 | incident record | Ruleset and classic protection both require bypass | High |
| GitHub App token action README | official docs | Repository-limited installation tokens | High |
| GitHub ruleset documentation | official docs | Pull-request-only App bypass | High |

### Handoff

The smallest safe implementation is a manual workflow in Marin, installation-based consumer discovery, a `marin-style` managed manifest, and the existing generated-PR lifecycle with a dynamic file boundary. Activation follows the reviewed manifest bootstrap, App installation expansion, and review-only bypass audit for both active protection layers.
