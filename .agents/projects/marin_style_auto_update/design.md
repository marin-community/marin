# Automate marin-style consumer updates

Keep `marin-style` exactly pinned in each Marin-community repository while removing the repeated human work needed to prepare, approve, and merge mechanical bumps. A central workflow will update every installed consumer after a source change, wait for that repository's required CI, and merge only a generator-owned diff. A failed or surprising update remains open for a person to inspect.

The research and repository inventory are in [research.md](research.md). Marin already has most of the security-sensitive machinery: a dedicated GitHub App, a main-only credential environment, fixed automation branches, PR identity validation, file allowlists, required-check gates, and head-SHA-bound merges. This proposal extends that mechanism across repositories instead of adding updater credentials and workflows to every fork.

## Challenges

The consumers do not pin `marin-style` in the same places. Most have a shim and two workflow references; MarinSkyRL also has a project dependency and lockfile. Upstream-tracking forks contain workflows and agent assets that Marin must preserve byte-for-byte.

Generated assets need a precise ownership boundary. `marin-style sync` currently writes the package's current assets but has no record of past outputs, so removing a packaged skill would leave an obsolete file behind. Allowlisting `.agents/**` would be unsafe because consumers maintain adjacent operations, project, and custom-skill files.

Auto-merge also crosses two independent GitHub controls. The updater App needs a review bypass in each consumer's ruleset and classic branch protection, where present, but must still satisfy all required CI. App installation repository selection is an owner-managed operation.

## Costs / Risks

- The App installation becomes the consumer allowlist. An owner must update that selection when adding or removing a consumer.
- The existing dependency-updater App gains access to six more repositories, limited to contents, workflows, and pull requests.
- A `marin-style` release can produce several PRs at once. Failures remain open and require diagnosis.
- Activation requires one final manual bootstrap update in each consumer so every repository has a trusted generated manifest.

## Design

Use the dependency-updater App installation as the consumer allowlist. The discovery job lists its repositories and excludes Marin itself. A checkout opts in through the canonical exact pin in `infra/pre-commit.py`; the updater fails when that pin is absent or ambiguous. This avoids a second catalog beside `config/external` and also covers consumers such as Axolotl that are not Marin runtime dependencies. Adding a repository requires App installation selection, the canonical pin, and protection configuration; no consumer-local updater workflow is added.

Add a manually dispatchable workflow in Marin. A discovery job resolves an optional revision, verifies that it is reachable from the `marin-community/marin-style` default branch, and emits the App installation repositories as a matrix. Each update job requests a read-only App token limited to one consumer for checkout, then mints a fresh publication token after setup with contents, workflows, and pull-request write permissions. The workflow resets the fixed `automation/marin-style` branch from the consumer's default branch. It discovers exact old-revision references in the pre-commit shim, workflow YAML, and root project file, runs the target revision's `marin-style sync`, and regenerates a root lockfile when it contains `marin-style`. Any old-revision reference outside those recognized pins, generated paths, and lockfile stops the update. Scheduled runs are added by a separate activation PR after the bootstrap and protection preflight pass.

Extend `marin-style sync` with a checked-in manifest under `.agents/marin-style/`. The manifest records the package revision and every generated destination with its rendered content hash. On the next sync, files present only in the old manifest are deleted when their content still matches the recorded hash. Sync validates the complete old manifest and all stale hashes before writing or deleting anything; a modified stale file stops the update instead of deleting consumer work. The manifest itself and the exact union of manifests produced by the installed old and new package revisions form the generated-file allowlist.

The first release containing the manifest is a bootstrap release and may not remove existing managed paths. Every consumer takes that exact revision through a normal reviewed PR. Automation is enabled only after the checked-in manifests match the old package revision in all consumers. Subsequent update jobs install both the old and target revisions to reconstruct the trusted old/new path union; they do not trust a repository-edited manifest to broaden the allowlist.

The cross-repository policy combines the discovered direct pins with the exact manifest-owned files and optional root lockfile. Worktree discovery includes tracked changes, deletions, and untracked generated files. The updater rejects an empty diff and any path outside that set before staging.

The published PR has a fixed title and branch and carries `agent-generated` and `dependencies`. Before every poll and before merge, the existing lifecycle verifies the App author, base and head branches, title, expected head commit, and changed paths. It reads the checks GitHub marks as required for the base branch; no required rows, pending rows, and failing rows all block merge. The merge uses squash, admin mode to exercise the review bypass, and `--match-head-commit`. A failing generator or check leaves the PR open and fails the workflow.

Reuse the existing `marin-external-runtime-updater` App and its protected environment for the first version. Its name is narrower than its role, but creating another credential would add operational surface without reducing permissions. Workflows write is required because direct pins include `.github/workflows/` files; the changed-file allowlist prevents unrelated workflow edits. Consumer protection is an explicit owner-run activation prerequisite: expand the App installation selection, add review-only bypass to both active protection layers where present, and confirm the App has no required-CI bypass. The workflow preflight checks labels and the default branch; the protection audit remains owner-recorded because the App intentionally lacks repository administration permission. The workflow receives no broad organization token and no CI bypass.

Each consumer has a cross-run concurrency group. A no-op target exits successfully only when there is no open automation PR; an older open PR fails the job for human inspection. Onboarding artifacts such as `AGENTS.md` references and `.claude/skills` are prerequisites and remain outside the updater allowlist. Publication and merge share the fresh token minted after setup; the merge wait is capped at 40 minutes and the whole job at 55 minutes so the one-hour installation token remains valid through the final head-bound merge.

## Testing

Unit tests cover installation discovery, exact revision replacement, generated manifest creation, stale-file pruning, refusal to delete a modified stale file, untracked-file discovery, protected-check evaluation, and allowlist rejection. Existing dependency-updater tests continue to exercise branch reset, force-with-lease, PR identity, and head-bound merge behavior.

Temporary Git repository tests generate bootstrap updates and verify that only discovered pins and manifest-owned files change. The existing dependency-updater test publishes an allowlisted update to a local remote. Before activation, run the generator against all installed consumers; the MarinSkyRL run must resolve the target source revision and package version in `uv.lock`.

Rollout is: merge a manifest-only bootstrap release; update all consumers to that revision through reviewed PRs; expand the App installation; configure and inspect both protection layers; run one real bump with merge mode `publish`; then merge an activation change that adds the six-hour schedule with merge mode `merge`. The activation preflight must pass for every consumer before that final change.

## Open Questions

- After activation, is the six-hour scheduled propagation window sufficient, or should a later change add an authenticated `repository_dispatch` from `marin-style`?
- Should the existing GitHub App be renamed from `marin-external-runtime-updater` to `marin-dependency-updater` after rollout, accepting a one-time credential and IAC migration for clearer ownership?
