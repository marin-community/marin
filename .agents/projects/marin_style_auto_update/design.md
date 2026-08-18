# Automate marin-style consumer updates

Keep `marin-style` exactly pinned in each Marin-community repository while removing the repeated human work needed to prepare, approve, and merge mechanical bumps. A central workflow will update every registered consumer after a source change, wait for that repository's required CI, and merge only a generator-owned diff. A failed or surprising update remains open for a person to inspect.

The research and repository inventory are in [research.md](research.md). Marin already has most of the security-sensitive machinery: a dedicated GitHub App, a main-only credential environment, fixed automation branches, PR identity validation, file allowlists, required-check gates, and head-SHA-bound merges. This proposal extends that mechanism across repositories instead of adding updater credentials and workflows to every fork.

## Challenges

The consumers do not pin `marin-style` in the same places. Most have a shim and two workflow references; MarinSkyRL also has a project dependency and lockfile. Their CI check names differ, and upstream-tracking forks contain workflows and agent assets that Marin must preserve byte-for-byte.

Generated assets need a precise ownership boundary. `marin-style sync` currently writes the package's current assets but has no record of past outputs, so removing a packaged skill would leave an obsolete file behind. Allowlisting `.agents/**` would be unsafe because consumers maintain adjacent operations, project, and custom-skill files.

Auto-merge also crosses two independent GitHub controls. The updater App needs a review bypass in each consumer's ruleset and classic branch protection, where present, but must still satisfy all required CI. App installation repository selection is an owner-managed operation.

## Costs / Risks

- A central registry becomes the source of truth for consumer pin locations and required check names. Repository CI changes must update it.
- The existing dependency-updater App gains access to six more repositories, limited to contents and pull requests.
- A `marin-style` release can produce several PRs at once. Failures remain open and require diagnosis.
- Activation requires one final manual bootstrap update in each consumer so every repository has a trusted generated manifest.

## Design

Add a typed consumer registry in Marin. Each record names the repository and default branch, explicit non-generated pin files, required GitHub checks, and an optional lock command. The initial registry contains Harbor, TPU inference, vLLM, Evalchemy, Axolotl, and MarinSkyRL. Adding a repository requires a registry entry, App installation selection, and protection configuration; no consumer-local updater workflow is added.

Add a manually dispatchable workflow in Marin. A discovery job resolves an optional revision, verifies that it is reachable from the `marin-community/marin-style` default branch, and emits the registry as a matrix. Each update job requests a short-lived App token limited to one consumer, checks out that consumer, and resets the fixed `automation/marin-style` branch from its default branch. The workflow replaces the old revision only in the registered direct-pin files, runs the target revision's `marin-style sync`, and updates any registered generated lockfile with a pinned `uv` release. Scheduled runs are added by a separate activation PR after the bootstrap and protection preflight pass.

Extend `marin-style sync` with a checked-in manifest under `.agents/marin-style/`. The manifest records the package revision and every generated destination with its rendered content hash. On the next sync, files present only in the old manifest are deleted when their content still matches the recorded hash. Sync validates the complete old manifest and all stale hashes before writing or deleting anything; a modified stale file stops the update instead of deleting consumer work. The manifest itself and the exact union of manifests produced by the installed old and new package revisions form the generated-file allowlist.

The first release containing the manifest is a bootstrap release and may not remove existing managed paths. Every consumer takes that exact revision through a normal reviewed PR. Automation is enabled only after the checked-in manifests match the old package revision in all consumers. Subsequent update jobs install both the old and target revisions to reconstruct the trusted old/new path union; they do not trust a repository-edited manifest to broaden the allowlist.

Generalize Marin's existing dependency update policy so required checks belong to `PullRequestPolicy`. The cross-repository policy combines the consumer's explicit pin files with the exact manifest-owned files. Worktree discovery includes tracked changes, deletions, and untracked generated files. The updater rejects an empty diff and any path outside that set before staging.

The published PR has a fixed title and branch and carries `agent-generated` and `dependencies`. Before every poll and before merge, the existing lifecycle verifies the App author, base and head branches, title, expected head commit, and changed paths. It waits for the consumer registry's fixed check set; missing, pending, or failing checks cannot merge. The merge uses squash, admin mode only to exercise the review bypass, and `--match-head-commit`. A failing generator or check leaves the PR open and fails the workflow.

Reuse the existing `marin-external-runtime-updater` App and its protected environment for the first version. Its name is narrower than its role, but creating another credential would add operational surface without reducing permissions. Consumer protection is an explicit owner-run activation prerequisite: expand the App installation selection, add review-only bypass to both active protection layers where present, and confirm the App has no required-CI bypass. The workflow preflight checks App installation access, labels, and the default branch; the protection audit remains owner-recorded because the App intentionally lacks repository administration permission. The workflow receives no broad organization token and no CI bypass.

Each consumer has a cross-run concurrency group. A no-op target exits successfully only when there is no open automation PR; an older open PR fails the job for human inspection. Direct pin files must each contain the same old revision, and every occurrence of that revision in those files is replaced. Onboarding artifacts such as `AGENTS.md` references and `.claude/skills` are prerequisites and remain outside the updater allowlist.

## Testing

Unit tests cover registry validation, exact revision replacement, generated manifest creation, stale-file pruning, refusal to delete a modified stale file, untracked-file discovery, per-consumer check selection, and allowlist rejection. Existing dependency-updater tests continue to exercise branch reset, force-with-lease, PR identity, and head-bound merge behavior.

A local integration test creates a bare consumer repository, generates an update that adds and removes packaged assets, publishes it to a local remote, and verifies that only registered pins and manifest-owned files change. A hermetic MarinSkyRL fixture verifies that `uv.lock` is regenerated with a pinned `uv` release and records the target source revision and package version.

Rollout is: merge a manifest-only bootstrap release; update all consumers to that revision through reviewed PRs; expand the App installation; configure and inspect both protection layers; run one manual, auto-merge-disabled real bump; then merge an activation change that adds the six-hour schedule and enables auto-merge. The activation preflight must pass for every consumer before that final change.

## Open Questions

- After activation, is the six-hour scheduled propagation window sufficient, or should a later change add an authenticated `repository_dispatch` from `marin-style`?
- Should the existing GitHub App be renamed from `marin-external-runtime-updater` to `marin-dependency-updater` after rollout, accepting a one-time credential and IAC migration for clearer ownership?
