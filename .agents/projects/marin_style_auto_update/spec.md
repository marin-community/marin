# Marin-style auto-update contract

## Files

| Path | Contract |
| --- | --- |
| `scripts/ci/marin_style_update.py` | Discover installed consumers, validate pins, and generate one update |
| `scripts/ci/dependency_update.py` | Reuse the generated-PR lifecycle with untracked-file discovery and protected-check evaluation |
| `.github/workflows/ops-marin-style-consumers.yaml` | Scheduled/manual matrix orchestration from Marin's protected updater environment |
| `src/marin_style/vendor.py` in `marin-style` | Generated manifest and safe pruning behavior |

## Consumer discovery

`installed_consumer_matrix_json()` lists repositories visible to the dependency-updater App installation and excludes `marin-community/marin`. Each row carries the repository name and default branch returned by GitHub. An optional selector accepts the short or `owner/name` repository name and must match exactly one installed repository.

The App installation is the only central repository list. A checkout opts in by containing exactly one canonical `marin-style` URL and full revision in `infra/pre-commit.py`. The update job fails if an installed repository lacks that pin. Adding a consumer requires App installation selection, the canonical pin, labels, and reviewed branch protection; it does not require a Marin registry entry.

## Generated update

```python
from pathlib import Path


@dataclass(frozen=True)
class GeneratedMarinStyleUpdate:
    old_revision: str
    new_revision: str
    changed_files: tuple[str, ...]
    allowed_files: frozenset[str]


def generate_marin_style_update(
    *,
    repo_root: Path,
    base_branch: str,
    revision: str,
    manifest_mode: ManifestMode,
) -> GeneratedMarinStyleUpdate:
    """Update one checkout to a full, default-branch marin-style revision.

    The old revision is read from `infra/pre-commit.py`. Every discovered direct
    reference is discovered from tracked files. Replacement occurs only in
    recognized pins. Sync and optional lock generation then run from the target
    revision. The function rejects an empty diff or any changed path outside the
    direct pins, generated lock output, and exact trusted old/new manifests.
    """
```

`revision` is exactly 40 lowercase hexadecimal characters. `base_branch` becomes the pull request base and must still match the repository's default branch at preflight. The workflow verifies the revision is reachable from the `marin-style` default branch; an omitted revision resolves the current head. After removing manifest-owned paths, the manifest itself, and the generated lockfile, every tracked file containing the old revision must be one of:

- `infra/pre-commit.py`, containing a `marin-style` reference;
- YAML beneath `.github/workflows/`, where each matching line names `marin-style` or `MARIN_STYLE_REV`;
- root `pyproject.toml`, where each matching line names `marin-style`.

The generator replaces every old-revision occurrence in those discovered pins. Another tracked reference fails the update.

The target package is invoked as `uvx --from git+https://github.com/marin-community/marin-style@<revision> marin-style sync --repo-root <checkout>`. Its reported installed revision must equal `revision`. When root `uv.lock` contains one `marin-style` Git package at the old revision, the workflow's pinned `uv` runs `uv lock --upgrade-package marin-style`, verifies the resulting source revision and package version, and permits dependency re-resolution only inside `uv.lock`. `AGENTS.md` references and `.claude/skills` setup must already be valid before onboarding.

## Generated manifest

`marin-style sync` writes `.agents/marin-style/manifest.json` with this versioned shape:

```json
{
  "format": 1,
  "revision": "0123456789abcdef0123456789abcdef01234567",
  "files": {
    ".agents/marin-style/AGENTS-core.md": "sha256:...",
    ".agents/skills/commit/SKILL.md": "sha256:..."
  }
}
```

Keys are repository-relative POSIX paths sorted lexicographically. Hashes cover the exact rendered bytes written by sync. Paths must be under `.agents/marin-style/` or `.agents/skills/`, may not contain `..`, and may not name `.agents/marin-style/manifest.json` itself.

```python
@dataclass(frozen=True)
class ManagedManifest:
    revision: str
    files: tuple[tuple[str, str], ...]


def managed_manifest() -> ManagedManifest:
    """Return the manifest for the installed package's rendered assets."""


def sync(repo_root: Path | None = None) -> SyncResult:
    """Write current assets and prune unchanged outputs from the old manifest.

    Sync creates or replaces the manifest. A stale old output is deleted only
    when its current digest equals the old manifest digest; a mismatch raises
    ValueError.
    """

def check_sync(repo_root: Path | None = None) -> SyncResult:
    """Report missing, drifted, or obsolete assets without modifying the checkout.

    The result distinguishes current missing/drifted assets, stale old outputs,
    and manifest drift.
    """
```

Repositories without an old manifest are supported for the reviewed bootstrap update. Sync discovers no deletions in that case, writes the initial manifest, and otherwise preserves current behavior. The bootstrap publishes a manifest-only update when the consumer already pins the target revision. Activated automation refuses a consumer without a manifest.

Before write mode mutates the checkout, it validates the manifest format, digest grammar, canonical relative paths, old manifest revision, and every existing stale path and hash. Managed paths are regular files beneath `.agents/marin-style/` or package-owned `.agents/skills/<name>/` paths; symlinks are rejected. When an updater invokes sync, the old manifest must equal the manifest produced by installing the old pinned package. The manifest bootstrap revision adds the manifest without removing existing managed paths.

`SyncResult` adds `stale: list[Path]` and `manifest_drifted: bool`. Check mode performs no writes, exits nonzero through the CLI when either field is non-empty/true, and reports obsolete generated files separately from content drift.

## Pull-request policy

```python
@dataclass(frozen=True)
class PullRequestPolicy:
    base_branch: str
    head_branch: str
    title: str
    allowed_files: frozenset[str]
```

A consumer policy uses branch `automation/marin-style`, title `[dependencies] Advance marin-style`, and the generated update's dynamic file set. Merge polling calls `gh pr checks --required` for the consumer PR. No rows, pending rows, or failing rows block merge. The App has no required-CI bypass, so repository protection remains the authority for the required set.

Changed-file discovery returns the union of tracked worktree changes and untracked non-ignored files. Publication stages only the validated result. PR validation permits only a non-empty subset of `allowed_files` and still binds author, base, head, title, and expected head SHA.

## Updater command

```python
class ManifestMode(StrEnum):
    VALIDATE = "validate"
    BOOTSTRAP = "bootstrap"


class MergeMode(StrEnum):
    PUBLISH = "publish"
    MERGE = "merge"


class ConsumerUpdateStatus(StrEnum):
    CURRENT = "current"
    PUBLISHED = "published"
    MERGED = "merged"


@dataclass(frozen=True)
class ConsumerUpdateResult:
    status: ConsumerUpdateStatus
    pull_request_url: str


def update_consumer(
    *,
    repository: str,
    base_branch: str,
    revision: str,
    merge_mode: MergeMode,
    app_slug: str,
    manifest_mode: ManifestMode,
) -> ConsumerUpdateResult:
    """Prepare, publish, and optionally merge one consumer update.

    CURRENT has an empty pull-request URL; PUBLISHED and MERGED carry the exact
    URL. BOOTSTRAP with MERGE raises ValueError because the first manifest must
    receive human review.
    """
```

`ManifestMode.BOOTSTRAP` and `LEGACY_MANAGED_FILES` are rollout scaffolding owned by Marin maintainers. Remove both after every repository selected in the App installation has a manifest on its default branch.

The command-line entry point is:

```text
python -m scripts.ci.marin_style_update run \
  --repository <owner/name> --base-branch <branch> \
  --revision <sha> --app-slug <slug> \
  --manifest-mode <validate|bootstrap> --merge-mode <publish|merge>
```

The workflow invokes the module once per consumer inside an environment containing the target `marin-style` revision. The command uses one in-memory pull-request policy through branch preparation, publication, and optional merge, and returns the structured status and pull-request URL. It obtains the old manifest with `uvx --from git+https://github.com/marin-community/marin-style@<old> marin-style managed-files`; that JSON must exactly equal the checked-in old manifest. `marin-style managed-files` emits the same deterministic JSON shape as `manifest.json` without mutating a checkout.

## Workflow

The workflow is manual-only and accepts inputs:

- `revision`: optional full `marin-style` commit; empty resolves the default-branch head.
- `consumer`: optional installed repository name; empty selects all installed consumers except Marin.
- `manifest_mode`: `validate` or `bootstrap`, default `validate`.
- `merge_mode`: `publish` or `merge`, default `publish`; bootstrap requires `publish`.

The discovery job uses a read-only installation token to list the App's selected repositories. Each matrix job uses environment `external-runtime-updater`, grants the workflow's native token only `contents: read` in Marin, and requests installation tokens for exactly one consumer. A read-only token performs checkout. After setup, the job mints a fresh token with contents, workflows, and pull-request write permissions for branch publication, PR upsert, and optional merge. The job replaces checkout's persisted Git credential with the fresh token before publication. Jobs do not share installation tokens. Each job's concurrency group is `marin-style-update-<consumer>` with `cancel-in-progress: false`.

The publication token expires after one hour. Merge polling is capped at 40 minutes and the matrix job at 55 minutes, leaving time for generation, publication, and final head-bound merge validation. A timeout leaves the pull request open and fails the job.

If the consumer already pins `revision`, the job exits successfully only when the fixed automation branch has no open PR. An older open PR fails for inspection. A generator error, unexpected path, absent/failing check, altered PR identity, timeout, or merge failure fails that consumer's job and leaves any PR open.

After the bootstrap, installation, and protection preflight pass, a separate reviewed activation change adds schedule `17 */6 * * *` and runs scheduled updates with manifest mode `validate` and merge mode `merge`. Manual runs retain both explicit mode inputs.

## Protection policy

The updater App is installed only on Marin and the intended `marin-style` consumer repositories. It receives contents, workflows, and pull-request write permissions. Workflows write is necessary because exact revision pins live under `.github/workflows/`; the updater's generated allowlist constrains which workflow files may change. Each consumer review ruleset gives the App pull-request-only bypass. Classic required-review protection, where present, also names the App in its PR bypass allowances. Required-status-check policy does not name the App as a bypass actor.

App installation selection and consumer protection remain owner-managed prerequisites in the first version. Before publishing, the workflow preflight confirms the canonical pin, the existence of `agent-generated` and `dependencies` labels, and the discovered default branch. The App lacks administration permission and cannot audit protection itself. Auto-merge activation requires an owner-recorded audit showing both protection layers and no App CI bypass for every consumer.

## Out of scope

- Mutable `marin-style@main` dependencies in consumers.
- Updating consumer-owned `AGENTS.md`, `.claude/skills`, `.agents/ops`, `.agents/projects`, or custom skills.
- Enabling, disabling, or editing upstream workflows in fork repositories.
- Updating repositories outside the App installation selection.
- Importing consumer branch-protection resources into Pulumi before their complete fork-specific policies are audited.
- Renaming the existing updater GitHub App in the first rollout.
