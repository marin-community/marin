# Marin-style auto-update contract

## Files

| Path | Contract |
| --- | --- |
| `scripts/ci/marin_style_consumers.py` | Typed registry of consumer repositories, pin files, required checks, and lock mode |
| `scripts/ci/marin_style_update.py` | Validate and generate one consumer update |
| `scripts/ci/dependency_update_policy.py` | Carry required checks in each pull-request policy |
| `scripts/ci/dependency_update.py` | Reuse the generated-PR lifecycle with a resolved consumer policy and untracked-file discovery |
| `.github/workflows/ops-marin-style-consumers.yaml` | Scheduled/manual matrix orchestration from Marin's protected updater environment |
| `src/marin_style/vendor.py` in `marin-style` | Generated manifest and safe pruning behavior |

## Consumer registry

```python
from dataclasses import dataclass
from enum import StrEnum


class LockMode(StrEnum):
    NONE = "none"
    UV = "uv"


@dataclass(frozen=True)
class MarinStyleConsumer:
    name: str
    repository: str
    base_branch: str
    revision_file: str
    pin_files: tuple[str, ...]
    required_checks: tuple[str, ...]
    lock_mode: LockMode


def marin_style_consumer(name: str) -> MarinStyleConsumer:
    """Return one registered consumer or raise ValueError for an unknown name."""


def marin_style_consumer_matrix() -> str:
    """Return deterministic compact JSON for a GitHub Actions matrix."""
```

Names, repository paths, pin files, and check names are non-empty and unique. Repository paths must belong to `marin-community`. Pin files are exact repository-relative POSIX paths that receive direct revision replacement; directory, glob, and lockfile entries are invalid. `LockMode.UV` adds `uv.lock` as a generated output. `base_branch` is explicit even though all initial consumers use `main`.

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
    consumer: MarinStyleConsumer,
    revision: str,
) -> GeneratedMarinStyleUpdate:
    """Update one checkout to a full, default-branch marin-style revision.

    The old revision is read from `infra/pre-commit.py`. Every registered direct
    pin file must contain that revision. Replacement occurs only in those files.
    Sync and optional lock generation then run from the target revision. The
    function rejects an empty diff or any changed path outside the direct pins,
    generated lock outputs, and exact trusted old/new generated manifests.
    """
```

`revision` is exactly 40 lowercase hexadecimal characters. The workflow verifies it is reachable from the `marin-style` default branch; an omitted revision resolves the current head. Each direct pin file must exist and contain the same old revision at least once. The generator replaces every old-revision occurrence and fails rather than partially updating a consumer.

The target package is invoked as `uvx --from git+https://github.com/marin-community/marin-style@<revision> marin-style sync --repo-root <checkout>`. Its reported installed revision must equal `revision`. `LockMode.UV` uses the workflow's pinned `uv` version to run `uv lock --upgrade-package marin-style`, verifies the resulting Git source revision and package version, and permits dependency re-resolution inside `uv.lock`; no other file becomes allowed. `AGENTS.md` references and `.claude/skills` setup must already be valid before onboarding.

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

Repositories without an old manifest are supported for the reviewed bootstrap update. Sync discovers no deletions in that case, writes the initial manifest, and otherwise preserves current behavior. Activated automation refuses a consumer without a manifest.

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
    required_checks: tuple[str, ...]
```

Existing Marin dependency policies populate `required_checks` with Marin's current required-check tuple. A consumer policy uses branch `automation/marin-style`, title `[dependencies] Advance marin-style`, and its registry check tuple. Merge validation reads `policy.required_checks`; no global fallback exists.

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
    consumer: MarinStyleConsumer,
    revision: str,
    merge_mode: MergeMode,
    app_slug: str,
    manifest_mode: ManifestMode,
) -> ConsumerUpdateResult:
    """Prepare, publish, and optionally merge one consumer update.

    CURRENT has an empty pull-request URL; PUBLISHED and MERGED carry the exact
    URL. The function constructs the dynamic pull-request policy in memory
    after comparing the old checked-in manifest with output from the old pinned
    package and the target installed package. The same policy object is passed
    to publication and merge validation, so old-only paths are not lost between
    CLI steps.
    """
```

The command-line entry point is:

```text
python -m scripts.ci.marin_style_update run \
  --consumer <name> --revision <sha> --app-slug <slug> [--auto-merge]
```

The workflow invokes the module once per consumer inside an environment containing the target `marin-style` revision. The command calls the reusable `prepare_update_branch`, `publish_update`, and `merge_when_green` functions directly. It obtains the old manifest with `uvx --from git+https://github.com/marin-community/marin-style@<old> marin-style managed-files`; that JSON must exactly equal the checked-in old manifest. `marin-style managed-files` emits the same deterministic JSON shape as `manifest.json` without mutating a checkout.

## Workflow

The bootstrap workflow is manual-only and accepts inputs:

- `revision`: optional full `marin-style` commit; empty resolves the default-branch head.
- `auto_merge`: boolean, default `false`.
- `consumer`: optional registered consumer name; empty selects all consumers.

Each matrix job uses environment `external-runtime-updater`, requests an installation token for exactly one consumer, and grants the workflow's native token only `contents: read` in Marin. The App token performs consumer checkout, branch publication, PR upsert, and merge. Jobs do not share installation tokens. Each job's concurrency group is `marin-style-update-<consumer>` with `cancel-in-progress: false`.

If the consumer already pins `revision`, the job exits successfully only when the fixed automation branch has no open PR. An older open PR fails for inspection. A generator error, unexpected path, absent/failing check, altered PR identity, timeout, or merge failure fails that consumer's job and leaves any PR open.

After the bootstrap, installation, and protection preflight pass, a separate reviewed activation change adds schedule `17 */6 * * *` and makes scheduled runs merge automatically. Manual runs retain the explicit `auto_merge` input.

## Protection policy

The updater App is installed only on Marin and the registered consumer repositories. It receives contents and pull-request write permissions. Each consumer review ruleset gives the App pull-request-only bypass. Classic required-review protection, where present, also names the App in its PR bypass allowances. Required-status-check policy does not name the App as a bypass actor.

App installation selection and consumer protection remain owner-managed prerequisites in the first version. Before publishing, the workflow preflight confirms installation access, the existence of `agent-generated` and `dependencies` labels, and the registered base branch. The App lacks administration permission and cannot audit protection itself. Auto-merge activation requires an owner-recorded audit showing both protection layers and no App CI bypass for every consumer.

## Out of scope

- Mutable `marin-style@main` dependencies in consumers.
- Updating consumer-owned `AGENTS.md`, `.claude/skills`, `.agents/ops`, `.agents/projects`, or custom skills.
- Enabling, disabling, or editing upstream workflows in fork repositories.
- Automatically changing the registry when consumer CI check names change.
- Importing consumer branch-protection resources into Pulumi before their complete fork-specific policies are audited.
- Renaming the existing updater GitHub App in the first rollout.
