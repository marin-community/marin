# Loom deployment: remote image context does not rebuild

Determine why `pulumi up --cwd infra/loom` did not plan a rebuild of
`loom-release-image` after the default branch of the remote Loom repository
changed.

## Initial status

The preview planned only a GCE instance metadata update and replacement of
`loom-activate`. It listed `loom-release-image` among the unchanged resources.
The stack uses `pulumi-docker-build` 0.0.15 with
`context.location = https://github.com/marin-community/loom.git`,
`buildOnPreview = true`, and `push = true`.

## Hypothesis 1

`buildOnPreview` controls whether an already-planned image create or update
performs a build. It does not force the image resource to have a diff.

The provider's `Diff` implementation calls `hashBuildContext`. That function
hashes local Dockerfiles and local context directories but does not resolve or
hash remote contexts. The unpinned Git URL therefore produces the same empty
context hash as its default branch advances, so Pulumi does not schedule an
image update.

## Changes

The default remote source now uses the GitHub provider to resolve `main` to a
full commit SHA. The Docker build context is
`https://github.com/marin-community/loom.git#<sha>`, so a changed branch tip
changes the image resource input. A configured local `buildContext` still
bypasses GitHub resolution.

The README now describes the SHA resolution and rebuild condition.

## Results

The provider source for version 0.0.15 confirms that remote context contents
are omitted from `contextHash`. The pinned context URL changes when `main`
advances, so the provider's direct `context.location` comparison schedules an
image update.

Pulumi deployed Loom commit
`c10430bd11b83dfb9c0b7dfb862fdfbcd7b8f7b9` as image digest
`sha256:7b1e9957cce4e5099cce01b8f4d485c4a37106e8dbc17d74921df7f2a240865a`.
The activation command passed, `/api/ready` reported both migration streams at
their expected versions with no degraded checks, and the final preview reported
24 unchanged resources.

`./infra/pre-commit.py --changed-files --fix` and focused Pyrefly checking
passed. The five synchronous tests in `infra/loom/tests/test_infrastructure.py`
passed before the first existing Pulumi async test timed out after 60 seconds.
No regression test was added.

## Future work

- [x] Resolve and pin the remote Loom commit in the image resource inputs.
- [x] Correct the README description of remote image builds.
