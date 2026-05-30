# Branch protection rollout for the workflow rename PR

The rename PR changes 11 required job ids. Branch protection on `main`
must be updated for the PR to merge — `gh` API call below.

## Current required contexts

Captured 2026-05-01:

```text
lint-and-format
build-docs
marin-tests
levanter-tests
levanter-entry-tests
levanter-torch-tests
haliax-tests
iris-tests
zephyr-tests
fray-tests
marin-itest
```

All 11 are emitted by `app_id: 15368` (GitHub Actions).

## Mapping

| Old | New |
| --- | --- |
| `lint-and-format` | `marin-lint` |
| `build-docs` | `marin-docs` |
| `marin-tests` | `marin-unit` |
| `levanter-tests` | `levanter-unit` |
| `levanter-entry-tests` | `levanter-entry` |
| `levanter-torch-tests` | `levanter-torch` |
| `haliax-tests` | `haliax-unit` |
| `iris-tests` | `iris-unit` |
| `zephyr-tests` | `zephyr-unit` |
| `fray-tests` | `fray-unit` |
| `marin-itest` | `marin-integration` |

## Rollout sequence

The rename PR cannot merge while `lint-and-format`, `marin-tests`, etc.
are required, because the renamed PR branch emits the new names instead.

### Step 1 — verify the renamed checks are green on the PR branch

```bash
gh pr checks <PR-NUMBER>
```

Confirm each of the 11 new contexts (`marin-lint`, `marin-docs`,
`marin-unit`, `levanter-unit`, `levanter-entry`, `levanter-torch`,
`haliax-unit`, `iris-unit`, `zephyr-unit`, `fray-unit`,
`marin-integration`) shows as `pass`. If any fail, fix the workflow
in the PR before continuing.

### Step 2 — swap required contexts

Single PATCH that replaces the entire required-checks list.
The previous list is preserved at the top of this file so a rollback
PATCH is trivial.

```bash
gh api \
  --method PATCH \
  -H "Accept: application/vnd.github+json" \
  /repos/marin-community/marin/branches/main/protection/required_status_checks \
  --input - <<'EOF'
{
  "strict": false,
  "checks": [
    {"context": "marin-lint", "app_id": 15368},
    {"context": "marin-docs", "app_id": 15368},
    {"context": "marin-unit", "app_id": 15368},
    {"context": "levanter-unit", "app_id": 15368},
    {"context": "levanter-entry", "app_id": 15368},
    {"context": "levanter-torch", "app_id": 15368},
    {"context": "haliax-unit", "app_id": 15368},
    {"context": "iris-unit", "app_id": 15368},
    {"context": "zephyr-unit", "app_id": 15368},
    {"context": "fray-unit", "app_id": 15368},
    {"context": "marin-integration", "app_id": 15368}
  ]
}
EOF
```

### Step 3 — merge

The PR is now mergeable.

### Step 4 — verify

```bash
gh api repos/marin-community/marin/branches/main/protection/required_status_checks --jq '.checks'
```

Should print the 11 new contexts.

## Rollback

If the renamed checks fail post-merge for some reason and the renames
need to be reverted, restore the old required contexts:

```bash
gh api \
  --method PATCH \
  -H "Accept: application/vnd.github+json" \
  /repos/marin-community/marin/branches/main/protection/required_status_checks \
  --input - <<'EOF'
{
  "strict": false,
  "checks": [
    {"context": "lint-and-format", "app_id": 15368},
    {"context": "build-docs", "app_id": 15368},
    {"context": "marin-tests", "app_id": 15368},
    {"context": "levanter-tests", "app_id": 15368},
    {"context": "levanter-entry-tests", "app_id": 15368},
    {"context": "levanter-torch-tests", "app_id": 15368},
    {"context": "haliax-tests", "app_id": 15368},
    {"context": "iris-tests", "app_id": 15368},
    {"context": "zephyr-tests", "app_id": 15368},
    {"context": "fray-tests", "app_id": 15368},
    {"context": "marin-itest", "app_id": 15368}
  ]
}
EOF
```

Then revert the rename commit on `main`.

## Active rulesets

`gh api repos/marin-community/marin/rulesets` returns the `protect main`
ruleset. As of 2026-05-01 the required status checks live in the classic
branch protection captured above, not in a ruleset, so the PATCH above
is sufficient. If a future change moves required checks into the ruleset,
the same sequence applies through the rulesets API
(`/repos/{owner}/{repo}/rulesets/{ruleset_id}`) instead of branch
protection.
