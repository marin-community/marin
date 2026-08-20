---
name: archive-experiments
description: Retire legacy experiment scripts behind a dated archive tag.
---

# Archive Legacy Experiments

Preserve experiment history in a dated tag and leave one canonical breadcrumb
on each originating issue. This skill never deletes experiment files.

Before posting anything, obtain the cleanup PR number if it was not supplied.
In local/manual mode, confirm authorization before pushing tags or comments.

## Snapshot

Record the snapshot commit (usually the cleaned `main`), then create and push
an `archive/YYYYMMDD` tag:

```sh
TAG=archive/YYYYMMDD
git tag "${TAG}" <commit-sha>
git push origin "${TAG}"
```

If an older tag names the same snapshot, and the operator authorized cleanup,
remove that superseded tag locally and remotely. Do not delete experiment files.

## Issue breadcrumbs

For every issue/file pair, post exactly one canonical comment, updating an old
archive notice if the tag changes:

```sh
URL="https://github.com/marin-community/marin/tree/${TAG}/experiments/${FILE}"
gh issue comment "${ISSUE}" --body \
  "This experiment has been archived to reduce clutter and preserve velocity (see PR #${PR_NUM}). It is last available in the \`${TAG}\` tag at ${URL}. Please open an issue if you need help unarchiving it."
```

Track issue numbers to avoid skips or duplicate comments. Keep the current tag
in both inline code and the URL.

## Validate

```sh
git ls-remote --tags origin | rg "${TAG}$"
gh issue view <issue> --comments | rg "archived"
```

Spot-check archive URLs and record affected issues in the handoff. See
`organize-experiments` for report-index curation.
