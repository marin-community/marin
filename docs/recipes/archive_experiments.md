# Recipe: Archive Legacy Experiments

## Overview
Use this recipe when old experiment scripts need to be retired without losing their history. The goal is to put the code behind a dated archive tag and leave a canonical comment on the originating GitHub issues so future readers know where to find the last snapshot.

## Prerequisites
- Local checkout of `marin` with push access to `origin`.
- `gh` CLI installed and authenticated for `marin-community/marin`.
- List of experiment issues and their corresponding filenames under `experiments/`.
- The PR number for the cleanup you are referencing; if the directing human has not given one, ask them before posting anything.

## Guidelines for Humans

### 1. Snapshot the experiments behind an archive tag
1. Identify the commit that should house the archived experiments (usually `main` after the cleanup). Record it with `git rev-parse HEAD`.
2. Pick a date-based tag name in the form `archive/YYYYMMDD`.
3. Create and push the tag:
   ```bash
   TAG=archive/20251114
   git tag "${TAG}" <commit-sha>
   git push origin "${TAG}"
   ```
4. If an older tag (e.g., `exp_cleaning/YYYYMMDD`) described the same snapshot, delete it locally and remotely so only the `archive/` tag remains.

### 2. Post the canonical archive comment on each issue
1. Confirm the cleanup PR number (`PR_NUM`); if nobody has provided one yet, ping the human operator for it before continuing.
   <!-- Example: For a previous batch, the PR number was `1999`. -->
2. For every experiment issue, build the file URL: `https://github.com/marin-community/marin/tree/${TAG}/experiments/<filename>`.
3. Use the standardized note below, swapping in the filename-specific URL and PR reference:
   ```
   This experiment has been archived to reduce clutter and preserve velocity (see PR #<PR_NUM>). It is last available in the `archive/YYYYMMDD` tag at <URL>. Please open an issue if you need help unarchiving it.
   ```
4. Post the comment with `gh issue comment`:
   ```bash
   ISSUE=102
   FILE=exp102_classifier_ablations.py
   PR_NUM=1999
   URL="https://github.com/marin-community/marin/tree/${TAG}/experiments/${FILE}"
   gh issue comment "${ISSUE}" \
     --body "This experiment has been archived to reduce clutter and preserve velocity (see PR #${PR_NUM}). It is last available in the \`${TAG}\` tag at ${URL}. Please open an issue if you need help unarchiving it."
   ```
5. Confirm the comment renders correctly (PR reference should auto-link and the URL should jump straight to the file in the archive tag).

### 3. Track progress
- Maintain a checklist of issue numbers while you work (even a scratch buffer is fine) to avoid double-posting or skipping an experiment.
- When the batch is complete, drop a short note in your agent handoff or PR description listing the affected issues.

## Rules for Agents
- Do not delete experiment files; archival work only tags commits and leaves issue breadcrumbs.
- Always reference the current `archive/` tag name in both inline code and URLs.
- If the archive tag changes after comments were posted, update the affected issues rather than stacking multiple archive notices.

## Validation
- `git ls-remote --tags origin | rg "${TAG}$"` to ensure the tag is published.
- `gh issue view <issue> --comments | rg "archived to reduce clutter"` to confirm each issue carries the canonical message.
- Spot-check a few URLs to verify they open the expected files inside the archive tag.

## See Also
- `docs/recipes/organize_experiments.md` for related curation workflows.
