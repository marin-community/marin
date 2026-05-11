---
name: github-action-upgrade
description: Used to upgrade a step in a github action
---

# Skill: Upgrade GitHub Action

You are to upgrade GitHub Action dependencies to use immutable SHAs instead of mutable tags, ensuring security and reproducibility across the project.

## Phase 1: Discovery
1. **Identify Outdated Actions:** Scan `.github/workflows/` and `.github/actions/` for `uses:` lines that use tags (e.g., `@v3`) or SHAs without identifying version comments.
2. **Deduplicate:** If upgrading multiple files, compile a list of unique actions first to avoid redundant work.

## Phase 2: Upgrading a Single Step
To find the upgraded dependency for a single step, do the following:

1. Go to `https://github.com/<action name>/releases` (e.g. `https://github.com/actions/checkout/releases`)
2. Find the most recent release (it will be tagged with "Latest").
3. Go to `https://github.com/<action name>/releases/tag/<release tag>` (e.g. `https://github.com/actions/checkout/releases/tag/v6.0.2`)
4. Find the sha1 of the commit for that release tag. It will look like a link of the form (`https://github.com/actions/checkout/commit/<sha1>`) with the link text being the first 7 characters of the sha1.
5. Remember, you need to get the full sha1 sum from the link. Do not read the first 7 characters and hallucinate the remaining characters.
6. Double check the sha1 by going to `https://github.com/actions/checkout/commit/<sha1>` and making sure it's a valid sha1. You should see the `<release tag>` listed on that page.
7. Update the action by changing the line in the GitHub Action step to look like:
   `uses: <action name>@<sha1> # <release tag>`

## Phase 3: Upgrading Multiple Actions
If you're asked to upgrade multiple actions:
1. Follow Phase 1 to identify all unique actions needing updates.
2. For each unique action, use a sub-agent to follow the "Upgrading a Single Step" process in parallel.
3. Apply the results across all relevant files in the repository using targeted edits.
