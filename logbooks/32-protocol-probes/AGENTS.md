# Research archive guidelines

This directory holds the notebook and supporting work for the task issue linked
in `notebook.md`. Marin's root guidelines govern source changes elsewhere in
this branch. The logbooks default-branch guidelines are not inherited by this
Marin-derived branch; these instructions carry its archive rules with the work.

- Keep notebook entries in chronological order. Include rejected approaches,
  failed runs, and restarts when they change the interpretation or next step.
  Record material corrections as new entries.
- Before executing an experiment, commit and push its source and configuration.
  Record that exact revision, command, material environment details, and
  external inputs. Commit the resulting notebook entry and supporting evidence
  before posting an issue milestone; link to the published evidence commit.
- Preserve published history. Fetch the task branch and merge remote changes
  before pushing. Resolve concurrent edits without dropping either writer's
  entries. Never rebase published commits, force-push, or use `--mirror`.
- Use explicit remote refspecs as described in
  [the branch procedure](https://github.com/marin-community/marin-logbooks/blob/main/WORKFLOW.md).
- Keep inherited `.github/workflows` files removed before initial publication
  and after upstream merges. CI setup is a separate reviewed change: GitHub
  requires a manual workflow on the repository's default branch before it can
  be dispatched for a research branch. Preserve only approved manual workflows
  during upstream merges. Do not change repository settings or promote
  archive-only workflow removals into Marin.
- Keep large artifacts in their existing stores. Record exact references,
  checksums when available, and access limitations. Publish no secrets,
  credentials, personal data, private results, or restricted links.
- Extract selected changes into clean Marin contribution branches based on
  current Marin `main`. Preserve this research branch and its history; do not
  merge it into logbooks `main`.
- Keep the task issue current with conclusions, decisions, open questions, and
  commit-pinned evidence. Add milestone comments when findings change the plan.
  Agent comments begin with `🤖` unless their exact text was approved. Agent
  issues and PRs need `agent-generated`. Do not merge PRs without authorization.
