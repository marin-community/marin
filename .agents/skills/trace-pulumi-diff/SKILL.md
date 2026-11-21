---
name: trace-pulumi-diff
description: Run a read-only preview for a specified Marin infra/pulumi stack and trace each pending resource change to merged pull requests since its latest successful update when that update records a clean Git checkout. Use when a stack has unapplied changes or drift and someone needs to know which PRs explain the preview.
---

# Trace a Pulumi diff

Work only with the `marin-iac` project in `infra/pulumi`. Require an explicit stack name.

## Preview current main

Read `infra/pulumi/README.md` and satisfy its credentials and stack-specific prerequisites. Fetch `origin/main` and run from a clean checkout at that commit. Preserve existing work; use another worktree when necessary.

Prepare the environment from the repository root:

```bash
uv sync --package marin-iac --extra deploy --frozen
export PULUMI_PYTHON_CMD="$PWD/.venv/bin/python"
```

Run the preview without changing the selected stack:

```bash
pulumi -C infra/pulumi preview --stack <stack> --diff --color never
```

Keep the raw preview local. It may contain decrypted identifiers. If the preview is empty, report that the stack has no pending diff and stop.

## Find the deployed commit

Read recent history without secrets:

```bash
pulumi -C infra/pulumi stack history \
  --stack <stack> --json --page-size 100 > /tmp/marin-pulumi-history-<stack>.json
jq -r '
  map(select(.kind == "update" and .result == "succeeded"))
  | first
  | [.startTime, .environment["git.head"], .environment["git.dirty"]]
  | @tsv
' /tmp/marin-pulumi-history-<stack>.json
```

Use `git.head` only when it is a commit on `origin/main` and `git.dirty` is `false`. If either condition fails, report that an exact Git baseline is unavailable and stop attribution.

## Attribute the preview

List merged commits after the deployed commit:

```bash
git log --first-parent --format='%H%x09%s' <deployed-sha>..origin/main
```

Inspect each candidate with `git show`. Match concrete preview resource names and changed properties to the code or configuration diff. Do not attribute a change merely because its commit is in the range.

Map a matching commit to its pull request:

```bash
gh api repos/marin-community/marin/commits/<commit>/pulls \
  --jq '.[] | [.number, .title, .html_url] | @tsv'
```

Report each preview change with its matching PR and the file or hunk that explains it. Report changes with no matching commit as live drift or unresolved attribution. A preview can contain changes from multiple PRs.

Never run `pulumi up`, `pulumi refresh`, `pulumi destroy`, an import, or a state mutation. Call out any NodePool replacement or deletion from the preview.
