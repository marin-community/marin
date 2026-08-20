---
name: review-grant
description: Review a marin-iac user-grant PR by decrypting the changed principals into the real emails and grants, confirm with the user, then approve, merge, and drive pulumi up. Use when reviewing a PR that edits iam_data.yaml or a service's IAP viewers.
---

# Review a user grant

Read `infra/pulumi/README.md`. Never approve or merge before the user confirms
the decrypted identity and access.

```bash
gh pr view <n> --repo marin-community/marin --json title,body,headRefName,files,url
gh pr checkout <n>
git fetch origin main
```

The diff may contain only `infra/pulumi/src/iac/gcp/iam_data.yaml` and/or a
service `Pulumi.marin-<service>.yaml` `viewers` list. Any other file is an
ordinary PR: stop this workflow. Decrypt IAM changes and map each output to its
surrounding role/resource:

```bash
git diff origin/main...HEAD -- infra/pulumi/src/iac/gcp/iam_data.yaml \
  | uv run --package marin-iac --extra deploy \
      python infra/pulumi/iam_principal.py decrypt --diff
```

IAP emails are plaintext in the diff. Present one line per grant, including
additions/removals and suspicious breadth/wildcards, and ask for clear yes/no.
If confirmation is absent, stop.

After confirmation, merge and pull `main` before applying:

```bash
gh pr review <n> --repo marin-community/marin --approve
gh pr merge <n> --repo marin-community/marin --squash
git checkout main && git pull origin main
```

`iam_data.yaml` maps to stack `marin`; a service viewer change maps to that
service's stack. Ask whether the user will apply. If applying, inspect preview
and stop on unexpected drift, especially NodePool replacement/deletion:

```bash
cd infra/<dir>
pulumi stack select <stack>
pulumi preview
pulumi up
```

Only the intended grant should change. Report success as live and add a
`🤖`-prefixed comment to the merged PR; skip it when the user applies.
