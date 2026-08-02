---
name: review-grant
description: Review a marin-iac user-grant PR by decrypting the changed principals into the real emails and grants, confirm with the user, then approve, merge, and drive pulumi up. Use when reviewing a PR that edits iam_data.yaml or a service's IAP viewers.
---

# Skill: Review a user grant

A grant PR (usually from the `add-grant` skill) uses opaque `human-NNN`
references whose emails are KMS ciphertext in `iam_data.yaml`. This skill
reveals the real grant, gets an explicit human confirmation, then lands it and
applies it.

Read first:

- `infra/pulumi/README.md` — the marin-iac stacks and the `pulumi up`
  prerequisites (you need `roles/cloudkms.cryptoKeyEncrypterDecrypter` on the key
  and, for a `viewers` change, access to that service's stack).

**Never approve or merge before the user confirms the decrypted grant.** The
whole point is that a second person sees the actual identity and access before it
is applied.

## 1. Fetch the PR

```bash
gh pr view <n> --repo marin-community/marin --json title,body,headRefName,files,url
gh pr checkout <n>          # pull the branch into the worktree
git fetch origin main
```

Confirm the diff only touches grant surfaces — `infra/pulumi/src/iac/gcp/iam_data.yaml`
and/or a `infra/<service>/Pulumi.marin-<service>.yaml` `viewers` list. If it
changes anything else (code, other Pulumi resources), stop and review it as an
ordinary PR, not a grant.

## 2. Decrypt the changed principals

For the `iam_data.yaml` changes, turn the opaque principal references into real
emails:

```bash
git diff origin/main...HEAD -- infra/pulumi/src/iac/gcp/iam_data.yaml \
  | uv run --package marin-iac --extra deploy \
      python infra/pulumi/iam_principal.py decrypt --diff
```

Each output line is `+ user:<email>` (added) or `- user:<email>` (removed). Map
each back to the role and resource it sits under in the diff. The decryptor
shows the principal; read the surrounding `role` and container
(`project_grants`, a specific bucket/secret/repository/service account) from the
diff hunk. For a `viewers` change, the emails are already plaintext in the YAML
diff; read them directly.

## 3. Present the grant and get confirmation

Print a plain-language summary, one line per grant, and ask the user to confirm.
For example:

```
PR #1234 grants:
  + alice@openathena.ai → roles/storage.objectViewer on project hai-gcp-models
  + alice@openathena.ai → IAP viewer on evaldash.oa.dev
  - bob@openathena.ai → roles/bigquery.dataViewer (revoked)
Apply this? (yes/no)
```

Call out anything that looks off: a broader role than the resource needs, a
principal you do not recognize, a `viewers` wildcard, or a revocation that might
cut off active access. If the user does not clearly approve, stop and report
back — do not merge.

## 4. Approve, merge, land

After the user confirms:

```bash
gh pr review <n> --repo marin-community/marin --approve
gh pr merge <n> --repo marin-community/marin --squash
git checkout main && git pull origin main   # land the merged change locally
```

Wait for the merge to land on `main` and pull it before applying, so `pulumi up`
runs against the committed state.

## 5. Apply with pulumi up

The change is not live until `pulumi up` runs — CI never applies. Identify the
affected stack(s) from the diff:

- `iam_data.yaml` → the **`marin`** stack in `infra/pulumi`.
- a `viewers` change → that service's stack, e.g. **`marin-evaldash`** in
  `infra/evaldash` or **`marin-grafana`** in `infra/grafana`.

Prompt the user: they can run it themselves, or ask you to. If you run it, per
stack:

```bash
cd infra/<dir>
pulumi stack select <stack>
pulumi preview      # confirm ONLY the intended grant is added/removed, no other drift
pulumi up
```

Read the preview before applying. For the `marin` stack, expect only the
`IAMMember` create/delete for this grant — **any NodePool or other unexpected
`replace`/`delete` means stop** and reconcile, exactly as the pulumi README
warns. Once `up` is clean, tell the user the grant is live.

## 6. Confirm on the PR

Comment on the merged PR that `pulumi up` ran and the grant is live, so the
requester and any watcher see the change reached production, not just `main`:

```bash
gh pr comment <n> --repo marin-community/marin \
  --body "🤖 \`pulumi up\` on the \`<stack>\` stack succeeded — the grant is live."
```

An agent comment must begin with `🤖` (see AGENTS.md). If `pulumi up` did not run
(the user is applying it themselves), skip this and let them confirm instead.
