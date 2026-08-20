---
name: add-grant
description: Add a GCP IAM or IAP user grant to the marin-iac Pulumi stacks, encrypting human principals. Use when someone needs access to a project role, bucket/secret/service account, or an IAP-gated web service (evaldash, grafana), whether requested locally or on a GitHub issue.
---

# Add a user grant

Create a reviewable Pulumi change. This skill never applies the change; a second
person uses `review-grant`, merges, and runs `pulumi up`.

Read `infra/pulumi/README.md` and the header of
`infra/pulumi/src/iac/gcp/iam_data.yaml` first.

## Choose the surface

- **GCP IAM:** `infra/pulumi/src/iac/gcp/iam_data.yaml`, applied by stack
  `marin`. Project roles live in `project_grants`; resource grants live under
  `buckets`, `secrets`, `artifact_repositories`, or `service_accounts`. Human
  `user:<email>` principals are registered once and referenced as encrypted
  `human-NNN`; service accounts, groups, and domains remain plaintext.
- **IAP:** the service's `infra/<service>/Pulumi.marin-<service>.yaml`, under
  plaintext `viewers`, applied by that service's stack. Workspace domain and
  Loom access are already present; add an entry for an outside viewer.

One request may touch both surfaces when explicitly requested; ask only when
the intended surface or surfaces are ambiguous. Collect the
principal, exact resource/capability, and one-line justification (including an
expiry if explicitly requested). Choose the narrowest existing role that meets
the capability; do not guess at vague or broad requests.

For a GitHub issue, fetch with:

```bash
gh issue view <n> --repo marin-community/marin --json title,body,comments
```

If principal, resource, or justification is missing, post one `🤖` comment with
the missing fields and stop. Otherwise create the change and link its PR in the
issue.

## Register and edit

For project roles, register and grant in one command:

```bash
uv run --package marin-iac --extra deploy \
  python infra/pulumi/iam_principal.py grant alice@openathena.ai \
    --project-role roles/logging.viewer \
    --project-role roles/monitoring.viewer
```

For a KMS key, secret, bucket, repository, or service account, register first:

```bash
uv run --package marin-iac --extra deploy \
  python infra/pulumi/iam_principal.py register alice@openathena.ai
```

The command reuses or creates the encrypted `human-NNN` registry entry and
writes deterministic YAML. It requires KMS encrypt/decrypt access. Put the
returned `principal: human-NNN` under the requested resource grant; use plain
member strings for automation identities. Never put a personal email in
`iam_data.yaml`, a commit message, or a PR body. IAP `viewers` emails are the
intentional plaintext exception. Quote `*@domain` viewer entries in YAML.

## Verify and hand off

- Edit only the requested grant surfaces. Project-role requests are complete after
  `iam_principal.py grant`.
- Run `./infra/pre-commit.py --files <edited files>` (or `--changed-files`).
  Stage new files first so lint sees them.
- Do not run `pulumi preview` or `pulumi up`; local preview can decrypt and print
  emails, and application belongs to the reviewer.
- Follow `commit` to open a PR against `main` with label `agent-generated`.
  Describe the capability, resource, role, and justification without emails;
  name stack `marin` for IAM or `marin-<service>` for IAP and request
  `review-grant`. Use a capability-based title and assign:

```bash
gh pr edit <n> --repo marin-community/marin \
  --add-assignee yonromai,ravwojdyla,rjpower
```
