---
name: add-grant
description: Add a GCP IAM or IAP user grant to the marin-iac Pulumi stacks, encrypting human principals. Use when someone needs access to a project role, bucket/secret/service account, or an IAP-gated web service (evaldash, grafana), whether requested locally or on a GitHub issue.
---

# Skill: Add a user grant

Turn an access request into a reviewable Pulumi change. Human principals in the
project-IAM data are KMS-encrypted; IAP viewers on Cloud Run services are
plaintext by design. The change is never applied here — a second person runs the
`review-grant` skill, merges, and runs `pulumi up`.

Read first:

- `infra/pulumi/README.md` — the marin-iac stacks, the KMS key, and the
  `pulumi up` prerequisites.
- `infra/pulumi/src/iac/gcp/iam_data.yaml` header — why human `user:` principals
  are encrypted and this file is public.

## Two grant surfaces

Decide which one the request needs before editing anything. A single request can
touch both.

1. **Project / resource GCP IAM** — a role on the `hai-gcp-models` project, the
   KMS key, a Secret Manager secret, a GCS bucket, an Artifact Registry repo, or
   a service account (who may impersonate it). Lives in
   `infra/pulumi/src/iac/gcp/iam_data.yaml`, applied by the **`marin`** stack in
   `infra/pulumi`. Each human `user:<email>` principal is KMS-encrypted once in
   the `principals` registry; grants reference its opaque `human-NNN` ID.
   Service accounts, groups, and domains stay plain strings.

2. **IAP access to a Cloud Run web service** — admitting a person to
   `evaldash.oa.dev`, Grafana, or a similar IAP-gated site. Lives in that
   service's `infra/<service>/Pulumi.marin-<service>.yaml` under `viewers` (a
   plaintext list of emails, `*@domain` wildcards, or qualified IAM members),
   applied by that service's own stack. `domain:openathena.ai` and the Loom VM
   are already admitted to every service; only add a `viewers` entry for someone
   outside the OpenAthena Workspace.

If you are unsure which surface a request means (e.g. "give Alice access to eval
results" could be an IAP viewer on evaldash, a `roles/storage.objectViewer`
grant on the record bucket, or both), ask before editing.

## Collect the request

You need, per grant:

- **Principal** — an email for a person, or a `serviceAccount:`/`group:`/`domain:`
  member for automation. Only personal emails get encrypted.
- **What they need access to** — the specific resource, stated as a capability
  ("read the eval record bucket", "impersonate the ray autoscaler SA") rather
  than a raw role when the requester does not know GCP roles.
- **Why / for how long** — a one-line justification. If the access is temporary,
  note it; `GcpIamCondition` can scope a grant with a CEL expiry, but prefer a
  follow-up removal PR unless the requester asks for an expiry.

Translate a capability into the narrowest role that satisfies it. Reuse a role
already present in `iam_data.yaml` for the same resource class before reaching for
a broader built-in role. If the request is vague or over-broad, ask for
specifics instead of guessing — an IAM grant is hard to walk back once applied.

### Running against a GitHub issue

When invoked to respond to an issue rather than a local prompt:

- Fetch it with `gh issue view <n> --repo marin-community/marin --json title,body,comments`.
- If the issue is missing a principal, the target resource, or a justification,
  **do not guess** — post one comment (prefixed `🤖`) listing exactly what you
  need, and stop. Do not open a half-specified PR.
- If the request is complete, build the change and open a PR (below), then
  comment on the issue linking the PR.

## Register and grant the principal

For project-level roles, update the principal registry and every requested role
in one command:

```bash
uv run --package marin-iac --extra deploy \
  python infra/pulumi/iam_principal.py grant alice@openathena.ai \
    --project-role roles/logging.viewer \
    --project-role roles/monitoring.viewer
```

The command decrypts existing registry entries locally to find and reuse the
person's opaque ID. It encrypts and registers the email once when the person is
new, then writes deterministic YAML. Encryption and lookup need
`roles/cloudkms.cryptoKeyEncrypterDecrypter` on the marin-iac key (the same
access `pulumi up` needs).

For a KMS key, secret, bucket, Artifact Registry repository, or service-account
grant, register the principal first:

```bash
uv run --package marin-iac --extra deploy \
  python infra/pulumi/iam_principal.py register alice@openathena.ai
```

The command prints the existing or new `human-NNN` ID. Add
`principal: human-NNN` to the requested resource grant. Never write a personal
email in plaintext into `iam_data.yaml`, a commit message, or the PR body — the
repo is public. IAP `viewers` emails in `Pulumi.<service>.yaml` are the
documented exception and stay plaintext.

## Make the edit

**Project / resource IAM** — update `iam_data.yaml`:

- Find the grant for the target role and resource, or add one. Project
  roles go in `project_grants`; a bucket/secret/repo/service-account grant goes
  under that resource's entry in `buckets` / `secrets` /
  `artifact_repositories` / `service_accounts` (add the resource entry if it is
  not there yet).
- Project-role requests are already complete after `iam_principal.py grant`.
  For other resource grants, add the registered `principal: human-NNN`
  reference. Add a plain member string for service accounts, groups, domains,
  workload identities, or other automation.

**IAP viewer** — add the email to `viewers` in the service's
`infra/<service>/Pulumi.marin-<service>.yaml`. Quote a `*@domain` wildcard (YAML
reads a leading `*` as an alias).

## Verify and open the PR

- `./infra/pre-commit.py --files <edited files>` (or `--changed-files`), fixing
  anything it reports. `git add` a new file before linting so it is scoped in.
- **Do not run `pulumi preview`/`up`** — a local preview decrypts and prints the
  real emails, and applying is the reviewer's step. CI runs a redacted preview on
  the PR.
- Follow the `commit` skill to commit, push, and open the PR against `main`. Add
  the `agent-generated` label. Title the PR for the capability, not the person:
  `[iac] Grant eval-bucket read to a new operator`, never the email. The body
  states the resource, the role, and the one-line justification — **no personal
  emails**. Note in the body that a reviewer should run `review-grant`, then
  `pulumi up` on the affected stack(s): `marin` for `iam_data.yaml`,
  `marin-<service>` for a `viewers` change.
- Assign the PR to the grant approvers so one of them picks up `review-grant`:

  ```bash
  gh pr edit <n> --repo marin-community/marin \
    --add-assignee yonromai,ravwojdyla,rjpower
  ```
