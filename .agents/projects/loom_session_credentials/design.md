# Make Loom configuration ownership explicit

Loom should tell a user which identity or policy they are changing, who owns it, and which future sessions receive it. Marin's production deployment should declare shared session files such as the CoreWeave kubeconfig instead of relying on manual copies into `/home/app`. This removes the two failure modes seen in the current setup: a valid personal token with the wrong permission set silently wins credential selection, and a credential file exists only because someone copied it onto the VM.

The detailed current-state review and source ledger are in [research.md](research.md).

## Challenges

Loom has several configuration systems because their security and lifecycle requirements differ. Registered non-secret settings have runtime, deployment, and built-in layers. Profiles hold launch policy and write-only environment values. Personal GitHub tokens belong to one user. The GitHub App belongs to the deployment. Repository config belongs in source control. These boundaries are useful, but the Settings information architecture currently groups personal and deployment GitHub credentials together and labels only one of two profile-environment views as `Environment`.

Files add another boundary. Marin's production control plane and ordinary session containers share `/home/app`. A file written there is durable and immediately available to every trusted interactive session, regardless of profile. Declarative provisioning can improve provenance and rotation, but it cannot claim per-profile isolation.

## Costs / Risks

- Moving controls between Settings tabs changes a familiar screen and requires UI journey updates.
- The shared-home provisioner gives every ordinary session access to each declared file. It must be limited to credentials already accepted as machine-wide in this single-tenant deployment.
- A bad secret reference or unsafe path must fail activation before Loom starts. This makes a configuration error visible and may delay a rollout.
- Kubeconfig migration needs an out-of-band one-time secret upload. A code PR cannot safely manufacture or recover that payload.

## Design

Use four ownership classes everywhere Loom explains configuration:

| Owner | Set it in | Appropriate contents |
|---|---|---|
| User | `Settings → Access` | Personal sign-in/password and personal GitHub PAT used by that user's interactive sessions |
| Session profile | `Settings → Agents → Profiles`, preferably deployment IaC in production | Agent/model policy, instructions, repository allowlist, and write-only session secrets |
| Deployment | `Settings → Connections` for status/manual setup; operator IaC and Secret Manager as the production source | Loom GitHub App, Slack App, federations, and shared machine credentials/files |
| Repository | `.weaver/config.toml`, `WEAVER.md`, and `AGENTS.md` | Non-secret setup, repository environment, and workflow instructions |

`Settings → Environment` becomes `Session environment`. Its description states that it is the readable, non-secret environment of the default profile and applies only to future sessions. Write-only values remain under each profile because they have a different disclosure policy.

The GitHub App editor moves from `Access` to `Connections`. `Access` keeps the current user, password, personal GitHub token, approved users, and Loom API tokens. The personal-token card names the interactive order:

1. the launching user's PAT, when configured;
2. the selected profile's explicit `GH_TOKEN`, when configured;
3. a short-lived GitHub App installation token when the profile allowlists the current repository.

The create-token link preselects `Contents: write`, `Pull requests: write`, and `Issues: write`. The copy explains that repository selection and permissions are separate. Loom does not request `Administration: write`, because direct branch pushes are the normal path and creating a fork is a separate operation.

Add a short `Configuration ownership` section to Loom's configuration reference. It becomes the canonical answer to “where does this go?” and links to the credential-specific details. The UI uses the same terms.

Marin's `infra/loom` stack gains a `homeFiles` map:

```yaml
marin-loom:homeFiles:
  .kube/coreweave-iris:
    secretRef: projects/hai-gcp-models/secrets/loom-coreweave-iris-kubeconfig/versions/1
    mode: "0600"
```

Paths must be relative to `/home/app`, normalized, and free of `..`. Secret references must use numbered versions. Modes are limited to owner-readable files (`0400` or `0600`). Pulumi stores only the reference, grants the Loom VM service account access to the named secret, and places the redacted manifest in VM metadata.

Before starting Compose, the activation script resolves every declared secret into a root-only temporary directory, then starts a network-disabled one-shot container with the shared home volume. The container installs each payload through directory file descriptors that reject symlinks, atomically replaces the target, and restores the session-home owner's uid and gid. It records only managed paths in root-owned host state outside the session-writable volume. Removing a declaration deletes only a regular file previously recorded there. Managed paths cannot overlap; file-to-directory transitions remove only stale managed blockers and stop if unmanaged content prevents the change. Secret values never enter metadata, Pulumi state, command arguments, logs, or the ledger.

The production migration creates a dedicated Secret Manager secret, uploads the current kubeconfig as version 1, adds the declaration, previews the IAM and metadata changes, and activates Loom. Rotation adds a numbered version and updates the reference in the stack. The existing manually copied file remains until the first successful managed activation replaces it atomically.

## Testing

Loom's existing Settings Playwright journey verifies the GitHub App editor under `Connections`, the personal token under `Access`, the permission-prefilled PAT URL, and the renamed session environment tab. The frontend typecheck, unit suite, formatter, Rust formatter, and Clippy remain part of the repository gate.

Marin unit tests validate paths, modes, pinned secret references, redacted metadata, Secret Manager IAM grants, and the generated home-file manifest. Filesystem tests run the container-side applicator against a temporary home and root-owned state directory. They cover atomic creation, permissions, managed-file pruning, preservation of unmanaged files, symlink rejection, and file-to-directory transitions. A production preview must show only IAM and VM metadata changes before the kubeconfig version is selected.

## Open Questions

- Should per-profile file mounts become a Loom feature after sessions have isolated homes, or remain deployment-specific?
- Should Loom add an optional repository probe when saving a PAT, even though GitHub does not provide complete fine-grained permission introspection?
- The initial kubeconfig declaration should land in a short deployment follow-up after an operator creates version 1; the mechanism PR must not activate against a missing secret.
