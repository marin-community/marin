# Background Research Brief

- Effort: medium
- Stop rule: stop when repository history, current code, and deployment code agree on the existing ownership boundaries and no new source changes the proposed split
- Date: 2026-08-15

## Question

Where should a Loom user or operator configure GitHub identity, service credentials, session environment, repository defaults, and files such as a kubeconfig? How should the Marin deployment preload shared session state without copying secrets into Pulumi state?

## Current Loom Context

Loom exposes eight Settings categories. `Agents` owns launch profiles and their write-only environment. `Environment` is a readable compatibility view over the default profile. `Access` currently combines the signed-in user, password, personal GitHub token, the deployment GitHub App, approved users, and API tokens. `Connections` contains GitHub/Slack runtime settings and Slack connection state, but the GitHub App editor remains under `Access` ([Settings.vue](https://github.com/marin-community/loom/blob/f5ef7f2373913d738e093ca744b78bc5f2bed8c9/crates/loom/frontend/src/views/Settings.vue#L20-L88), [AccountPanel.vue](https://github.com/marin-community/loom/blob/f5ef7f2373913d738e093ca744b78bc5f2bed8c9/crates/loom/frontend/src/components/AccountPanel.vue#L265-L365)).

Registered non-secret settings already have a sound precedence rule: runtime override, deployment value, then built-in default. The API and UI expose that provenance. The rule does not cover credentials, profile environment, or files in the shared home volume ([configuration.md](https://github.com/marin-community/loom/blob/f5ef7f2373913d738e093ca744b78bc5f2bed8c9/docs/configuration.md#L1-L22)).

Interactive GitHub access has three credential sources. The launching user's PAT is exported as `GH_TOKEN` when set. Otherwise a profile `GH_TOKEN` or lower environment layer can supply a long-lived credential. Profiles with a repository allowlist can instead broker a short-lived GitHub App token scoped to the current repository. The same shell variable therefore hides several owners and lifetimes. Loom issue [#195](https://github.com/marin-community/loom/issues/195) identified this ambiguity and recommended purpose-specific labels and source diagnostics. PRs [#289](https://github.com/marin-community/loom/pull/289) and [#291](https://github.com/marin-community/loom/pull/291) added repository-scoped App tokens but did not reorganize Settings.

## Current Marin Deployment Context

`infra/loom/Pulumi.marin-loom.yaml` declares global settings, profiles, GitHub repository allowlists, federations, and profile secret references. `infra/loom/infrastructure.py` renders those into Loom's deployment manifest and grants the VM service account Secret Manager access. Profile environment entries currently accept only `secretRef` values.

The Loom control plane and every ordinary session container mount the same Docker volume at `/home/app` ([docker-compose.yml](https://github.com/marin-community/marin/blob/deee9eabb335772460de9237e17362ce094c127c/infra/loom/runtime/docker-compose.yml#L1-L63)). The CoreWeave kubeconfig was copied manually to `/home/app/.kube/coreweave-iris`, so it persists across sessions and deploys but has no IaC provenance, rotation contract, or restore path. The startup script already resolves a pinned `LOOM_DOTENV` Secret Manager version without placing the payload in Pulumi state ([startup-script.sh](https://github.com/marin-community/marin/blob/deee9eabb335772460de9237e17362ce094c127c/infra/loom/startup-script.sh#L77-L112)). This is the closest reusable mechanism.

The `hai-gcp-models` project has no general CoreWeave Iris kubeconfig secret as of 2026-08-15. It has a Marinfold-specific kubeconfig secret, which should not be reused without confirming its scope. The existing `CW_KUBECONFIG` GitHub Actions recovery declaration is write-only and cannot supply a GCE startup payload.

## Internal Prior Work

- Loom issue [#195](https://github.com/marin-community/loom/issues/195) proposed purpose-aware GitHub credentials, explicit labels, and source diagnostics. It is closed without a single implementing PR; the App-token work landed only part of the proposal.
- Loom issue [#174](https://github.com/marin-community/loom/issues/174) established deployment-managed profiles and Secret Manager references. Its sample sets `KUBECONFIG` directly from a secret payload, but Kubernetes clients expect a file path, so that shape does not materialize a usable kubeconfig.
- Loom PR [#113](https://github.com/marin-community/loom/pull/113) introduced the personal PAT store and deliberately exports it as `GH_TOKEN` for unrestricted interactive sessions.
- Loom PR [#177](https://github.com/marin-community/loom/pull/177) introduced write-only profile environments and deployment-managed automation policy.
- Marin PR [#8300](https://github.com/marin-community/marin/pull/8300) configured interactive profiles to use the GitHub App fallback when a user PAT is absent.

## External Prior Art

GitHub fine-grained PAT repository selection and permission selection are independent. A token that targets every repository can still lack `Contents: write`. GitHub's create-token URL accepts query parameters that preselect permissions, so Loom can link users to a safer starting configuration. Fork creation has a different permission set and is not required for Loom's normal direct-branch workflow.

Google Secret Manager recommends granting access to individual secrets and pinning versions for reproducible consumers. Pulumi state should contain resource references, never kubeconfig contents.

## Negative / Failed Leads

- A profile `KUBECONFIG` secret reference sets the variable to the kubeconfig contents, not a filesystem path.
- Committing the kubeconfig or putting it in `.weaver/config.toml` would expose a credential to the repository.
- Reusing the daemon dotenv would couple an independently rotated cluster credential to all Loom application secrets.
- A generic Loom profile-file feature would imply profile isolation that the shared `/home/app` volume does not provide. The first change should stay deployment-scoped and name the shared trust boundary.
- Echo found the App-token rollout and current deployment docs but no prior shared-home-file implementation.

## Evidence Map

### Claim: configuration needs an ownership vocabulary

- Support: Settings places personal and deployment GitHub credentials in one component; issue #195 records the resulting source ambiguity.
- Contradictions: the registered non-secret settings registry already exposes provenance and should remain unchanged.
- Directness to Marin: high; the reported PAT incident followed this ambiguity exactly.
- Confidence: high.
- Action: organize help and labels around personal, profile, deployment, and repository ownership.

### Claim: shared home files belong in Marin deployment IaC

- Support: production sessions already share `/home/app`; the startup script already retrieves pinned secret versions; the kubeconfig is already treated as a shared interactive credential.
- Contradictions: files are visible to every session sharing the volume and cannot enforce profile isolation.
- Directness to Marin: high; this replaces the existing manual copy with a reviewed declaration.
- Confidence: high for trusted interactive sessions, low for restricted or multi-tenant use.
- Action: add deployment `homeFiles` declarations backed by pinned Secret Manager references and document the boundary.

## Recommended Changes

1. Reorganize Loom Settings labels and help around four owners, move the GitHub App editor to `Connections`, rename the default environment surface, and show the interactive GitHub credential order.
2. Prefill required fine-grained PAT permissions in Loom's create-token link.
3. Add `homeFiles` to the Marin Loom Pulumi program. Materialize pinned Secret Manager versions into the shared home volume before Loom starts and grant access only to declared secrets.
4. Document the current settings and credential map in Loom and the production preload/rotation procedure in `infra/loom/README.md`.
5. Migrate `/home/app/.kube/coreweave-iris` after a dedicated Secret Manager secret and numbered version exist. Do not create or upload the credential in a code PR.

## Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Loom Settings | code | `crates/loom/frontend/src/views/Settings.vue` | current tabs | high | SHA pinned above |
| Loom Account | code | `crates/loom/frontend/src/components/AccountPanel.vue` | credential placement | high | SHA pinned above |
| Loom configuration policy | docs | `docs/configuration.md` | setting precedence | high | current main |
| Loom #195 | issue | https://github.com/marin-community/loom/issues/195 | credential ambiguity | high | direct prior design |
| Loom #174 | issue | https://github.com/marin-community/loom/issues/174 | profiles and secret refs | medium | sample conflates file and env |
| Marin Loom Pulumi | code | `infra/loom/infrastructure.py` | IaC capabilities | high | current checkout |
| Marin Compose | code | `infra/loom/runtime/docker-compose.yml` | shared home boundary | high | current checkout |
| Secret inventory | live metadata | `gcloud secrets list` on 2026-08-15 | migration prerequisite | high | names only; no values read |

## Handoff

- Open question: should a later Loom release add per-profile file mounts after session homes are isolated, or keep file materialization deployment-specific?
- Open question: should Loom validate a PAT against one selected repository at save time, given that fine-grained permission introspection is incomplete?
- Stop reason: repository history and current code converge on a small UX change plus a Marin-only shared-file provisioner; further searching did not change that boundary.
