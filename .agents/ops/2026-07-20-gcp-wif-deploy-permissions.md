---
date: 2026-07-20
system: gcp
severity: degradation
resolution: mitigated
pr: none
issue: weaver #543
---

# WIF deployments lacked downstream service permissions

## TL;DR

- The merge of #7440 successfully moved the Ducky and Grafana workflows to WIF
  authentication and KMS-backed Pulumi state, but both deployments then failed on
  service-specific permissions.
- Grafana needed Secret Manager metadata and IAM-policy access, Writer on its
  Artifact Registry repository, and permission to update the IAP web-service policy.
  These privileges were added to the Pulumi permissions stack with additive,
  protected, resource-scoped grants where GCP supports them.
- [Grafana workflow run 29780970098](https://github.com/marin-community/marin/actions/runs/29780970098)
  completed after the permissions were applied. Cloud Run revision
  `marin-grafana-00016-2dt` became ready, and `https://grafana.oa.dev` returned the
  expected IAP redirect.
- Ducky remained blocked because its five configured Secret Manager resources do
  not exist. Their values must be migrated from the corresponding GitHub secrets
  before resource-scoped accessor grants can be added.

## Original problem report

After #7440 merged, the request was to run the updated actions, verify that WIF and
the permissions worked in a real deployment, and file a fix PR for any problems.

The Grafana workflow first failed with a missing
`secretmanager.secrets.list` permission. Later attempts exposed missing Artifact
Registry upload access and missing IAP policy-management access. The Ducky workflow
reported `secretmanager.versions.access` denied for `ducky-cw-secret-key`.

## Investigation path

1. The merge-triggered Ducky and Grafana runs were inspected step by step. Both
   successfully exchanged the GitHub OIDC token through WIF, authenticated to GCP,
   and opened their KMS-backed Pulumi stacks. This isolated the failures from the
   WIF/KMS migration itself.
2. The Grafana failure was traced to the optional-secret discovery and the
   resource-level Secret Manager IAM grants in `infra/grafana/__main__.py`.
3. A read-only inventory confirmed that all four Grafana secrets existed, while
   none of the five Ducky secrets configured in
   `infra/ducky/Pulumi.ducky-marin.yaml` existed.
4. Secret metadata visibility and resource-scoped secret IAM management were
   applied through the `marin-permissions` stack. The next Grafana run passed the
   secret operations and failed while pushing its image.
5. Artifact Registry Writer was granted only on the `marin-grafana` repository.
   The following run pushed the image and updated Cloud Run, then failed while
   setting the IAP viewer binding.
6. A custom role containing only
   `iap.webServices.getIamPolicy` and `iap.webServices.setIamPolicy` was applied.
   Workflow run 29780970098 then completed successfully.
7. A final permissions preview reported 19 unchanged resources. The deployed Cloud
   Run revision was ready, and the public Grafana hostname reached the expected IAP
   redirect.
8. Ducky was left blocked rather than creating empty secrets or granting broad
   project-level payload access. PR #7394 had documented population of these five
   secrets as a prerequisite to its first deployment.

## User course corrections

The task moved from static configuration review to a live deployment check after
the migration PR merged. The investigation therefore used each workflow failure as
the next least-privilege permission probe. Earlier configuration-shape tests were
removed; the remaining tests cover permission behavior rather than restating the
Pulumi config.

## Root cause

The initial permissions component covered WIF token exchange, KMS-backed Pulumi
state, and the baseline deployment roles. It did not model all of the downstream
capabilities exercised by Grafana: secret metadata discovery, per-secret IAM
updates, image uploads, and IAP policy updates.

Ducky had a separate prerequisite failure. Its Pulumi configuration referenced five
Secret Manager resources whose payloads had not been migrated from GitHub Actions
secrets. A permission grant cannot repair a missing resource or reconstruct its
value.

## Fix

`infra/iac/src/iac/gcp/permissions.py` now models additive grants for Secret Manager
metadata viewers, per-secret IAM managers, repository-scoped Artifact
Registry writers, and narrowly defined IAP IAM managers. The active
`infra/permissions/Pulumi.hai-gcp-models.yaml` configuration grants Grafana only the
capabilities exercised by its deployment. The resources remain protected against
accidental deletion.

`infra/permissions/README.md` documents the permission boundaries, and
`lib/ducky/README.md` records the exact GitHub-secret-to-Secret-Manager mapping and
the bootstrap order. Ducky's accessor change remains deferred until an authorized
operator creates and populates those secrets.

## How OPS.md could have shortened this

No component OPS.md change was appropriate: both failures happened in GitHub
Actions before Iris submitted any work. The deployment capability inventory and
secret bootstrap procedure now live beside the relevant Pulumi stacks in
`infra/permissions/README.md` and `lib/ducky/README.md`.

## Artifacts

- [Debugging log](../../docs/debug-log-wif-deploy-secret-access.md)
- [Successful Grafana deployment](https://github.com/marin-community/marin/actions/runs/29780970098)
- [Initial Grafana failure](https://github.com/marin-community/marin/actions/runs/29779777482)
- [Initial Ducky failure](https://github.com/marin-community/marin/actions/runs/29779777439)
- [WIF/KMS migration PR #7440](https://github.com/marin-community/marin/pull/7440)
- [Deployed Grafana service](https://grafana.oa.dev)
