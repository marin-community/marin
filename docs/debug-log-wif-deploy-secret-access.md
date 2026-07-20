# Debugging log for WIF deployment secret access

Bring the Ducky and Grafana deployment workflows through their first keyless production
deployment after the WIF/KMS migration.

## Initial status

The merge of #7440 triggered both workflows on `main`. Both service accounts successfully
authenticated through WIF, configured Artifact Registry credentials, and opened their
KMS-backed Pulumi stacks. The failures occurred later:

- Grafana could not list Secret Manager metadata while probing for its optional SMTP secret.
- Ducky could not access the first configured secret version while resolving its Iris job
  environment.

## Hypothesis 1: grant the service deployment permissions it actually uses

Grafana needs metadata visibility for the optional-secret probe and must manage the runtime
service account's IAM access on four existing secrets. It does not need secret payloads. Add a
custom role containing only secret metadata and IAM-policy permissions, grant metadata viewer
at the project, and grant the custom role on the four secrets.

Ducky's five configured secret resources do not exist in `hai-gcp-models`. PR #7394 documented
their creation and population as a prerequisite to the first Pulumi deployment; IAM alone
cannot supply their values. Do not broaden the deploy account to every project secret or
create empty versions merely to make the deployment advance.

## Changes to make

- Extend `marin-permissions` with protected, additive Grafana Secret Manager grants.
- Apply the permissions stack and confirm a clean follow-up preview.
- Rerun Grafana and follow the deployment to completion.
- Leave the exact Ducky secret migration as an explicit blocker unless the five GitHub secret
  values are migrated by an authorized operator or a separately reviewed bootstrap workflow.

## Results

The first permissions update let Grafana pass its secret probe and refresh all secret IAM
members. The image then built successfully but the push failed because the deploy account had
Artifact Registry Reader rather than Writer. Add resource-level Writer on only the
`marin-grafana` repository and rerun. That update allowed the image push and Cloud Run rollout;
the final IAP viewer grant then failed because the deploy account could not manage IAP web
service policies. Add a custom role with only the two required IAP web-service IAM permissions.

After the IAP role was applied, workflow run
[`29780970098`](https://github.com/marin-community/marin/actions/runs/29780970098) completed. WIF,
KMS state access, secret IAM refresh, the image push, Cloud Run update, and IAP viewer grant all
succeeded. Cloud Run reported `marin-grafana-00016-2dt` as both the latest created and latest
ready revision; `https://grafana.oa.dev` returned the expected IAP redirect. The permissions
stack then previewed with 19 unchanged resources.

## Future work

- [ ] Migrate the five Ducky repository secrets into their documented Secret Manager names.
- [ ] Grant `iris-ci-smoke` accessor only on those five secrets after they exist.
