# Grafana operations

## Secrets and rotation

All secrets live in Secret Manager, hand-placed, and reach the container as env
vars via the `CloudRunService` `secrets` field; the values never enter Pulumi or
git. The deploy account is fail-closed on secret creation, so the program only
references secrets — each must exist and be declared in
`infra/pulumi/src/iac/gcp/grafana.py` before the `marin` stack grants access.

Loom itself needs no secret: the bridge authenticates with the Cloud Run service
account and short-lived Google/Loom tokens, and Pulumi owns the
identity-to-profile binding. The Slack bot token is the one alerting credential.

| Env var | Secret | Feeds |
|---|---|---|
| `GITHUB_APP_PRIVATE_KEY` | `marin-grafana-github-app-private-key` | ferry/build/nightly panels |
| `GF_DATABASE_PASSWORD` | `cloudsql-grafana-password` | Grafana's Postgres state (see Deploy) |
| `CW_READ_TOKEN` | `marin-grafana-cw-read-token` | k8s source (all CW clusters) |
| `SLACK_ALERTS_BOT_TOKEN` | `marin-grafana-slack-bot-token` | the bridge's alert announcements |
| `GF_SMTP_PASSWORD` | `marin-grafana-smtp-credentials` | Grafana SMTP (email alerts, optional) |

`GF_DATABASE_PASSWORD`, `CW_READ_TOKEN`, and `SLACK_ALERTS_BOT_TOKEN` must exist
before a deploy — Cloud Run fails to start a revision that references a missing
secret. `GF_SMTP_PASSWORD` and `GITHUB_APP_PRIVATE_KEY` are optional: `__main__.py`
probes for each and wires it only when the secret exists (the GitHub App also
needs its `github_app_client_id` config). Unset, the GitHub panels deploy
unauthenticated and the build panel shows no data.

`CW_READ_TOKEN` is an org-wide CoreWeave API token minted with only the `read` role
(CKS binds it to the built-in `view` ClusterRole): read-only kubectl across every
cluster in the org, no Secrets, no writes. The built-in role omits Nodes and the
NodePool CRD, so the CoreWeave Pulumi stacks bind each exact Managed Auth username
to the read-only `marin-grafana-node-reader` role for those resources. The usernames
live under `provisioning.coreweave.grafana_observer_rbac` in each Grafana cluster
config so both tokens can retain access during a rotation.

Rotation is overlap-safe:

1. Mint a second read-role token in the CoreWeave console. Save it as a single
   line with no CR/LF; Secret Manager preserves trailing newlines.
2. Use the token against one cluster's `SelfSubjectReview` to get its
   `cwtoken-…` username. Append it to `grafana_observer_rbac.usernames` in
   `cw-us-east-02a.yaml`, `cw-us-east-08a.yaml`, and `cw-rno2a.yaml`, retaining
   the old username during the handoff.
3. Preview and update the three CoreWeave Pulumi stacks. Verify both tokens can
   list Nodes and NodePools, while pod creation, Secret reads, and impersonation
   remain denied.
4. Add the new token as a `marin-grafana-cw-read-token` version, deploy a fresh
   Grafana revision, and verify every k8s bridge route.
5. Remove the old username from the three configs and update the stacks again.
   Then disable the old secret version and revoke the old CoreWeave token.

The same Secret Manager overlap pattern applies to the Slack bot token and SMTP
password: add a version, redeploy, then retire the old credential. Write the
payload with `printf '%s'`, not `echo` — a trailing newline reaches the
`Authorization` header, though the bridge strips it defensively.

Creating the secrets:

1. CoreWeave console → API access → new token (e.g. `grafana-observer`) with only the
   `read` role, then
   `echo -n "<token>" | gcloud secrets create marin-grafana-cw-read-token --project=hai-gcp-models --data-file=-`
2. The `@russbot` bot-user token (`xoxb-…`) the bridge announces alerts with —
   the same Slack app Loom posts as, so the announcement and Loom's replies come
   from one identity. Reuse the token from Loom's `LOOM_DOTENV`
   (`LOOM_SLACK_BOT_TOKEN`; see `infra/loom`), then
   `printf '%s' "xoxb-..." | gcloud secrets create marin-grafana-slack-bot-token --project=hai-gcp-models --data-file=-`
3. Apply the `marin` GCP stack, which grants the deploy account IAM management on
   that secret and the runtime account access to it (declared in
   `infra/pulumi/src/iac/gcp/grafana.py`). Without it the Grafana deploy fails
   Cloud Run's secret-access validation, even once the value exists.
4. Confirm `slack_alerts_channel` names the channel you want and that `@russbot`
   is in it. It is `#marin-alerts` (`C0BN20081CH`); to move it, take the id from
   the channel's Copy link and `/invite @russbot` there. `pulumi up` fails naming
   the key if it is ever unset.
5. (optional, enables email) Gmail app password for grafana@openathena.ai, then
   `printf '%s' "<app-password>" | gcloud secrets create marin-grafana-smtp-credentials --project=hai-gcp-models --data-file=-`
6. Send a test notification to `ops-critical` and confirm email arrives, one alert
   message lands in the channel, and its thread gains a triage-session link. Test
   `ops-slack` too: it should post one message with a session link in its thread.

## Develop

```bash
uv run pytest                     # bridge unit tests
cd marin-infra-panel
npm ci
npm run typecheck && npm run lint && npm run test:ci && npm run build
docker build -t marin-grafana .
docker run --rm -p 3000:8080 -e PORT=8080 marin-grafana
# → http://localhost:3000 (without an IAP identity header, anonymous Viewer; panels need VPC access to finelog)
```

No alerting credentials are needed to start: both contact points are loopback
webhooks, so Grafana boots without them and the bridge answers 503 on the alert
routes until `LOOM_ALERT_URL` and the Slack settings are present.

Panels only render against the real VPC: querying needs credentials that list the
finelog VMs and a network path to them. Locally you get Grafana, the provisioned
dashboards, and a bridge that 500s on query — enough to confirm every dashboard
parses and its variables survive provisioning:

```bash
curl -s 'http://localhost:3000/api/search?type=dash-db'
curl -s http://localhost:3000/api/dashboards/uid/marin-accel
```

## Deploy

Pulumi owns the deploy: the runtime service account, Artifact Registry repo and image, the
Cloud Run service, and the IAP settings. The `marin` infrastructure stack separately owns the
runtime account's project and secret grants plus the Cloud Run and IAP grants. The service
and its image build come from the reusable `iac.gcp.cloud_run.CloudRunService` component
(`infra/pulumi`); this directory is its own Pulumi project. It runs on the shared repo venv
and shares `infra/pulumi`'s state backend.

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev  # once: let buildx push to Artifact Registry
uv run --all-packages --extra deploy marin-deploy grafana rollout
```

The deploy command loads the Cloudflare provider token from Secret Manager and
Pulumi previews the update before asking for confirmation.

Add IAP viewers through `infra/pulumi/src/iac/gcp/grafana.py`, referencing encrypted principals
from `iam_data.yaml`, and apply the `marin` stack. The Grafana stack does not own IAM grants.

Production reads the `grafana-alerts` URL and profile from the `marin-loom`
stack. Apply that stack before rolling a Grafana revision with
`marin-grafana:loom_alerts` enabled. For the first deployment in a new project,
disable Loom alerts to create the Grafana service account, apply Loom, then
enable the integration and deploy Grafana again.

The stack uses the shared `marin-iac-key` KMS secrets provider. The operator needs
`roles/cloudkms.cryptoKeyEncrypterDecrypter` on that key; no passphrase is used.

The rollout builds the Dockerfile with buildx, pushes it digest-pinned to Artifact
Registry, and rolls the service to that digest. `min` and `max` instances are both 1: one
warm instance serves this internal dashboard, min 1 keeps alert evaluation warm and first
paint off a cold start, and max 1 avoids duplicate alert notifications from parallel
evaluators.

Grafana's state is the `grafana` database on the shared `marin-metadata` Cloud SQL Postgres
(`infra/cloudsql`). `__main__.py` reads the instance connection name from a
`pulumi.StackReference` to the `marin-cloudsql` stack, mounts the Cloud SQL connector socket
under `/cloudsql`, and hands the socket directory to `entrypoint.sh` as
`DATABASE_SOCKET_DIR`, which composes `GF_DATABASE_URL` from it (Grafana's host:port
settings reject the colons in a connection name). `GF_DATABASE_PASSWORD` comes from the
`cloudsql-grafana-password` secret. Prerequisite: bring up the `marin-cloudsql` stack and
create the `grafana` SQL user + its secret version (see `infra/cloudsql/README.md`) before
the rollout, or Grafana fails to reach its database.

Grafana's metadata pool is capped at 10 open and 2 idle connections so bursts
leave capacity for Marina and the other databases on this instance.

IAP is the outer gate. Its `X-Goog-Authenticated-User-Email` header becomes a Grafana
auth-proxy account. The container's nginx listener adds a fixed `Editor` role for those
accounts, which lets every admitted person create and expire alert silences. Grafana syncs
the role on authenticated requests. Before Grafana starts, `python -m grafana_migrations`
applies pending `src/grafana_migrations/migrations/mNNNN_*.py` files and records each version
in `marin_schema_migrations`. Migration `m0001` changes memberships created as `Viewer` by
older revisions to `Editor`; Grafana's normal role synchronization rejects that change when
the organization has no `Admin`. Requests without IAP's identity header remain anonymous
`Viewer`.

Cloud Run grants `run.invoker` only to the IAP service agent, and direct Cloud Run IAP
[routes every ingress path through IAP](https://cloud.google.com/run/docs/securing/identity-aware-proxy-cloud-run).
IAP [strips client-provided `X-Goog-*` headers](https://cloud.google.com/iap/docs/signed-headers-howto)
before forwarding a request, so the presence of `X-Goog-Authenticated-User-Email` safely
distinguishes authenticated users in this deployment. To allow anonymous viewing later, grant
IAP's `roles/iap.httpsResourceAccessor` role to `allUsers`. IAP then admits ordinary requests
[without checking authentication credentials](https://cloud.google.com/iap/docs/force-login),
so they have no identity claim and remain Grafana `Viewer`. A sign-in link to
`?gcp-iap-mode=FORCE_LOGIN` lets a visitor authenticate through IAP; subsequent requests carry
the email header and become `Editor`.

The OAuth consent screen is project-level and shared across the project's IAP services. The
central IAM config admits the OpenAthena Workspace domain, Loom VM service account, and
service-specific viewers. The Cloud Run component registers the Marin desktop OAuth client as
a programmatic audience. IAP admission is independent of the Grafana organization role.

The ferry, build, and nightly panels read the GitHub API, which gates the GraphQL
build query behind auth even for public repos. The bridge authenticates as the
"Marin Ops Agent" GitHub App (`src/github_app.py`): it signs a JWT with the app's
private key, looks up its installation, and mints a read-only token scoped to the
repos the panels read (the main repo and every nightly lane repo, all under
`marin-community`), refreshing it before expiry. A static token that expired is
what blanked the build panel.

The app's client id is committed as `github_app_client_id` (not secret); the
private key is hand-placed. `__main__.py` wires the app only when both are present,
so the merge-triggered deploy never blocks. The client id is already set, so
enabling auth is one step (plus its permissions grant):

```bash
# Its grants are declared in infra/pulumi/src/iac/gcp/grafana.py, so:
gcloud secrets create marin-grafana-github-app-private-key \
  --project=hai-gcp-models --data-file=key.pem
```

Install the app on `marin-community` with access to the main repo and every
nightly lane repo (`evalchemy`, `harbor`, `MarinSkyRL`, `vllm`, `tpu-inference`),
read-only on Contents, Metadata, Commit statuses, Checks, and Actions. The minted
token is attenuated to that subset even if the app holds broader grants.

