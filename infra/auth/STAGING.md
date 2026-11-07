# Synthetic CW authorization staging

The `staging` stack in `infra/auth` and `us-west4-staging` stack in
`infra/postgres` own a persistent synthetic deployment on
`marin_US-WEST-04A`. They use `marin-auth-staging`,
`marin-auth-db-staging`, and `marin-auth-task-staging`. They do not alter the
production Iris controller or its RBAC.

Three OpenBao Raft replicas and three PostgreSQL instances share the cluster's
two observed hosts and zones. PostgreSQL requires one synchronous standby.
OpenFGA and grant brokers use separate database owners; recovery databases are
owned by the corresponding source owner. `shared-vast` is a shared storage
backend with `Retain` reclamation. These choices do not establish independent
host or storage failure tolerance, or management isolation from cluster-wide
Iris permissions.

## Bootstrap

Use the repository environment and a private directory outside the checkout.
The helper creates mode-0600 files in a mode-0700 directory. Repeating `prepare`
preserves passwords, signing identity, and certificates for this deployment.
OpenBao, OpenFGA, and the broker have separate leaf private keys and service
SANs, signed by the private staging CA. Each leaf key mounts only into its own
service Secret.
Do not put this directory, initialization keys, or token values in Pulumi config,
command arguments, logs, or the repository.

```bash
export KUBECONFIG=/home/power/.kube/coreweave-iris
STAGING_PRIVATE=/home/power/.local/state/marin-auth-staging
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" prepare
pulumi -C infra/postgres preview --stack us-west4-staging --diff
pulumi -C infra/postgres up --stack us-west4-staging
```

On the first apply, run this from another terminal after the database namespace
exists, while CloudNativePG waits for its role credentials:

```bash
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" install-database-secrets
```

The checked-in auth config pins the OpenBao, OpenFGA, and TLS gateway images.
Deploy without `broker_image` initially. After the auth namespace exists,
install its external Secrets while its first apply waits for them:

```bash
pulumi -C infra/auth preview --stack staging --diff
pulumi -C infra/auth up --stack staging
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" install-secrets
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" install-database-ca
```

Forward the three OpenBao pods to separate local ports, then initialize and
unseal them using verified TLS. Port-forwarding each pod can remain in its own
terminal:

```bash
kubectl --context marin_US-WEST-04A -n marin-auth-staging port-forward pod/openbao-0 18200:8200
kubectl --context marin_US-WEST-04A -n marin-auth-staging port-forward pod/openbao-1 18201:8200
kubectl --context marin_US-WEST-04A -n marin-auth-staging port-forward pod/openbao-2 18202:8200
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" initialize \
  --bao-address https://localhost:18200 \
  --bao-address https://localhost:18201 \
  --bao-address https://localhost:18202
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" install-secrets
```

Initialization uses one Shamir share with threshold one for this synthetic
deployment. Its root token and unseal share remain in `initialization.json` in
the private directory. This is not a production key-custody procedure. Restarted
OpenBao pods must be unsealed again; rerunning `initialize` preserves entity IDs
and verifies them against the saved metadata. A joining follower may need a
repeat after its Raft join completes.

The fixed generated userpass passwords enroll `launcher` and `iris-controller`
at `auth/synthetic/`. Both receive only `marin-auth-identity`. Initialization
prints public subject/principal metadata. The controller maps to
`service:iris-staging`; it must not share the launcher's principal mapping.
The broker token has a 24-hour renewable TTL. Renew or replace that external
Secret before expiry. The synthetic TLS leaf expires after 14 days and its CA
after 30 days. Renew certificates before expiry, update the external TLS/trust
Secrets, and replace/unseal one OpenBao replica at a time. `prepare` deliberately
does not rotate a deployment's certificates or passwords.

Forward one OpenFGA pod's loopback port 8080 to localhost:18080. Publish the
synthetic model/relationships, then set the returned store/model IDs in
`openfga_store_id` and `broker_policy.openfga`:

```bash
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" publish-policy
```

Configure the broker's database as public references only:

```yaml
database:
  backend: postgres
  url_file: /var/run/marin-auth/database/url
  ca_file: /var/run/marin-auth/ca/ca.crt
```

Set `broker_image` to a reviewed digest and provide the complete `broker_policy`
with the enrolled subject mappings before applying again. Two brokers share the
PostgreSQL grant/lifecycle ledger. The OpenFGA write API remains bound to pod
loopback; workload traffic can reach only its authenticated TLS Check gateway.
The private OpenBao issuer does not establish CoreWeave WIF: provider federation
also needs registered, publicly reachable OIDC discovery and JWKS.

## Isolated Iris tasks

The auth stack creates `auth-staging-proof` in `marin-auth-staging`. Its task
Role covers pods, logs, configmaps, events, delivered Secrets, and disruption
budgets only in `marin-auth-task-staging`. A separate ClusterRole permits only
reading nodes. It cannot mutate nodes, provision nodepools, create priority
classes, or read auth/database Secrets through the API.
Task Secret reads require an exact name; listing Secrets is denied. The proof
can exec into its isolated task pods to release a deterministic success case.
Its LocalQueue `auth-staging` references the existing `iris-cq` ClusterQueue;
the proof does not modify that ClusterQueue or request GPUs.

The proof Pod must explicitly mount its ServiceAccount token and carry labels
`app.kubernetes.io/name=auth-staging-proof` and
`marin.dev/auth-client=true`. Its network permits the broker, OpenBao,
staging PostgreSQL, approved DNS, and the observed Kubernetes API addresses.
Tasks run without a ServiceAccount token. The task namespace denies ingress and
egress by default. The sole task exception selects `iris.managed=true`,
`iris.runtime=iris-kubernetes`, and `iris.egress=internet`; it permits DNS and
TCP 8090 to broker pods in the auth namespace. Both namespaces enforce
restricted Pod Security; task images need a numeric non-root user.

## Execution proof image

`execution/Dockerfile` overlays current Iris, Rigging and proof source onto the
reviewed `80703c8b` proof image. It preserves PostgreSQL 17 recovery tools, Bao,
OpenFGA, Node and Git. Its Python environment and source live under `/opt`, so
the task's `/app` workdir mount does not hide them. The final user is `1000:1000`.

Build Loom and its matching Tapestry first, including the authenticated callback
changes. Verify both binaries run inside the base image before copying them.
Keep the small build inputs in the ignored repository artifact directory:

```bash
mkdir -p artifacts/auth-execution-tools
install -m 755 /home/power/.cache/cargo-build/debug/loom artifacts/auth-execution-tools/loom
install -m 755 /home/power/.cache/cargo-build/debug/tapestry artifacts/auth-execution-tools/tapestry
strip --strip-debug artifacts/auth-execution-tools/loom artifacts/auth-execution-tools/tapestry
docker buildx build --network=none --load \
  --build-context execution-tools=artifacts/auth-execution-tools \
  --build-context loom-source="$AUTH_LOOM_SOURCE" \
  --build-arg IRIS_REVISION_DATE="$(git log -1 --format=%cs -- lib/iris)" \
  --file infra/auth/execution/Dockerfile --tag marin-auth:execution-10950 .
```

Run the image locally without network, including with a temporary empty `/app`
mount, before publishing it. Pin the published digest for the proof driver and
task image. `python -m infra.auth.prove_execution --help` lists the private
configuration, identity, public-key, CA and binary inputs. The helper creates a
distinct private callback credential for each isolated Loom instance and gives
its file path only to Loom and its local lifecycle broker.

## Recovery drills

Pause synthetic writes while taking and comparing logical PostgreSQL backups.
Run the helper inside the isolated proof Pod, with the source URL Secret and
database CA mounted at the paths below. It restores only to the dedicated
`grants-recovery` or `openfga-recovery` database, compares each table's row count
and digest, and preserves its private custom-format dump. It never restores
over the source database. Each output dump is created once; use a fresh private
output directory for a second drill.

```bash
uv run --no-sync -m infra.auth.recover_staging --directory /private/recovery database \
  --url-file /var/run/marin-auth/database/url \
  --ca-file /var/run/marin-auth/ca/ca.crt \
  --postgres-bin /usr/lib/postgresql/17/bin
```

Download an OpenBao Raft snapshot through the local verified port-forward:

```bash
uv run --no-sync -m infra.auth.recover_staging --directory "$STAGING_PRIVATE" openbao-snapshot
```

Restore it in a separate container with `--network=none`, mounting the private
directory read/write. The restore helper requires a network namespace containing
only loopback, so restored peer metadata cannot reach the original cluster.
Use a digest-pinned proof image with the repository modules and Bao binary:

```bash
docker run --rm --network=none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$STAGING_PRIVATE,dst=/private" \
  --entrypoint python "$PROOF_IMAGE_DIGEST" \
  -m infra.auth.recover_staging --directory /private openbao --bao /usr/local/bin/bao
```

The helper force-restores into a disposable local Raft instance, unseals it with
the saved share, compares signing keys and the synthetic entity, and logs in
with the deployment's fixed password. A same-cluster logical PostgreSQL restore
and local Raft recovery do not establish off-cluster DR, WAL/PITR coverage,
concurrent-write recovery, node loss, or storage loss guarantees. Row equality
alone does not establish authorization safety after restoring an old backup:
reconcile revocations before reopening restored authorization services.

## Retirement

Keep this deployment only while its synthetic proof remains useful. Before
retirement, save needed evidence/backups and record each staging PVC's UID and
bound PV name. Inspect every resource before deleting it. Unprotect only these
stacks, delete their three namespaces, then remove their remaining Pulumi
resources. Namespace deletion also removes the deliberately retained servers.

```bash
pulumi -C infra/auth state unprotect --all --stack staging
pulumi -C infra/postgres state unprotect --all --stack us-west4-staging
kubectl --context marin_US-WEST-04A delete namespace marin-auth-task-staging marin-auth-staging marin-auth-db-staging
pulumi -C infra/auth destroy --stack staging
pulumi -C infra/postgres destroy --stack us-west4-staging
```

Before destroying the PostgreSQL operator's cluster-scoped resources, confirm
that no other CloudNativePG installation uses its CRDs or admission resources.
The `Retain` storage class leaves PVs and backend data. Follow the recorded
UID/Released-state reclamation procedure in [the database guide](../postgres/README.md)
for only these disposable staging volumes. Retire the private credential
directory after the deployment and required backup retention have ended.
