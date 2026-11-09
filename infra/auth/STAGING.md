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

For healthy existing staging, skip to [Run the execution proof](#run-the-execution-proof).
The bootstrap below is for empty staging stacks; omitting a broker from a live
stack would request its deletion. Run all commands from the Marin checkout.

Terminal A owns the applies:

```bash
export KUBECONFIG=/home/power/.kube/coreweave-iris
export STAGING_PRIVATE=/home/power/.local/state/marin-auth-staging
umask 077
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" prepare
pulumi -C infra/postgres preview --stack us-west4-staging --diff
pulumi -C infra/postgres up --stack us-west4-staging
```

While that first apply waits for role credentials, terminal B provisions them:

```bash
export KUBECONFIG=/home/power/.kube/coreweave-iris
export STAGING_PRIVATE=/home/power/.local/state/marin-auth-staging
kubectl --context marin_US-WEST-04A wait --for=create namespace/marin-auth-db-staging --timeout=120s
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" install-database-secrets
```

After the database apply completes, terminal A copies the complete checked-in
[configuration and broker policy](Pulumi.staging.yaml), then temporarily omits
the broker until the foundation is initialized:

```bash
cp infra/auth/Pulumi.staging.yaml "$STAGING_PRIVATE/staging.yaml"
pulumi -C infra/auth config rm broker_image --stack staging --config-file "$STAGING_PRIVATE/staging.yaml"
pulumi -C infra/auth config set openfga_store_id 00000000000000000000000000 \
  --stack staging --config-file "$STAGING_PRIVATE/staging.yaml"
pulumi -C infra/auth preview --stack staging --config-file "$STAGING_PRIVATE/staging.yaml" --diff
pulumi -C infra/auth up --stack staging --config-file "$STAGING_PRIVATE/staging.yaml"
```

That first auth apply waits for external Secrets. While it runs, terminal B
installs their values outside Pulumi:

```bash
export KUBECONFIG=/home/power/.kube/coreweave-iris
export STAGING_PRIVATE=/home/power/.local/state/marin-auth-staging
kubectl --context marin_US-WEST-04A wait --for=create namespace/marin-auth-staging --timeout=120s
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" install-secrets
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" install-database-ca
```

OpenBao, OpenFGA and the broker receive separate leaf keys and service SANs
signed by the private staging CA. Repeating `prepare` preserves deployment
passwords and certificates. Files are mode 0600 in a mode-0700 directory outside
Git; never put their values in Pulumi inputs, command arguments or logs.

Start one tunnel per separate terminal and leave each running:

```bash
KUBECONFIG=/home/power/.kube/coreweave-iris kubectl --context marin_US-WEST-04A -n marin-auth-staging port-forward pod/openbao-0 18200:8200
KUBECONFIG=/home/power/.kube/coreweave-iris kubectl --context marin_US-WEST-04A -n marin-auth-staging port-forward pod/openbao-1 18201:8200
KUBECONFIG=/home/power/.kube/coreweave-iris kubectl --context marin_US-WEST-04A -n marin-auth-staging port-forward pod/openbao-2 18202:8200
```

Terminal A initializes/unseals the replicas and installs the broker token:

```bash
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" initialize \
  --bao-address https://localhost:18200 --bao-address https://localhost:18201 --bao-address https://localhost:18202
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" install-secrets
```

Start this operator tunnel in another terminal:

```bash
KUBECONFIG=/home/power/.kube/coreweave-iris kubectl --context marin_US-WEST-04A -n marin-auth-staging port-forward deployment/openfga 18080:8080
```

In terminal A publish policy, supply the initial bounded canary credential,
and substitute only this deployment's public metadata into the complete policy:

```bash
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" publish-policy
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" issue-proof --ttl 900 \
  > "$STAGING_PRIVATE/proof-current.json"
uv run --no-sync python - "$STAGING_PRIVATE" <<'PY'
import json
import sys
from pathlib import Path
import yaml
root = Path(sys.argv[1])
path = root / "staging.yaml"
config = yaml.safe_load(path.read_text())
values = config["config"]
standing = json.loads((root / "openfga.json").read_text())
proof = json.loads((root / "proof-current.json").read_text())
policy = values["marin-auth:broker_policy"]
identities = [json.loads((root / name).read_text()) for name in ("identity.json", "iris-controller.json")]
policy["subjects"] = {item["subject"]: item["principal"] for item in identities}
policy["openfga"].update({key: standing[key] for key in ("store_id", "model_id", "principals")})
values["marin-auth:openfga_store_id"] = standing["store_id"]
values["marin-auth:grant_public_keys"] = [(Path(proof["directory"]) / "grants.pem").read_text()]
template = yaml.safe_load(Path("infra/auth/Pulumi.staging.yaml").read_text())["config"]
values["marin-auth:broker_image"] = template["marin-auth:broker_image"]
path.write_text(yaml.safe_dump(config, sort_keys=False))
PY
pulumi -C infra/auth preview --stack staging --config-file "$STAGING_PRIVATE/staging.yaml" --diff
pulumi -C infra/auth up --stack staging --config-file "$STAGING_PRIVATE/staging.yaml"
```

Keep using that config file for later applies. Its token/URL values remain
external Secret references, and its policy keeps `loom: null` for the local
execution proof. Refresh proof credentials again just before that proof.

Initialization uses one Shamir share with threshold one. Its root token and
unseal share remain in `initialization.json` outside Git and Pulumi. Restarted
Bao replicas need unsealing; repeating `initialize` preserves saved entity IDs.
Independent key custodians and a tested recovery procedure are still required
before production use. The fixed userpass
identities have only `marin-auth-identity`; Google SSO enrollment and production
user provisioning remain unimplemented.

The broker token has a 24-hour renewable TTL. Leaf certificates expire after
14 days, their CA after 30 days. Renew before expiry; `prepare` does not rotate
them. Certificate reloads require replacing/unsealing one Bao replica at a time.
The private issuer does not establish CoreWeave WIF: registration also needs
publicly reachable OIDC discovery and JWKS.

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
`marin.dev/auth-client=true`. Its network permits the broker, OpenBao, the
OpenFGA Check gateway, staging PostgreSQL, approved DNS, and the observed
Kubernetes API addresses.
Tasks run without a ServiceAccount token. The task namespace denies ingress and
egress by default. The sole task exception selects `iris.managed=true`,
`iris.runtime=iris-kubernetes`, and `iris.egress=internet`; it permits DNS and
TCP 8090 to broker pods in the auth namespace. Both namespaces enforce
restricted Pod Security; task images need a numeric non-root user.

The CPU execution proof selects the existing `iris-ci-gvisor-cpu` RuntimeClass,
whose handler and node selector target CoreWeave's CPU gVisor runtime. Other
Iris deployments retain the `gvisor` default unless
`kubernetes_provider.sandbox_runtime_class` is explicitly configured.

This cluster's CPU gVisor path timed out for UDP and TCP DNS through the default
`10.16.0.10` resolver. Direct UDP/TCP queries succeeded through the node-local
DNS Service `10.16.0.30` and its observed Pod addresses `10.0.11.112` and
`10.0.0.245`. The proof therefore sets `dnsPolicy: None`, nameserver
`10.16.0.30`, and searches `marin-auth-task-staging.svc.cluster.local`,
`svc.cluster.local`, and `cluster.local`. Its existing DNS egress rules already
permit those destinations. Other Iris workloads retain their default DNS
configuration; no node runtime or cluster DNS setting is changed.

## Execution proof image

Use the published digest below unless rebuilding is needed. The image keeps
Python/current source under `/opt`, so a task workdir at `/app` does not hide
them. It runs as 1000:1000 and retains PG17, Bao, OpenFGA, Node and Git tools.

Authenticated callback changes are in [Loom PR420](https://github.com/marin-community/loom/pull/420).
The proof source is [53e9ef1](https://github.com/marin-community/loom/commit/53e9ef10850952effaa8ba8ade68f9db35d67cae).
Reuse the matching checkout and shared build cache:

```bash
export AUTH_LOOM_SOURCE=/home/power/.codex/worktrees/loom-shared-auth/loom
export CARGO_TARGET_DIR=/home/power/.cache/cargo-build
test "$(git -C "$AUTH_LOOM_SOURCE" rev-parse HEAD)" = 53e9ef10850952effaa8ba8ade68f9db35d67cae
cargo build --manifest-path "$AUTH_LOOM_SOURCE/Cargo.toml" --locked -p loom -p tapestry
mkdir -p artifacts/auth-execution-tools
install -m 755 "$CARGO_TARGET_DIR/debug/loom" artifacts/auth-execution-tools/loom
install -m 755 "$CARGO_TARGET_DIR/debug/tapestry" artifacts/auth-execution-tools/tapestry
strip --strip-debug artifacts/auth-execution-tools/loom artifacts/auth-execution-tools/tapestry
docker buildx build --network=none --load \
  --build-context execution-tools=artifacts/auth-execution-tools \
  --build-context loom-source="$AUTH_LOOM_SOURCE" \
  --build-arg IRIS_REVISION_DATE="$(git log -1 --format=%cs -- lib/iris)" \
  --file infra/auth/execution/Dockerfile --tag marin-auth:execution-10950 .
```

Run offline imports, tools and native callback checks before publishing a new
image, including with an empty `/app` mount. Pin its reviewed published digest
for driver and task; rebuilding alone does not update the digest below.

## Run the execution proof

The pinned image passed real CPU task completion and cancellation on 2026-11-07.
Both tasks read the approved canary, rejected wider access, and lost authority
before their tokens expired.

Run only while the prior proof is idle. Save its evidence before removing its
driver. Keep the Bao0 localhost:18200 tunnel running; root credentials, unseal
shares and userpass passwords never enter the driver.

```bash
export KUBECONFIG=/home/power/.kube/coreweave-iris
export STAGING_PRIVATE=/home/power/.local/state/marin-auth-staging
export PROOF_IMAGE_DIGEST=ghcr.io/marin-community/iris-worker@sha256:f6bccc1c1d8813d5f84fcad2ab08c45cb2d37eb91021e1dd6a984efe62d1186e
umask 077
kubectl --context marin_US-WEST-04A -n marin-auth-staging delete pod auth-flow-proof-10950 --ignore-not-found --wait=true
uv run --no-sync -m infra.auth.staging --directory "$STAGING_PRIVATE" issue-proof --ttl 900 \
  > "$STAGING_PRIVATE/proof-current.json"
EXPECTED_CANARY_SHA256="$(uv run --no-sync python -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_canary_sha256"])' "$STAGING_PRIVATE/proof-current.json")"
uv run --no-sync -m infra.auth.proof_pod --image "$PROOF_IMAGE_DIGEST" | \
  kubectl --context marin_US-WEST-04A apply -f -
kubectl --context marin_US-WEST-04A -n marin-auth-staging wait --for=condition=Ready pod/auth-flow-proof-10950 --timeout=180s
kubectl --context marin_US-WEST-04A -n marin-auth-staging exec auth-flow-proof-10950 -- \
  python -P -m infra.auth.prove_execution \
  --service-config /etc/marin-auth/policy.yaml \
  --identity-token-file /var/run/marin-auth/proof/launcher.jwt \
  --controller-identity-token-file /var/run/marin-auth/proof/controller.jwt \
  --public-key-file /var/run/marin-auth/proof/grants.pem \
  --task-image "$PROOF_IMAGE_DIGEST" --authority-ca-file /var/run/marin-auth/ca/ca.crt \
  --expected-canary-sha256 "$EXPECTED_CANARY_SHA256" \
  --loom-executable /usr/local/bin/loom --loom-source /opt/loom-source \
  --output-dir /proof/execution
mkdir -p artifacts/auth-proof
kubectl --context marin_US-WEST-04A -n marin-auth-staging exec auth-flow-proof-10950 -- \
  cat /proof/execution/execution-proof.json > artifacts/auth-proof/execution-proof.json
```

`issue-proof` makes a fresh private directory with mode 0700 and files with mode 0600, checks the
canary identity is denied other paths, waits for broker Secret mounts to
refresh, then mints launcher/controller JWTs. It installs only
`task-canary-token` and `auth-proof-identities`, and prints paths, expected hash
and expiry metadata. The native token lasts at most 900 seconds; OIDC JWTs last
10 minutes. Reissuing changes the canary value, making old hash metadata stale.

The driver's home/output are under `/proof`; `/proof/execution` must not exist
before a run. Each isolated Loom gets a distinct private callback credential,
shared only with its local lifecycle broker. Deployed brokers keep `loom: null`
so they do not query these runs through a different Loom instance.

## Recovery drills

Pause synthetic auth writes until both logical backups and digest comparisons
finish. The driver mounts the grants SQLAlchemy URL at
`/var/run/marin-auth/database/url` and the verifying CA at
`/var/run/marin-auth/ca/ca.crt`. Open a shell with `kubectl --context marin_US-WEST-04A -n marin-auth-staging exec -it auth-flow-proof-10950 -- sh`. Inside it, run:

```bash
mkdir -p /proof/recovery
chmod 0700 /proof/recovery
python -P -m infra.auth.recover_staging --directory /proof/recovery database \
  --url-file /var/run/marin-auth/database/url \
  --ca-file /var/run/marin-auth/ca/ca.crt --postgres-bin /usr/lib/postgresql/17/bin
```

The chmod removes the inherited fsGroup/setgid bit; keep the permission guard
unchanged. The helper restores only to `grants-recovery` or `openfga-recovery`,
compares table row counts/digests and never restores over the source. Use a
fresh private output directory for a second drill; dump files are created once.

The driver cannot read auth-namespace Secrets through the API. From the operator
terminal, copy the staging `openfga-database` Secret's `uri` through captured
stdin into a mode-0600 file without displaying it:

```bash
uv run --no-sync python - <<'PY'
import base64
import json
import subprocess
base = ["kubectl", "--context", "marin_US-WEST-04A", "-n", "marin-auth-staging"]
secret = json.loads(subprocess.check_output([*base, "get", "secret", "openfga-database", "-o", "json"]))
uri = base64.b64decode(secret["data"]["uri"])
writer = "import os,sys; fd=os.open('/proof/recovery/openfga.url',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); os.write(fd,sys.stdin.buffer.read()); os.close(fd)"
subprocess.run([*base, "exec", "-i", "auth-flow-proof-10950", "--", "python", "-P", "-c", writer], input=uri, stdout=subprocess.DEVNULL, check=True)
PY
```

Then inside the driver:

```bash
python -P -m infra.auth.recover_staging --directory /proof/recovery database \
  --url-file /proof/recovery/openfga.url \
  --ca-file /var/run/marin-auth/ca/ca.crt --postgres-bin /usr/lib/postgresql/17/bin
```

Save each command's public JSON report. Copy `/proof/recovery/{grants,openfga}.dump`
to persistent private operator storage with mode 0600 and compare SHA256 values
with those reports. Remove copied driver scratch files before replacing it;
backups contain private authorization state.

In the operator terminal, take an OpenBao snapshot through the verified
localhost port-forward. A fresh private directory preserves earlier snapshots:

```bash
SNAPSHOT_PRIVATE="$STAGING_PRIVATE/snapshot-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -m 700 "$SNAPSHOT_PRIVATE"
install -m 600 "$STAGING_PRIVATE/initialization.json" "$STAGING_PRIVATE/identity.json" \
  "$STAGING_PRIVATE/launcher.password" "$STAGING_PRIVATE/ca.crt" "$SNAPSHOT_PRIVATE/"
uv run --no-sync -m infra.auth.recover_staging --directory "$SNAPSHOT_PRIVATE" openbao-snapshot
```

Restore in a separate container with no route to the source Raft peers:

```bash
docker run --rm --network=none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$SNAPSHOT_PRIVATE,dst=/private" \
  --entrypoint python "$PROOF_IMAGE_DIGEST" \
  -P -m infra.auth.recover_staging --directory /private openbao --bao /usr/local/bin/bao
```

These quiescent same-cluster logical restores and isolated Raft recovery do not
establish off-cluster DR, WAL/PITR, concurrent-write, host-loss or storage-loss
guarantees. Check restored revocations before serving authorization; row equality
alone cannot establish that an old backup is safe to reopen.

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
