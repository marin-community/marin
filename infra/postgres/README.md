# CloudNativePG authorization prototype

This Pulumi project owns a disposable PostgreSQL experiment on CW
`marin_US-WEST-04A`. It installs CloudNativePG 1.30.1 from Helm chart 0.29.1,
three PostgreSQL 17 instances, and an optional authorization failover Job.
The operator and PostgreSQL images are pinned by digest.

The operator watches only `marin-pg-trial`. Its namespaced Role can manage
database resources there; its cluster permissions read nodes and image catalogs
and update only its two admission configurations. Admission webhooks match only
the trial namespace. Installation still creates cluster-scoped CRDs. Check for
an existing operator and CRDs before using this prototype; it must not adopt or
replace another installation.

PostgreSQL commits require one synchronous standby, with strict durability and
quorum-aware failover enabled. Pods spread across the two observed zones. The
trial tolerates the GPU node's scheduling preference but requests no GPU.
Three instances across two hosts do not provide the failure tolerance of three
independent hosts. All volumes use `shared-vast`; this experiment does not
qualify its persistence, locking or shared-backend failure behavior for production.

The namespace allows internal traffic, DNS to the observed resolvers, and
Kubernetes API access. CW's API Service uses port 443 and translates to 6443;
both ports are permitted only for its configured API addresses. The private
admission service also accepts control-plane requests on 9443. Nothing exposes
PostgreSQL through an ingress or public load balancer.

## Run the experiment

Use the repository environment with `marin-iac[deploy]` installed and the CW
kubeconfig. The stack uses the shared GCS state backend and GCP KMS provider.
Its checked-in addresses are observations from this cluster; verify the API
EndpointSlice, API Service and DNS destinations before applying elsewhere.

```bash
export KUBECONFIG=/home/power/.kube/coreweave-iris
pulumi -C infra/postgres preview --stack us-west4 --diff
pulumi -C infra/postgres up --stack us-west4
kubectl --context marin_US-WEST-04A -n marin-pg-trial get clusters,pods,pvc -o wide
```

CloudNativePG generates the application password and TLS material inside
Kubernetes. Pulumi manages their references without reading their values. The
optional `proof_image` must name a digest-pinned image built from the repository
root with `docker build -f infra/postgres/Dockerfile .`. The image reuses the
service trial's digest-pinned dependency and native-tool image. Setting that config
adds the proof Job and its limited ServiceAccount:

```bash
pulumi -C infra/postgres config set proof_image "$PROOF_IMAGE_DIGEST" --stack us-west4
pulumi -C infra/postgres up --stack us-west4
kubectl --context marin_US-WEST-04A -n marin-pg-trial logs job/postgres-auth-proof
```

The Job mounts the disposable database user's credentials, public CA, and its
limited ServiceAccount token.
Its Kubernetes Role can read/patch `auth-postgres` and delete only its three
named instance pods. It cannot read Secrets or affect another namespace.
OpenFGA, OpenBao and the grant service run inside the proof Job and connect to
the CloudNativePG database over verified TLS. The proof deletes a nested group
membership edge, waits for OpenFGA to acknowledge it, and checks that the denial
survives primary-pod loss. It also checks OpenBao grant-token signing after
recovery, write blocking without synchronous standbys, authorization failure
during a complete database outage, and recovery of an existing grant.

The fault scenarios affect only synthetic data. Fenced instances are unfenced
in `finally` blocks; if the Job is forcibly stopped, clear the trial Cluster's
`cnpg.io/fencedInstances` annotation on the `auth-postgres` CloudNativePG Cluster
resource before inspecting recovery.

## Production gaps

This is an operator and database failover experiment. OpenFGA and OpenBao share
one disposable database owner for the proof; production needs separate
databases and least-privilege identities. The Marin runtime grant ledger still
uses one local SQLite database and is not replicated by CloudNativePG.

Production adoption also needs independent failure domains, qualified storage,
off-cluster backups and WAL archiving, restore drills, monitoring and alerting,
credential/certificate rotation, and management isolation from Iris's broad
cluster permissions. An old database restore can resurrect removed access;
recovery must reconcile revocations before reopening authorization. No backup,
point-in-time recovery, node-loss or storage-loss guarantee follows from a
successful primary-pod failover.

Remove the disposable installation after recording evidence:

```bash
pulumi -C infra/postgres destroy --stack us-west4
kubectl --context marin_US-WEST-04A get namespace marin-pg-trial --ignore-not-found
```

Confirm that the namespace, its PVCs, and only this trial's cluster-scoped
resources are gone. Do not remove CRDs if another PostgreSQL deployment has
started using them.

CW's observed `shared-vast` class uses `Retain`: deleting a namespace and its
PVCs leaves the underlying volumes and data. Record each trial PVC's UID and
bound PV name before destruction. After destruction, inspect the PV's claim
reference and require both the recorded UID and `Released` state to match. For
each verified disposable trial PV, request backend reclamation:

```bash
kubectl --context marin_US-WEST-04A patch pv "$TRIAL_PV" --type=merge \
  -p '{"spec":{"persistentVolumeReclaimPolicy":"Delete"}}'
kubectl --context marin_US-WEST-04A wait --for=delete "pv/$TRIAL_PV" --timeout=120s
```

Never change reclamation policy for a bound volume or an unrelated claim.
Production retention and retirement need their own reviewed recovery procedure.
