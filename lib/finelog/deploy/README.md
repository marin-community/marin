# Deploying finelog

The `finelog-server` is a structured-log store + push/fetch RPC server
listening on TCP port 10001. It writes Parquet segments to a local cache
and (optionally) archives them to a remote `fsspec` URI such as
`gs://bucket/path`.

## Security model

The server is **unauthenticated by design**. Restrict access at the
network layer:

- Kubernetes: NetworkPolicy + ClusterIP-only Service. No Ingress.
- Cloud Run: `ingress: internal`. No public exposure.
- Bare Docker: bind only to a private network, or front with a firewall.

Do not add an authentication wrapper — clients (iris workers) do not
speak any auth protocol.

## Build the image

The Docker build context must be the marin repo root:

```bash
docker build -f lib/finelog/deploy/Dockerfile -t finelog:dev /home/power/code/marin
```

## Deploy targets

| Target     | When to use                                         | Manifest                  |
|------------|-----------------------------------------------------|---------------------------|
| Docker     | Local dev, single-host runs.                        | `Dockerfile`              |
| Kubernetes | **Production.** Single-replica Deployment + PVC.    | `k8s/`                    |
| Cloud Run  | Short-lived demos only — not recommended for prod.  | `gcp/cloudrun.yaml`       |

GKE is the production target. Cloud Run is supported but compromised by
the single-writer constraint; see `gcp/README.md` for the trade-off.

### Docker (local)

```bash
docker run --rm \
  -p 10001:10001 \
  -v finelog-cache:/var/cache/finelog \
  -e FINELOG_REMOTE_DIR="" \
  finelog:dev
```

### Kubernetes

```bash
kubectl apply -f lib/finelog/deploy/k8s/  # applies 01-pvc, 02-deployment, 03-service in order
```

Set `FINELOG_REMOTE_DIR` in the Deployment env to a `gs://` URI to
enable archival offload. Iris references the resulting Service as
`k8s://log-server` in its endpoints config.

### Cloud Run

See `gcp/README.md`.
