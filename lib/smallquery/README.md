# smallquery

Distributed SQL on Iris via DataFusion-Ballista. This is the PoC from issue #7266:
stand up a Ballista scheduler and N executors on Iris and run simple distributed
queries over Parquet in GCS. Preemption and durability are out of scope (parent #6761).

Phase 0 findings: `.agents/projects/smallquery/ballista-poc-phase0.md`.

## Layout

- `ballista/gcs-support.patch` — the source patch that adds `gs://` object-store
  support to Ballista. Stock binaries register only `s3`/`http`/`file`; this adds a
  `gs` arm to `CustomObjectStoreRegistry` and enables the `object_store` `gcp` feature.
  It also wires that registry into `ballista-cli`, which otherwise resolves object
  stores with DataFusion's default registry and so cannot infer the schema of a
  `gs://` `CREATE EXTERNAL TABLE` client-side. Credentials resolve from the environment
  (ADC), so on Iris GCP workers there is no key material to manage.
- `docker/Dockerfile` — clones Ballista at a pinned tag, applies the patch, builds
  `ballista-scheduler`, `ballista-executor`, and `ballista-cli`, and ships them in a
  slim runtime image.
- `docker/smoke.sh` + `docker/docker-compose.smoke.yml` — a local smoke that runs a
  scan, a GROUP BY (shuffle), and a join over a local Parquet fixture.
- `src/smallquery/deploy.py` — the Iris launcher.

## Build and publish the image

```bash
# 1. Build (~45 min — a Rust release build of scheduler + executor + cli).
docker buildx build --load -t smallquery-ballista:54.0.0 \
  -f lib/smallquery/docker/Dockerfile lib/smallquery

# 2. Authenticate to GHCR once, with a classic PAT that has write:packages.
echo "$GH_TOKEN" | docker login ghcr.io -u <username> --password-stdin

# 3. Push → ghcr.io/marin-community/smallquery-ballista:54.0.0
iris build push smallquery-ballista:54.0.0 --image-name smallquery-ballista --version 54.0.0
```

If you already ran `smoke.sh`, it built `smallquery-ballista:local` — retag that instead
of rebuilding: `docker tag smallquery-ballista:local smallquery-ballista:54.0.0`, then
push. GCP workers pull the pushed image automatically through the GHCR → Artifact
Registry mirror; no multi-region push is needed. `--platform` defaults to the host arch
(build on/for the worker arch — amd64 for standard GCP CPU workers). Pin
`--build-arg BALLISTA_REF=<tag>` to change the Ballista version.

## Local smoke (no cluster, no GCS)

```bash
lib/smallquery/docker/smoke.sh
```

Builds the image if missing, generates a small Parquet fixture, brings up a scheduler
plus two executors, runs the queries, and checks the results. This proves the patched
binaries execute distributed queries with a real shuffle. `gs://` itself is exercised
on-cluster (it needs the worker service account).

## Deploy on Iris and run a query

```bash
python -m smallquery.deploy \
  --cluster <name> --region us-central2 --workers 4 \
  --image ghcr.io/marin-community/smallquery-ballista:54.0.0 \
  --data gs://<region-local-bucket>/<path>/ \
  --sql "SELECT count(*) FROM t"
```

The launcher submits the scheduler (non-preemptible), reads its routable `host:port`
from the Iris task status, submits the executor pool pointed at it, then runs the query
through an in-cluster `ballista-cli` job and streams the result. It tears the cluster
down on exit unless `--keep`. Pin all jobs to the data's region to avoid cross-region
egress.
