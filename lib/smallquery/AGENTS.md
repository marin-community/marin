# smallquery Agent Notes

Distributed SQL on Iris via DataFusion-Ballista. Start with the shared instructions in
`/AGENTS.md`; only smallquery-specific notes are below. Background and the design
rationale are in `.agents/projects/smallquery/ballista-poc-phase0.md` (issue #7266,
parent #6761).

## Structure

- The Ballista binaries are not vendored. `docker/Dockerfile` builds them from a pinned
  upstream tag (`BALLISTA_REF`, default `54.0.0`) with `ballista/gcs-support.patch`
  applied. To bump Ballista, change `BALLISTA_REF` and re-verify the patch applies —
  regenerate it against the new tag if `git apply` fails. The patch touches four files:
  `ballista/core/Cargo.toml` (add the `gcp` object_store feature),
  `ballista/core/src/object_store.rs` (the `gs` registry arm), and
  `ballista-cli/{Cargo.toml,src/main.rs}` (give the cli the same registry so client-side
  `CREATE EXTERNAL TABLE` schema inference over `gs://` works).
- Tasks run bare Ballista binaries with `EnvironmentSpec(setup_scripts=[])` (no uv
  sync); the image is used as-is. There is no Python in the task containers.
- Discovery is launcher-threaded: `deploy.py` reads the scheduler's routable address
  from `Job.status().tasks[].worker_address` (host) + `.ports["scheduler"]` and passes
  it to the executors and the client as argv. Do not add a service registry or a
  Python task wrapper unless the design outgrows this.

## GCS

`gs://` works because the patch adds a `gs` arm to Ballista's
`CustomObjectStoreRegistry` and enables the `object_store` `gcp` feature. Credentials
come from Application Default Credentials — on Iris GCP workers the metadata server
supplies the `iris-worker@` service account (which has `roles/storage.objectAdmin`).
Executors are what open the object store, so the credential must be present in the
executor environment, which it is by default on GCP workers.

## Verification

`docker/smoke.sh` is the local proof (scan + shuffle + join over a local fixture). The
`gs://` path is proven on-cluster with `python -m smallquery.deploy` because it needs
the service account. Keep the smoke green when changing the patch or the Dockerfile.
