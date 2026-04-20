# Log Server Deployment

The Iris log server is a standalone process that owns the log store (DuckDB +
Parquet tiers on disk, GCS archive). Controllers and workers push logs to it
and fetch logs from it over RPC.

There are three deployment modes:

- **In-process (default for tests, CI, local dev):** the controller starts a
  ``LogServiceImpl`` on a free port and advertises it via
  ``/system/log-server``. No extra container required.
- **Sidecar on the controller VM:** the controller VM runs a second Docker
  container on the same host. The controller container receives
  ``IRIS_LOG_SERVICE_ADDRESS=http://localhost:<port>`` from the bootstrap
  script and resolves the sidecar automatically. Enable with
  ``controller.enable_log_server_sidecar: true`` in cluster YAML. See
  [Sidecar mode](#sidecar-mode) below.
- **External (recommended for production):** the log server runs as its own
  container on its own host, and the controller is pointed at it with
  ``--log-service-address``.

## Image

The log server ships as ``ghcr.io/marin-community/iris-log-server:latest``.
Built from ``lib/iris/Dockerfile`` target ``log-server``. Reuses the shared
``deps`` layer plus ``duckdb`` and ``pyarrow`` (``[log-server]`` extras).

Build locally:

```bash
docker buildx build \
    --file lib/iris/Dockerfile \
    --target log-server \
    --tag iris-log-server:dev \
    lib/iris
```

## Run

```bash
docker run --rm \
    -p 10002:10002 \
    -v /var/cache/iris/logs:/var/cache/iris/logs \
    -e IRIS_LOG_SERVER_JWT_KEY=$SIGNING_KEY \
    -e IRIS_LOG_SERVER_AUTH_STRICT=1 \
    iris-log-server:dev \
    --port 10002 \
    --log-dir /var/cache/iris/logs \
    --remote-log-dir gs://your-bucket/iris-logs
```

Arguments:

- ``--port`` — port to bind (default in the image: ``10002``).
- ``--log-dir`` — local directory for tmp + compacted Parquet segments. Must
  be persisted across restarts; DuckDB reads rely on these files.
- ``--remote-log-dir`` — GCS (or any fsspec-supported) URI where compacted
  segments are uploaded for durable archival.
- ``--log-level`` — ``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``.

Environment:

- ``IRIS_LOG_SERVER_JWT_KEY`` — HMAC signing key used to verify JWTs minted
  by the controller. Must match the controller's signing key.
- ``IRIS_LOG_SERVER_AUTH_STRICT`` — set to any non-empty value to reject
  unauthenticated requests. Unset, the server accepts anonymous requests (for
  local dev only).

## Wire the controller

Start the controller with ``--log-service-address`` (or
``IRIS_LOG_SERVICE_ADDRESS``) pointing at the log-server instance:

```bash
iris cluster controller serve \
    --host 0.0.0.0 --port 10000 \
    --log-service-address http://logs.internal:10002
```

When ``--log-service-address`` is set the controller:

- does **not** start an in-process log server;
- advertises the external URL via ``/system/log-server`` so workers resolve
  it through ``controller.list_endpoints(...)``;
- continues to mint and verify JWTs locally.

When ``--log-service-address`` is omitted, the controller falls back to the
in-process log server. This is the default for tests, CI, and local dev.

## Sidecar mode

On a VM-backed controller (``controller.gcp`` or ``controller.manual``), you
can colocate the log server with the controller instead of managing a
separate VM. Cluster YAML:

```yaml
controller:
  image: ghcr.io/marin-community/iris-controller:latest
  enable_log_server_sidecar: true       # opt-in; default false
  gcp:
    zone: us-central1-a
storage:
  remote_state_dir: gs://your-bucket/iris/state
```

The sidecar has no other knobs:

- image: same registry and tag as ``controller.image``, with the last path
  segment ``iris-controller`` swapped to ``iris-log-server``;
- port: ``10002`` (host networking);
- archive URI: ``{storage.remote_state_dir}/logs``.

With the sidecar enabled, the bootstrap script on the controller VM runs a
second ``iris-log-server`` container with
``--network=host --restart=unless-stopped`` and points the controller
container at it via ``IRIS_LOG_SERVICE_ADDRESS=http://localhost:10002`` for
the controller's own pushes.

Workers connect directly to the sidecar rather than tunnelling through the
controller: when ``log_service_address`` is a loopback URL, the controller
advertises ``http://<controller-host>:10002`` via ``/system/log-server`` —
same VM, the sidecar's port, so no dashboard hop. Your controller VM's
firewall must allow port ``10002`` from the same network range that already
reaches the controller's own port.

**Stage 1 caveat — JWT signing key.** The sidecar currently starts without
the controller's JWT signing key, so it runs in null-auth mode and accepts
anonymous pushes. Strict-auth clusters should keep the sidecar disabled
until the standalone ``iris log-server deploy`` flow lands (Stage 2) with
a key handoff mechanism.

## Sharing the JWT signing key

On first start the controller persists a signing key in its database (see
``_get_or_create_signing_key`` in ``lib/iris/src/iris/cluster/controller/auth.py``).
Workers mint JWTs with this key; the log server verifies them with the same
key.

In split deploys there is no automatic key-sync mechanism — the operator
copies the key out of the controller DB and sets ``IRIS_LOG_SERVER_JWT_KEY``
on the log-server container. Both must hold the same value.

If the controller's DB is rebuilt, the key changes and existing log-server
containers must be restarted with the new key.

## Health check

The container ships with a ``HEALTHCHECK`` that GETs ``/`` on the configured
port. The log server replies with ``404`` to unknown routes, which is
sufficient to confirm the ASGI listener is up.
