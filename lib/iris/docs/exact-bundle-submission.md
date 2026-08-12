# Exact workspace bundle submission

`IrisClient.remote(..., exact_bundle=...)` submits caller-provided workspace ZIP
bytes without recreating the archive from a directory. Construct
`ExactWorkspaceBundle` with keyword arguments for the reviewed bytes and their
lowercase SHA-256 content ID:

```python
from iris.client import ExactWorkspaceBundle, IrisClient

client = IrisClient.remote(
    controller_url,
    exact_bundle=ExactWorkspaceBundle(blob=reviewed_zip, bundle_id=reviewed_sha256),
)
job = client.submit(
    entrypoint,
    name,
    resources,
    task_image="registry.example/task@sha256:<64-lowercase-hex>",
    bundle_init_image="registry.example/iris-init@sha256:<64-lowercase-hex>",
)
```

The exact bundle source is mutually exclusive with `workspace` and inherited
`bundle_id`. The controller requires both exact-upload members, recomputes the
content ID, rejects legacy bundle fields on the same request, and rejects
unknown request fields. Exact-upload clients and controllers must therefore be
upgraded together. Repeating the same bytes and declared content ID is a safe
content-addressed write; a different byte sequence cannot reuse the ID.

`bundle_init_image` is independent of the task image. When present, the
controller accepts only an OCI reference ending in
`@sha256:<64-lowercase-hex>`, persists it with the job, and places it on the
Kubernetes bundle/workdir staging container. Tasks receive the resolved value
for child submission: a child inherits it when its argument is `None`, while an
explicit child value overrides it. An omitted value on a root job preserves the
cluster default behavior.

Constructing the client and bundle does not upload data. Calling `submit`
uploads the exact ZIP to the controller's configured bundle store and creates a
job. A controller deployment or restart is a separate operational change.
Before either action, obtain explicit authorization naming the reviewed bundle
SHA-256, destination cluster and store, task and init image digests, resource
request, retry limits, environment disclosure, and job count. Source review or
a local test run does not authorize deployment, upload, or submission.
