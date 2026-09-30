# Custom task-image request contract

Use an existing image when its reference is already a portable, pinned OCI
digest such as `registry.example/team/base@sha256:<64 lowercase hex>`. Do not
create an image-capture request merely to recapture a public base image.
An `envgen.daytona/<snapshot>/dockerfile@sha256:<hash>` pointer identifies a
Daytona recipe, not an OCI manifest; it still requires capture and publication.
Any image named in `image-capture-request.json` follows that capture path even
if its authored pointer happens to have OCI digest syntax.

If the task requires a custom Daytona snapshot, write
`task/image-capture-request.json`. This is a request for trusted controller
work, not an approval or an OCI image. It must remain
`"state": "review_required"`. The controller freezes the source files,
supplies the capture-tool hashes, obtains a fresh independent GLM-5.3 review,
captures the reviewed snapshot, inspects the rootfs, and may later publish an
OCI digest. Do not write `capture_implementation`, an approval receipt, registry
credentials, or a claimed cold-pull result. Never invent a custom-image digest;
retain the observed snapshot pointer until the controller migrates it.

The authored pointer is the exact noncanonical image/snapshot value currently
used by the task. `source_snapshot` is the observed provider mapping for that
pointer: copy its exact `name`, `id`, and `ref` from the provider. A snapshot
reference or an ID suffix is not an OCI digest. The trusted capture worker
checks that the snapshot is active and that its recorded recipe bytes match the
hash-bound `source_recipe` before capture.

Use this complete request shape. Replace every `REPLACE_*` value with an actual
value; every SHA-256 is 64 lowercase hexadecimal characters. Add a second image
object only when the task also has a distinct custom `private_verifier` image.

```json
{
  "schema_version": "capability-generic-image-capture-plan-v1",
  "state": "review_required",
  "registry_host": "envreg.208261-marin-gpu.coreweave.app",
  "source_files": [
    {
      "path": "images/candidate.Dockerfile",
      "sha256": "REPLACE_64_HEX_SOURCE_HASH",
      "visibility": "public"
    },
    {
      "path": "private/checker.py",
      "sha256": "REPLACE_64_HEX_PRIVATE_HASH",
      "visibility": "private"
    },
    {
      "path": "images/verifier.Dockerfile",
      "sha256": "REPLACE_64_HEX_VERIFIER_RECIPE_HASH",
      "visibility": "private"
    }
  ],
  "private_assets": [
    {
      "workspace_path": "private/checker.py",
      "image_path": "/opt/verifier/checker.py",
      "sha256": "REPLACE_64_HEX_PRIVATE_HASH"
    }
  ],
  "images": [
    {
      "role": "candidate",
      "repository": "capability-env-gen/example-candidate",
      "authored_image_pointer": "daytona-snapshot:example-candidate-build",
      "source_snapshot": {
        "name": "example-candidate-build",
        "id": "REPLACE_PROVIDER_SNAPSHOT_ID",
        "ref": "REPLACE_PROVIDER_SNAPSHOT_REF"
      },
      "source_recipe": {
        "path": "images/candidate.Dockerfile",
        "sha256": "REPLACE_64_HEX_SOURCE_HASH"
      },
      "input_files": ["images/candidate.Dockerfile"],
      "architecture": "amd64",
      "operating_system": "linux",
      "image_config": {
        "Env": ["PATH=/usr/local/bin:/usr/bin:/bin"],
        "WorkingDir": "/workspace",
        "User": "root",
        "Entrypoint": null,
        "Cmd": ["/bin/sh"]
      },
      "capture_lifecycle": {
        "ready_command": "test -f /workspace/task/READY",
        "ready_attempts": 10,
        "ready_timeout_seconds": 30,
        "ready_interval_seconds": 2,
        "quiesce_command": null,
        "quiesce_probe": null,
        "exclude_mounts": [],
        "exclude_absent_paths": ["/opt/evaluator", "/opt/verifier", "/var/lib/postgresql/data"]
      },
      "required_ready_hashes": {
        "/workspace/task/READY": "REPLACE_64_HEX_READY_FILE_HASH"
      },
      "resources": {"cpu": 2, "memory": 2, "disk": 10}
    },
    {
      "role": "private_verifier",
      "repository": "capability-env-gen/example-verifier",
      "authored_image_pointer": "daytona-snapshot:example-verifier-build",
      "source_snapshot": {
        "name": "example-verifier-build",
        "id": "REPLACE_PROVIDER_VERIFIER_SNAPSHOT_ID",
        "ref": "REPLACE_PROVIDER_VERIFIER_SNAPSHOT_REF"
      },
      "source_recipe": {
        "path": "images/verifier.Dockerfile",
        "sha256": "REPLACE_64_HEX_VERIFIER_RECIPE_HASH"
      },
      "input_files": ["images/verifier.Dockerfile", "private/checker.py"],
      "architecture": "amd64",
      "operating_system": "linux",
      "image_config": {
        "Env": ["PATH=/usr/local/bin:/usr/bin:/bin"],
        "WorkingDir": "/opt/verifier",
        "User": "root",
        "Entrypoint": null,
        "Cmd": ["/bin/sh"]
      },
      "capture_lifecycle": {
        "ready_command": "test -f /opt/verifier/checker.py",
        "ready_attempts": 10,
        "ready_timeout_seconds": 30,
        "ready_interval_seconds": 2,
        "quiesce_command": null,
        "quiesce_probe": null,
        "exclude_mounts": [],
        "exclude_absent_paths": ["/var/lib/postgresql/data"]
      },
      "required_ready_hashes": {
        "/opt/verifier/checker.py": "REPLACE_64_HEX_PRIVATE_HASH"
      },
      "resources": {"cpu": 2, "memory": 2, "disk": 10}
    }
  ]
}
```

`source_files` is the full material source inventory used by every capture
role. Paths are workspace-relative regular files, have no symlink components,
and are exact-byte hashed. Every `source_recipe.path` and every `input_files`
entry must appear in that inventory. Candidate `input_files` may contain only
`public` sources. List every private source that could become a task-private
asset in `private_assets`, mapping it to its intended absolute private image
path. The candidate rootfs is checked for every listed private image path and
for evaluator, verifier, and credential roots.

Give each requested role its own `repository` under
`capability-env-gen/…`; roles are only `candidate` and `private_verifier`.
There can be at most one request for each role. Record the actual Linux/amd64
configuration that produced the snapshot: explicit environment strings,
working directory, user, entrypoint, and command. Do not invent defaults.

The lifecycle proves the state that can be captured. `ready_command` is run up
to `ready_attempts` with the supplied timeout and interval. If quiescence is
needed, give both a `quiesce_command` and a succeeding `quiesce_probe`; provide
neither only when the image is already quiescent. List paths on distinct mounts
in `exclude_mounts`; list task paths that must be absent in
`exclude_absent_paths`; a path cannot appear in both. Include hashes for every
task file that must exist after readiness. `required_ready_hashes` must be
nonempty even for a package-only image: create a deterministic marker in the
Dockerfile, verify its bytes inside the snapshot, and bind that file's path and
SHA-256. A successful import command alone is not a ready-state hash.
`resources` names the requested
positive integer CPU cores, memory GB, and disk GB for the cold reconstruction,
matching the Daytona SDK units. The sample is 2 cores, 2 GB RAM, and 10 GB disk;
use the actual measured task requirements. The current staged capture tool also
excludes `/var/lib/postgresql/data/*`, so that directory must be explicitly
verified as absent or as a distinct excluded mount. Never omit populated task
data just to satisfy this policy; use a complete capture mechanism instead.

Normal synthesis processes this request after basic bundle validation and before
TaskCompendium lowering. It creates no published image, and it does not establish
task acceptance. A custom image remains a pending publication condition until the
independent review, capture, isolated publication, cold pull, and migration
evidence are present. Leave an unsupported or unverified custom image unresolved
instead of replacing it with a fabricated digest.
