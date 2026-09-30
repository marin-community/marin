# Task image registry

Verified 2026-09-19 UTC. The existing authenticated registry is operational at
`envreg.208261-marin-gpu.coreweave.app`; no new public IPv4 was needed.
The user authorized retaining/using registry infrastructure for this pipeline.

## Endpoint, storage and ownership

- CoreWeave cluster `cw-us-east-02a`, Kubernetes context `marin-gpu_US-EAST-02A`,
  namespace `envreg`, Deployment/Service/Ingress `registry`.
- Existing Traefik ingress IP `166.19.9.80`; TLS certificate `registry-tls`,
  managed by cert-manager with issuer `letsencrypt-http01-prod`.
- Distribution `registry:3.0.0`; observed running image digest is retained in
  [health evidence](audits/registry_health_001.json).
- Persistent blobs/manifests live in `s3://marin-us-east-02a/registry/`.
  Kubernetes Secret `cw-s3` provides the existing storage credentials.
- This is a shared registry. Existing `envgen/*` images and the existing Daytona
  registry registration belong to the build_envs work and were preserved.
  This pipeline owns its new publisher identity and `capability-infra/oci-smoke`
  fixture. Reserve `capability-env-gen/*` for reviewed task publications.
- This remains a manually managed single-replica service. No HA, GC schedule,
  registry quotas, repository ACLs, backup restore test or Pulumi ownership was
  added. S3 persistence survives pod replacement; namespace deletion would still
  remove service configuration and secret references.

[registry-recovery.json](infra/registry-recovery.json) captures the current
credentials-free ConfigMap, Deployment, Service and Ingress. It references
existing secrets and namespace; it does not recreate their values. The older
`build_envs/envgen/probes/registry_k8s.yaml` is stale: it specifies 2.8.3, whereas
3.0.0 plus mounted `forcepathstyle: false` configuration is the working CW S3
combination. Do not reapply that older manifest.

## Credentials

Dedicated username: `capability-publisher`.

Durable JSON credential (`registry`, `user`, `password`):
`projects/hai-gcp-models/secrets/capability-registry-publisher/versions/1`.
GCP Secret Manager IAM controls retrieval; use the existing authenticated GCP
credential mechanism. No IAM grants were added. Fetch values only inside a
trusted controller/publisher, via a captured subprocess or secret injection.
For example, the secret retrieval command is:

```sh
gcloud secrets versions access 1 --project=hai-gcp-models --secret=capability-registry-publisher
```

Capture its stdout in memory or a mode-0600 temporary file; do not run that
command into task logs or shell history containing the returned value. Do not
pass the password as a CLI argument. Never inject this credential into a GLM
task-builder environment, candidate sandbox, public task bundle or verifier
artifact. A trusted isolated publication worker may receive it temporarily.

The account is **registry-wide**: htpasswd authenticates identities but has no
repository authorization. Its effective capability includes pull/push and the
server's enabled deletion API. This is a dedicated identity, not a scoped
repository token. The prior htpasswd entries were preserved, and the previous
account was checked after the rolling restart (`GET /v2/` returned 200).

Account retirement requires removing only its htpasswd line and restarting the
registry, then disabling its Secret Manager version. Do not rotate the previous
shared account or delete `registry-htpasswd`, `cw-s3`, `registry-tls`, the
namespace, or the S3 prefix as task cleanup.

## Verified transport and image identity

The benign fixture contains only `registry-probe.txt`; it has no executable and
is not a task environment. Canonical reference:

```text
envreg.208261-marin-gpu.coreweave.app/capability-infra/oci-smoke@sha256:74884d0a370cac94d38ea179dd58ffb7bfd5b9e22bef9ea24aabf3804af2e5a3
```

[probe_registry.py](../scripts/probe_registry.py) ran as a short-lived CoreWeave
Kubernetes Job. It generated the fixture there, authenticated the push, matched
the locally computed manifest hash against both PUT and GET
`Docker-Content-Digest`, read config/layer blobs back and required byte equality,
and checked the uncompressed layer digest in the image config. Anonymous registry
and manifest requests returned 401. The script rejects upload locations outside
the registry and removes Authorization on cross-origin blob redirects. It also
rejects non-HTTPS redirects. Credentials reached the job through a temporary
Secret volume, never command arguments or the script ConfigMap.

Evidence: [initial transport receipt](audits/registry_transport_001.json),
[redirect-observed transport receipt](audits/registry_transport_002.json), and
[cleanup receipt](audits/registry_probe_cleanup_001.json).
The harmless OCI fixture remains available by digest. Temporary Jobs, ConfigMap
and credential Secret were removed. No Daytona sandbox or snapshot was created.

This proves registry transport and durable credential delivery. It does not prove
a generated task's portability, task privacy boundary, runtime behavior, resource
limits, or current Daytona pull integration with this new account. Existing
Daytona registration continues using the preserved prior account.

## Production publication

The maintained publisher binds frozen, reviewed image inputs. The older
`push_oci.py` prototype suppresses decompression exceptions, does not make
`raw_sha_ok` a success condition, and lacks full readback validation. Do not treat
its `ok` field as a publication gate. Require compressed byte count/hash,
successful complete decompression and diff_id, exact image config provenance,
computed manifest digest plus registry readback, and credential-safe redirects.
Review public/private file boundaries before capture; then pull the published
digest into a fresh environment and rerun the task gates. The initial registry
setup published only benign fixtures; the later c32 execution below published
the first generated images.

The initial integrity component is now
[`oci_artifact.py`](../capability_pipeline/oci_artifact.py). It checks exact
compressed length/hash, gzip CRC and completion, a bounded uncompressed size and
diff_id, and rejects trailing data or concatenated gzip members. OCI metadata is
deterministic and preserves explicitly supplied source configuration, including
User, Entrypoint and Cmd; missing provenance fields are errors. Thirteen focused
tests cover corruption, truncation, expansion limits, configuration fidelity and
readback mismatch. These are controller-level tests on synthetic bytes, not
execution of generated task code.

[`oci_registry.py`](../capability_pipeline/oci_registry.py) now adds bounded-memory
layer upload/readback, verifies every blob before publishing a manifest by digest,
and verifies the manifest response header and actual readback bytes. It does not
overwrite a tag. Authenticated upload locations must stay on the registry origin;
read redirects must remain HTTPS and lose Authorization when crossing origins.
Transport errors omit provider bodies and signed URLs. Controller tests cover
corrupt source bytes, readback corruption/size errors, wrong returned digests,
foreign upload locations, credential redaction and deterministic publication.
The protocol follows the [OCI Distribution specification](https://github.com/opencontainers/distribution-spec/blob/main/spec.md).

The maintained library passed a fresh CoreWeave job using a benign 3 MiB fixture,
recorded in [the streaming receipt](audits/registry_streaming_transport_001.json).
The fixture is deliberately not an executable image. Its temporary credential
Secret and source ConfigMap were isolated from all builders and task sandboxes.
The [launch record](audits/registry_streaming_launch_001.json) binds the exact
publisher sources; [cleanup](audits/registry_streaming_cleanup_001.json) records
removal of the job and temporary resources.

`rootfs_review.py` and `image_publication.py` additionally bind tar-member and
task privacy checks to reviewed source files and capture receipts. The c32
candidate and private verifier passed these checks and were published by digest
with complete readback. Exact references, receipt hashes and temporary publisher
resource cleanup are recorded in
[the execution audit](audits/c32_image_publication_execution_002.json).

The first cold reconstructions exposed a Daytona launch-metadata issue: a
FROM-only recipe imports the image but starts `daytona sleep infinity`, losing
the OCI entrypoint. A fresh original-source snapshot starts PostgreSQL normally.
The [provider diagnosis](audits/c32_cold_pull_provider_recovery_001.json) retains
both observations and verified diagnostic cleanup. The credential-free
`image_runtime_metadata.py` catalog verifies raw manifest/config identities and
restates the exact ENTRYPOINT/CMD in the provider recipe. Both images then passed
two fresh network-blocked boots, automatic PostgreSQL readiness, reviewed file
hashes, database/workspace mutation reset and registry unreachability. All four
probe sandboxes were deleted with absence verified. The
[cold-pull audit](audits/c32_cold_pull_execution_003.json) binds those receipts.
Publication and cold-boot validation are complete. The reviewed TaskSpec pointer
migration also restored and launched both runtime environments. Its first authored
oracle then failed because replay ignored the control's declared workspace and
submitted no output file; the private verifier correctly returned zero. The
[migration execution audit](audits/c32_image_migration_revalidation_execution_001.json)
retains the failure and launch evidence. Controller workspace replay and fresh full
task validation remain pending; no task acceptance follows from image publication.

## Generic image helpers under integration

The generic path now accepts one or two explicit image roles and a builder's
`task/image-capture-request.json`. It freezes exact source files and capture-tool
hashes before a fresh GLM5.3 review. On a trusted remote controller with the normal
staged OMP configuration, that review can be run with:

```bash
uv run --frozen scripts/review_generic_task_image.py \
  --item "$ITEM_ROOT" --capture-tools "$STAGE/daytona-tools/capture-tools" \
  --omp-config "$STAGE/omp-research-overlay.yml" --output "$FRESH_REVIEW_DIR"
```

The CLI prints the frozen plan/workspace and, only after a completed approving
review, the controller's approval receipt. Capture, publication, and cold-pull
helpers independently validate the retained input packet and raw GLM verdict;
a summary label alone cannot approve publication. The corresponding entry points
are `capture_generic_task_image.py`, `publish_generic_task_image.py`, and
`probe_generic_task_image.py` under `scripts/`; use their `--help` for arguments.
The publisher still requires a separate trusted worker with isolated registry
credentials. Its CLI alone does not establish that isolation.

These helpers have controller tests, and normal synthesis invokes the image
pipeline after basic bundle validation and before lowering. After isolated
publication and cold-pull evidence, the controller validates the exact pointer
migration before continuing into the ordinary task gates. Pinned public base
images continue through the existing path without recapture.

## Automatic publication (publisher service)

Publication needs no operator step. The construction controller (object-store
credentials only) and a long-running trusted publisher (the only holder of the
registry credential) exchange data through an S3 queue,
`s3://marin-us-east-02a/users/muchanem/capability-pipeline/publication/`
(`CAPABILITY_PUBLICATION_QUEUE` overrides; `off` restores the manual hold):

```text
requests/<packet_sha>/handoff.tar.gz          credential-free handoff, keyed by its SHA-256
requests/<packet_sha>/resubmit-<n>.json       controller requests attempt n after a transient failure
returns/<packet_sha>/publication-<role>.json  receipts (published only)
returns/<packet_sha>/result.json              published | rejected | transient_failure (written last)
returns/<packet_sha>/started-<n>.json         publisher crash accounting
service/heartbeat.json                        rewritten every publisher loop (~15 s)
```

When `process_image_construction` reaches pending publication,
[`publication_exchange.py`](../capability_pipeline/publication_exchange.py)
freezes a deterministic handoff from the live attempt (not from `status.json`),
retains it at `<run>/image-publication/<item>/<attempt>/`, uploads it if absent,
and on later passes imports returned receipts through
`import_publication_for_item` (the same exactness checks as the manual import),
then continues to cold pull and migration in the same call. Return contract:

| outcome | returned dict (legacy `plan_path`/`capture_paths`/`publication_paths` keys kept) |
| --- | --- |
| waiting | `state=pending_publication, retryable=True, reason=awaiting_publisher` (or `awaiting_publisher_heartbeat_stale`), `packet_sha, submitted_at, backoff_seconds` (60..600, grows with wait), `publisher_heartbeat_age_seconds` |
| transient, resubmitting | `pending_publication, retryable=True, reason=publisher_transient_failure_backoff` (waits 5/10/20/40 min) or `..._resubmitted`, `publication_attempt` |
| transient, exhausted (5 attempts) | `pending_publication, retryable=False, reason=publisher_transient_retries_exhausted` |
| rejected | `state=failed_terminal, failure_stage=image_publication, retryable=False, reason, rejection_class, builder_repairable, issues` |
| object store unreachable | `pending_publication, retryable=True, reason=queue_unreachable, backoff_seconds=120` |
| inconsistent return | `pending_publication, retryable=True, reason=publisher_result_invalid, backoff_seconds=300` |
| queue lost the packet and it no longer rebuilds byte-identically | `pending_publication, retryable=True, reason=handoff_rebuilt_next_pass, backoff_seconds=30` |
| bug holds | `pending_publication, retryable=False, reason=handoff_export_failed` / `publication_receipt_import_failed` / `publication_import_incomplete` |
| imported | falls through: `pending_cold_pull`, `pending_migration` or `ready` as before |

`builder_repairable` is true only for `rootfs_review_failed` (for example an
undeclared file under `/opt/task` in a candidate image), which a new
`image-capture-request.json` can fix.

Publisher-side limits are transient, never rejections: a packet over the
capacity bounds (8 GiB compressed, 6 GiB expanded; real packets reach 1.5 GB
because they carry the retained review input) or a capture object outside
`--capture-object-prefix` exhausts its bounded attempts and then holds for an
operator, who raises the bound and runs `publisher_kit.py requeue --packet <sha>`.

The publisher ([`run_image_publisher_service.py`](../scripts/run_image_publisher_service.py))
validates every packet with the one-off path (`open_packet`,
`publish_packet_role`, `publish_role`), reads layers with pinned code confined to
`s3://marin-us-east-02a/users/muchanem/envrootfs/` (never the packet's staged
`cw_presign`), and never imports packet bytes. Deployment, health and teardown:
`ops/publisher/deploy.sh`, `ops/publisher/health.sh` (exit 1 alarms on a stale,
absent or degraded heartbeat), `ops/publisher/teardown.sh`.
