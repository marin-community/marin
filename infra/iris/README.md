# Iris controller Pulumi deployment

This application has one stack per production Iris cluster. The stack owns a
restart-only activation resource; it does not own the GCE VM, Kubernetes
controller objects, SQLite storage, or deletion.

`Pulumi.yaml` defines the Iris Pulumi project, Python entry point, and shared
state backend. Each `Pulumi.<cluster>.yaml` file supplies that project's stack
configuration for one cluster; stack files do not replace the project manifest.

Run deployments through the shared wrapper from the repository root:

```bash
uv sync --all-packages --extra deploy
uv run marin-deploy iris rollout <cluster>
uv run marin-deploy iris rollback <cluster>
```

The rollout wrapper resolves operator-side secrets and task environment variables,
takes a checkpoint, builds and pins the controller images, and invokes `pulumi up`.
CoreWeave S3-backed clusters require `CW_KEY_ID` and `CW_KEY_SECRET`; missing
variables fail this preflight before the checkpoint or image build. When a prior
deployment is recorded, the wrapper first writes a `pending` record with the paired rollback state.
Pulumi records the cluster name, image references, activation ID, and digest.
The dynamic provider loads the cluster configuration and resolves runtime values
in the operator process. It avoids the stdout and stderr state capture performed
by `pulumi_command.local.Command`; secret values do not enter Pulumi config,
inputs, outputs, or state.

GCE activation updates `startup-script` metadata and runs the script over SSH on
the existing VM. CoreWeave activation reconciles the existing controller through
the configured Kubernetes API. Resource deletion is a no-op, so removing the
stack never stops a controller or deletes its state.

If activation fails, the wrapper writes `rollback_requested` with the previous
image and the new pre-deploy checkpoint, then applies that pair through Pulumi.
The failed rollout exits nonzero after successful recovery. An explicit
`marin-deploy iris rollback` uses the previous image and checkpoint in the committed
rollout record. A Pulumi error before the resource reaches its mutation boundary
restores the prior rollout record without requesting a checkpoint rollback.

Direct `pulumi up` is intentionally unavailable: every update requires
temporary image and activation configuration produced after the checkpoint and
image build. The checked-in stack settings record the KMS provider used by
subsequent operations. Pulumi still selects that provider while creating the
backend stack, so pass the same provider to `stack init` before the first
wrapper-driven deployment:

```bash
cd infra/iris
pulumi stack init <cluster> \
  --secrets-provider="gcpkms://projects/hai-gcp-models/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key"
```
