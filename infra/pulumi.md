# Pulumi patterns across `infra/`

`infra/` hosts independent Pulumi projects for infrastructure, application
deploys, and SaaS resources. They share a state backend and Python environment,
but their resources have different owners and lifecycles.

Add a stack when the same Pulumi program manages another instance of the same
resource graph, with the same ownership and lifecycle. Create a directory and
project when the resource graph, owner, or deployment lifecycle differs. Each
project has its own `Pulumi.yaml` `name:`. For example, cluster stacks share the
`infra/pulumi` program, while Grafana has a separate project because it has its
own image and deployment lifecycle.

## The three patterns

### 1. Infrastructure (`infra/pulumi`, project `marin-iac`)

This project provisions cluster prerequisites and shared GCP infrastructure.
Cluster-scoped resources use one stack per cluster, with configuration in
`Pulumi.<cluster>.yaml`. Shared GCP resources, including IAM grants, belong to
the `marin` stack and its `Pulumi.marin.yaml` configuration. Do not create
another infrastructure stack or permissions project for shared GCP resources.

Resources often exist before a cluster or GCP resource is brought under Pulumi.
For a one-time adoption, set `marin-iac:import=true`; the program imports the
live resources instead of recreating them.

`infra/pulumi/src/iac` contains reusable components imported by other projects.
For example, application projects use `iac.gcp.cloud_run.CloudRunService`
instead of declaring Cloud Run resources independently. Add a component here
when multiple projects need the same resource lifecycle.

### 2. Application deploys

Each deployed service lives in a separate directory and project. Prefer a thin
leaf that configures a shared `iac` component. For example, `infra/grafana`
constructs `iac.gcp.cloud_run.CloudRunService` with Grafana's image, runtime
configuration, and dependencies.

Use a bespoke program only when a shared component cannot express the
deployment. `infra/loom` is one example: it builds an image, writes the image
digest and runtime configuration to a durable VM, and runs its own readiness
activation step.

### 3. SaaS resource declarations

SaaS projects record resources outside GCP. For example,
`infra/pulumi/github` declares GitHub Actions secret metadata as external
resources and audits the live state without owning secret values. Use
lookup/external resources when Pulumi should review the declaration but another
system owns the resource contents.

## Shared conventions

All projects share these conventions:

- **State backend.** GCS, `gs://marin-iac-state/`, committed as `backend.url` in
  every project's `Pulumi.yaml`. The CLI and CI read the project setting
  directly; do not add `pulumi login` steps or CI `cloud-url` overrides.
- **Secrets provider.** GCP KMS
  (`gcpkms://…/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key`), configured
  per-stack in `Pulumi.<stack>.yaml`.
- **Shared virtualenv.** Leaf projects pin `runtime.options.virtualenv` to the
  repository `.venv` so Pulumi reuses the repository dependency closure.
  `infra/pulumi` uses the plain `python` runtime and runs under the repository
  uv environment.
- **Cross-project data.** Read another project's outputs with
  `pulumi.StackReference`; for example, `infra/grafana` reads outputs from the
  Cloud SQL stack. Do not duplicate outputs in another project's config.
- **Secret values never enter state.** Keep values out of Pulumi config,
  resource arguments, and outputs. Pass Secret Manager references or external
  resource metadata instead. The runtime mechanism depends on the project:
  Loom passes a pinned version identifier to its VM, Cloud Run mounts the
  version named by `SecretEnv` (use an explicit numeric version when a deploy
  must remain pinned across rotations), Cloud SQL creates empty secret
  containers whose values are added out of band, and the GitHub project records
  recovery metadata without dereferencing it.

## Picking a pattern for new work

1. Add cluster-scoped or shared GCP infrastructure to `infra/pulumi`. Put
   shared GCP configuration in `Pulumi.marin.yaml`.
2. Give a deployed service its own directory and project. Configure a shared
   `iac` component unless the service needs a different deployment lifecycle.
3. Put third-party resources in a SaaS project that matches their provider and
   ownership model.
