# Internal onboarding

This page is for Marin team members who need access to the shared GCP and
CoreWeave infrastructure. External contributors should start with
[Installation](../tutorials/installation.md) and
[Contributing to Marin](contributing.md).

## Ask a coding agent to verify your setup

Marin's commands and infrastructure change often. Use a coding agent from the
repository checkout for setup and infrastructure tasks. The repository's
`AGENTS.md`, skills, and operational guides give the agent the current
procedures and safety constraints.

Ask the agent:

> Onboard me as an internal Marin developer. Verify my local checkout,
> development dependencies, GitHub access, GCP access, and Iris access. Fix
> local setup problems when safe. Do not print secrets, change shared
> infrastructure, or request accelerator capacity. Ask before submitting a
> remote smoke job. Report what is ready and which access is missing.

The agent should use the repository's `onboard-marin` skill. Keep the agent in
the Marin checkout so it can read current project instructions.

## Access you will need

Ask a Marin infrastructure maintainer for the access your work requires:

- Write access to `marin-community/marin` on GitHub.
- The `Marin Dev` role in the `hai-gcp-models` GCP project.
- Iris IAP access to the shared clusters.
- Access to the `marin-community` Weights & Biases entity.
- CoreWeave object-storage credentials if you need to inspect GPU job outputs.
  Iris authorizes and schedules the GPU job separately.

Create a Hugging Face account and accept the licenses for any gated models or
datasets you need. Keep API keys and tokens in environment variables or a
gitignored `.env` file. Do not commit them.

GCP project access, Iris IAP access, GitHub access, and external-service access
are separate grants. The onboarding agent can identify a missing grant. Ask
your manager or a Marin infrastructure maintainer to route the request.

### Pulumi operators

Running `pulumi up` for the `marin` GCP stack requires elevated access outside
ordinary developer onboarding. An operator needs both project custom roles:

- `projects/hai-gcp-models/roles/marindev`
- `projects/hai-gcp-models/roles/marinPulumiAdmin`

`marinPulumiAdmin` can change resource-scoped IAM policy. Grant it only to
trusted operators of the `marin` stack. A maintainer adds an operator through
the [Pulumi user-grant workflow](https://github.com/marin-community/marin/blob/main/infra/pulumi/README.md#user-grants):
`add-grant` creates a PR with the encrypted principal and role bindings. A
separate reviewer runs `review-grant`, confirms the decrypted grant, merges the
PR, and applies the `marin` stack. Do not store a plaintext email in the public
repository or change the managed IAM bindings with `gcloud`.

The onboarding agent should verify both live role bindings, local Pulumi
tooling, and read access to the Pulumi state. It must not run `pulumi up` as an
access check. CoreWeave stacks also require the kubeconfig credentials described
in the Pulumi infrastructure guide.

## What humans should read

Ask the onboarding agent to execute the installation and access checks. Read
the product and workflow docs that explain the work you will do:

1. The [project README](https://github.com/marin-community/marin) explains
   Marin's purpose and current work.
2. [First Experiment](../tutorials/first-experiment.md) introduces Marin's
   experiment model through a small run. Ask the agent to run it when you are
   ready to download the tutorial data and create local artifacts.
3. [Experiments](../explanations/experiments.md),
   [Lazy artifacts](../explanations/lazy-artifacts.md), and
   [The language modeling pipeline](../explanations/lm-pipeline.md) explain how
   experiment graphs, artifacts, and the standard pipeline fit together.
4. [Contributing to Marin](contributing.md) defines the development, test, and
   pull-request workflow.

The operational references are primarily for coding agents and infrastructure
work. [Iris operations](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md)
covers jobs and shared compute. [Marin infrastructure](https://github.com/marin-community/marin/blob/main/infra/README.md)
routes infrastructure changes to the relevant subsystem guide. Neither is
required reading before ordinary experiment development.

Do not start, stop, or restart a shared cluster as part of onboarding.
