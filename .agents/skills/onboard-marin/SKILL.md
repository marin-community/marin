---
name: onboard-marin
description: Verify or complete a new internal Marin developer's local setup and access to GitHub, GCP, Iris, Weights & Biases, Hugging Face, and optional CoreWeave storage. Use when a team member asks to onboard, validate onboarding, or diagnose missing development access.
---

# Onboard a Marin developer

Establish which parts of the developer environment are ready, fix local setup
when safe, and identify each missing external grant. Keep secrets out of
command output and the final report.

## Sources

Use these as the current sources of truth:

- Local installation: `docs/tutorials/installation.md`.
- Development workflow: `docs/dev-guide/contributing.md`.
- Iris authentication and access checks: `lib/iris/OPS.md` and the `use-iris`
  skill.
- CoreWeave credentials and routing: `docs/tutorials/cloud-gpu.md`.

Do not copy procedures from `infra/README.md`; it is an infrastructure index.

## Verify

Start with read-only checks. Report each area as ready, missing access, missing
local setup, or not checked.

1. Confirm the command is running from a Marin checkout and inspect the working
   tree without changing user work.
2. Check the required local tools and Python version. Check dependency and
   pre-commit setup against the installation and contributing guides. Install
   or repair local dependencies when the user's onboarding request authorizes
   it.
3. Check GitHub authentication and repository push permission without pushing a
   branch.
4. Check the active GCP account and project, Application Default Credentials,
   and read access to `gs://marin-us-central2`. Do not print credential
   contents.
5. Check Iris authentication and read-only cluster status. Use `iris login`
   only for an interactive human session; let the browser or headless login flow
   request the human's input.
6. Check whether `WANDB_API_KEY` and `HF_TOKEN` are present without printing
   their values. When useful, perform a read-only identity check with the
   service's CLI and confirm access to the `marin-community` W&B entity. Do not
   persist a token outside the user's chosen credential store.
7. Check CoreWeave object-storage access only when the user needs to inspect GPU
   job outputs. Do not treat storage credentials as proof of GPU scheduling
   access; Iris controls compute access separately.

Distinguish local configuration failures from permissions that a maintainer
must grant. GCP project access, Iris IAP access, GitHub access, Weights & Biases,
Hugging Face, and CoreWeave storage are independent.

## Smoke tests

Run local import or CPU checks when they are cheap and do not download a large
dataset. A remote Iris job changes shared state: submit one only when the user
explicitly authorizes the smoke job. Keep it CPU-only, small, and bounded. Never
request a GPU or TPU during onboarding.

Do not start, stop, restart, deploy, or otherwise mutate a shared cluster.

## Report

Give the user a compact checklist of verified capabilities and remaining
actions. Include the failing command category and error summary without secret
values. Distinguish grants the user can request from local fixes. Do not edit
IAM data or file a grant request unless the user asks.
