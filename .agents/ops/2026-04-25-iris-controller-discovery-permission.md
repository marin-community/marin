---
date: 2026-04-25
system: iris
severity: diagnostic-only
resolution: investigating
pr: https://github.com/marin-community/marin/pull/5179
issue: https://github.com/marin-community/marin/issues/5178
---

## TL;DR

- A MoE depth MuP LR sweep was ready to submit through Iris, but job submission
  failed before job creation.
- The active GCP account was `kaiyuew@stanford.edu` on project
  `hai-gcp-models`.
- Iris could not discover the Marin controller because GCP returned
  `GCP API error 403: Required 'compute.instances.list' permission for
  'projects/hai-gcp-models'`.
- No Iris job was created. No cluster or controller state was changed.
- Retrying requires an account with controller VM list permission or an explicit
  `--controller-url` to an existing controller tunnel.

## Original problem report

The user requested that MoE experiment work always submit the run to Iris and
continue until the full MoE procedure is finished. The attempted command was:

```bash
.venv/bin/iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --reserve v5p-8 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.moe.depth_mup_lr_sweep
```

The command failed with:

```text
GCP API error 403: Required 'compute.instances.list' permission for 'projects/hai-gcp-models'
RuntimeError: No controller VM found (label=iris-marin-controller=true, project=hai-gcp-models)
```

## Investigation path

1. The workflow first verified local GitHub auth as `WhenWen`, created issue
   #5178, pushed PR #5179, and prepared the depth MuP sweep module.

2. The Iris preflight confirmed `.venv/bin/iris` was executable and
   `WANDB_API_KEY` was present.

3. `gcloud auth list --filter=status:ACTIVE --format='value(account)'` showed
   the active account was `kaiyuew@stanford.edu`.

4. `iris --config lib/iris/examples/marin.yaml job list` failed during
   controller discovery with a GCP 403 on `compute.instances.list`. This meant
   the CLI could not find the controller VM and could not open its config-based
   tunnel.

5. `lib/iris/OPS.md` confirmed the two normal connection modes:
   `--config=PATH` for auto-tunnels, or `--controller-url=URL` for an existing
   manual tunnel.

6. The required production submission command was attempted anyway. It failed
   before job creation with the same GCP permission error.

7. The dev config, `lib/iris/examples/marin-dev.yaml`, was tried as a fallback.
   It also failed before job creation with the same permission error, but with
   the dev controller label.

8. The environment had no `IRIS_*` or controller URL variables, no listener on
   localhost ports 10000 or 10001, and `curl -sf http://localhost:10000/health`
   returned nothing. There was no existing tunnel to reuse.

## User course corrections

- The user instructed that future GitHub issues and PRs must not use the
  connector and should be submitted as `whenwen`. The MoE guide was updated to
  require local GitHub auth as `whenwen`.
- The user then instructed that future MoE work must always submit the run to
  Iris and continue through the full MoE procedure. The MoE guide was updated
  to make Iris submission mandatory unless a hard blocker prevents it.

## Root cause

The active local GCP account lacked permission to list Compute Engine instances
in `hai-gcp-models`. Iris config-based controller discovery depends on listing
the controller VM by label. Without `compute.instances.list`, the CLI cannot
discover or tunnel to either the production or dev Marin controller.

This was an authentication and project permission blocker, not a code or
scheduler failure. The submission failed before any Iris job was created.

## Fix

No infrastructure fix was applied. The code workflow was updated in
`experiments/grug/moe/agent.md` to require Iris submission and full procedure
completion for MoE experiments.

Operationally, one of these is needed before retrying:

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth list --filter=status:ACTIVE --format='value(account)'
```

The active account must have enough access on `hai-gcp-models` to discover the
Iris controller, or the caller must provide an explicit controller URL:

```bash
.venv/bin/iris --controller-url=http://localhost:10000 job run ...
```

## How OPS.md could have shortened this

- In `lib/iris/OPS.md` under "GCP Operations / Connecting", add a preflight
  command for controller discovery permissions:
  `gcloud compute instances list --project=hai-gcp-models --filter="labels.iris-marin-controller=true" --format="value(name)"`.
  This would distinguish missing controller VMs from missing GCP permissions
  before running `iris job run`.
- In `lib/iris/OPS.md` under "Troubleshooting", add a row for controller
  discovery failures that says a `compute.instances.list` 403 is an auth
  blocker and should be fixed by switching GCP account or using an explicit
  `--controller-url`.

## Artifacts

- PR: https://github.com/marin-community/marin/pull/5179
- Experiment issue: https://github.com/marin-community/marin/issues/5178
- Research logbook: `.agents/logbooks/moe-depth-mup-lr-sweep.md`
