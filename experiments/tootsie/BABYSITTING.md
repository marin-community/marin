This is an attempt at a guide for babysitting the various tootsie runs.

> **Operator review pending.** This doc was rewritten mechanically from the
> legacy Ray launcher to the current Iris-based pattern. @dlwh / @Helw150 /
> @rjpower — please sanity-check the exact flags before relying on these
> commands. The `--force_run_failed` / `--run_only` semantics on the tootsie
> experiments are preserved verbatim.

See [`lib/iris/OPS.md`](../../lib/iris/OPS.md) for Iris CLI reference and
troubleshooting (job logs, stop, tunnel, etc.).

# 8b tootsie run (#600)

ATM we are running it off of the *main* branch of marin on the shared Marin
Iris cluster.

I use this command to launch it:

```
uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_DATASETS_TRUST_REMOTE_CODE true \
  -- python -m experiments.exp600_tootsie --run_only "[adept]" --force_run_failed true
```

I keep an eye on the wandb dashboard. I have runs sorted by last modified time. https://wandb.ai/marin-community/marin?nw=nwuserdlwh

I have seen a few hangs, I think due to bugs in eval, which I think I have fixed. But if it hangs, just kill it (`iris job stop <job_id>`).

Under Iris, preemption handling and TPU restart are the scheduler's job — manual
`ray_worker_launch`-style reattachment is no longer needed. If a run is stuck
for a different reason, open the Iris dashboard (`iris cluster dashboard`) or
check `iris job bug-report <job_id>`.

# Big Tootsies

These runs require more attention than the 8b run because of the preemptible compute and us not having a flexible trainer yet.

**IMPORTANT**: These are all currently running off the big_tootsie branch. It is important that you use this branch.

**ALSO IMPORTANT**: These all run on preemptible compute, so they die all the time.

**Before relaunching,** check `iris job list --state running` and:

1. The job isn't still trying to run.
2. There is enough unreserved capacity on the target cluster (`iris cluster vm status`).
3. If the job is running, use `iris job logs /<user>/<job-id> -f` or `iris job summary /<user>/<job-id>` to see if it's just timing out or dying for some other reason. Probably it's timing out or there's too much preemption.

Chances are, if the Iris job hasn't failed, you don't need to intervene, unless you want to increase/decrease the number of slices.

I have been using XXX v6e-128s, where XXX ranges from 2-8. The 70b needs at least 6. My procedure has been to monitor the dashboards and periodically stop the run (`iris job stop <job_id>`) and relaunch with a different value if needed.

To cancel a stuck run:

```
uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_DATASETS_TRUST_REMOTE_CODE true \
  -- python -m marin.execution.status_actor kill
```

## 70b

```
uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_DATASETS_TRUST_REMOTE_CODE true \
  -- python -m experiments.exp750_tootsie70b --force_run_failed true --run_only '[real]'
```

## 22b

```
uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_DATASETS_TRUST_REMOTE_CODE true \
  -- python -m experiments.exp750_tootsie70b --force_run_failed true --run_only '[22b]'
```

## 13b

```
uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_DATASETS_TRUST_REMOTE_CODE true \
  -- python -m experiments.exp750_tootsie70b --force_run_failed true --run_only '[13b]'
```


Cf. https://github.com/marin-community/marin/issues/750
