This is an attempt at a guide for babysitting the various tootsie runs


# 8b tootsie run (#600)

ATM we are running it off of the *main* branch of marin on the marin-big-run cluster.

```
ray dashboard ./infra/marin-big-run.yaml
```

I use this command to launch it:

```
python marin/run/ray_run.py --env_vars WANDB_API_KEY ${WANDB_API_KEY}  --env_vars HF_DATASETS_TRUST_REMOTE_CODE true --  python experiments/exp600_tootsie.py --run_only "[adept]" --force_run_failed true
```

I keep an eye on the wandb dashboard. I have runs sorted by last modified time. https://wandb.ai/stanford-mercury/marin?nw=nwuserdlwh

I have seen a few hangs, I think due to bugs in eval, which I think I have fixed. But if it hangs, just kill it.

For some stupid reason, Ray is frequently dropping the v4-2048 claiming it missed heartbeats. I have no idea why. I have to manually restart the run
and also reattach the v4-2048.

This is the command I run:

```bash
python infra/manual_ray_worker_launch.py --cluster_yaml infra/marin-big-run.yaml --reserved --tpu_type v4-2048 --zone us-central2-b --head 10.130.1.66 --tpu_name ray-worker-manual-66nw3n5u
```

(The IP and resource name are current as of 2022-03-26)


# Big Tootsies

These runs require more attention than the 8b run because of the preemptible compute and us not having a flexible trainer yet.

Locations:

- 70b: marin-eu-west4-a
- 22b: marin-us-east1
- 13b: marin-us-east5

**IMPORTANT**: These are all currently running off the big_tootsie branch. It is important that you use this branch.

**ALSO IMPORTANT**: These all run on preemptible compute, so they die all the time.

**Before relaunching,** be sure you go to the ray dashboard and check:

1. The job isn't still trying to run
2. there are enough v6e-128s
3. If the job is running, see if it's just timing out or if it's dying for some other reason. Probably it's timing out or there's too much preemption.

Chances are, if the ray job hasn't failed, you don't need to intervene, unless you want to increase/decrease the number of slices.

I have been using XXX v6e-128s, where XXX ranges from 2-8. The 70b needs at least 6. My procedure has been to monitor the dashboards and periodically kill the run and set to a different value if needed.

To relaunch a run,

```
ray dashboard ...
python marin/run/ray_run.py --env_vars WANDB_API_KEY ${WANDB_API_KEY}  --env_vars HF_DATASETS_TRUST_REMOTE_CODE true -- python marin/execution/status_actor.py kill
```

## 70b:

```
ray dashboard infra/marin-eu-west4-a.yaml
python marin/run/ray_run.py --env_vars WANDB_API_KEY ${WANDB_API_KEY}  --env_vars HF_DATASETS_TRUST_REMOTE_CODE true --  python experiments/exp750_tootsie70b.py --force_run_failed true --run_only '[real]'
```

#3 22b

```
ray dashboard infra/marin-us-east1.yaml
python marin/run/ray_run.py --env_vars WANDB_API_KEY ${WANDB_API_KEY}  --env_vars HF_DATASETS_TRUST_REMOTE_CODE true --  python experiments/exp750_tootsie70b.py --force_run_failed true --run_only '[22b]'
```


## 13b

```
ray dashboard infra/marin-us-east5.yaml
python marin/run/ray_run.py --env_vars WANDB_API_KEY ${WANDB_API_KEY}  --env_vars HF_DATASETS_TRUST_REMOTE_CODE true --  python experiments/exp750_tootsie70b.py --force_run_failed true --run_only '[13b]'
```


Cf. https://github.com/stanford-crfm/marin/issues/750
