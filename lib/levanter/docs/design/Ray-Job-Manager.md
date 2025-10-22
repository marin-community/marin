# Ray TPU Job Manager

This is a quick design document to explain how our Ray TPU Job Manager works.

## Introduction

Please see the [Ray documentation](https://docs.ray.io/en/latest/index.html) for more information on how Ray works. We provide only a brief overview here.

Ray is a resource-aware job scheduler, so you can specify the resources that a job requires:

```python
@ray.remote(num_cpus=4)
def my_cpu_job():
    ...
```

For GPUs, Ray lets you specify the number of GPUs you need:

```python
@ray.remote(num_gpus=1)
def my_gpu_job():
    ...
```

In Ray, TPUs are roughly represented the same way, but there are a number of problems with that approach.
In particular:

* Ray's granularity allows it to schedule a task on a single machine, not across multiple machines. In particular,
Ray can't directly schedule a task on a TPU slice that spans multiple machines (more precisely, multiple workers that)
are part of the same TPU slice.)
* Google requires that only one process on a machine can access the TPU at a time. This causes issues with Ray's
worker pool, which doesn't exit between tasks. We need to work around this.

This document explains how we work around those problems.

### A Note on Terminology

In the TPU world, a "TPU" is an accelerator card that is controlled by a VM called a worker. TPUs are arranged in "pods" and you can
get a slice of a pod (e.g. v4-256). Each worker controls 4 TPU cards, which is sometimes modeled as 8 TPU devices
and sometimes as 4 TPU devices, depending on TPU version.

Ray's fundamental abstraction is the "task." A task is modeled as a Python function decorated with `@ray.remote`
that runs in a process pool on some machine. It returns a future that can be used to get the result of the task.

In this document, I use "job" to mean something like an experiment run. It's a command that we want to run on
all the workers of a TPU slice until it completes, resuming from where it left off if it is preempted.
To run a job on a TPU slice, we will need to create a number of tasks that run on the workers of the TPU slice.
When a job is preempted, we need to reschedule the job by creating new tasks.

## Ray+TPU

### Scheduling Slices of TPUs

TPU slices must be used in a SPMD manner (this is probably not quite true, but it's the easiest way to use them).
This means that you need to run the same code on all workers of a slice at once.
Ray can't really do this directly. That is, you can't say:

```python
@ray.remote(tpu_slice="v4-256")
def my_tpu_job():
    ...
```

But you almost can, with a bit of indirection. Allen Wang (@allenwang28) at Google wrote [this gist](https://gist.github.com/allenwang28/e3400b9e9212b50aa1cda55ebeccea60#file-ray_tpu_task-py) that is most
of the way to a solution. The key idea is to schedule a task on the special `"TPU-${TPU_TYPE}-head"` resource
(where `${TPU_TYPE}` is like `"v4-256"`). If you start a job with this resource, you essentially get a "lock" on the TPU
slice. Once you have the lock, you can query the VM to get a unique resource that is shared only for this particular
slice. (Specifically, this resource is the unique pod slice name of the TPU slice `ray.util.accelerators.tpu.get_current_pod_name()`.)
You can then use this resource to schedule K tasks on the K workers that are on the same slice. These tasks do the actual work.

Managing preemption is then just a question of rescheduling the job when it gets preempted: getting a new head node,
getting a new pod slice name, and rescheduling the K tasks.
Detecting preemption (as opposed to application failure) is a bit tricky and still not fully tested.

### Dealing with `libtpu`

`libtpu` is the library that interfaces with the TPU. `libtpu` has a hard requirement that only one process on a machine
can access the TPU at a time. It manages this with a lockfile called `/tmp/libtpu_lockfile`. Ordinarily, this is fine,
as the lockfile is removed when the process exits. However, Ray maintains a pool of workers that don't ordinarily exit
between tasks. This means that the lockfile is not removed, and the next task that tries to access the TPU will fail.

As best I can tell, it's actually fine to remove this lockfile so long as you're not trying to access the TPU from
multiple processes on the same machine. Because we're using Ray to lock the resources, we're guaranteed that only one
process will be accessing the TPU at a time, so we just remove the lockfile when the task finishes.

Also note that we say that each worker only has 1 TPU, even though it has 4 (or 8) TPU devices. This is because
`libtpu` only lets one process access the TPU at a time, so the TPU functions more as a boolean lock than
as a semaphore.

## Ray+TPU+Docker

So above we have the core idea of how to use Ray with TPUs. However, there are a few additional complications when
we want our jobs to be running in separate docker containers. Some of this is just dealing with Ray+Docker, but some of it
is Ray+TPU+Docker specific.

We use a Docker container to run the core Ray scheduling process on each machine. We also want to use a different
per-job Docker container to run actual jobs. In theory, Ray can run tasks inside task-specific docker images, but I've heard it
doesn't work well. We also want to avoid a full Docker-in-Docker setup (which I've also heard is tricky), so we
instead want the scheduler to launch sibling containers. To do that, we bind-mount the docker socket into the
scheduler container.

## Ray+TPU+Docker

Above we discussed dealing with the TPU lockfile. The only real remaining issues are:

* you have to use `--privileged` to use TPUs.
* There's a bug in Ray's TPU/Docker support that [causes the `TPU-<TPU_TYPE>-head` resource to be assigned to all workers](https://github.com/ray-project/ray/pull/47777),
not just the leader. We have a patch.
