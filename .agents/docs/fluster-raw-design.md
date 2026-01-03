# Fluster Design Notes - Raw Transcription

**Source**: Handwritten notes from `ray-rewrite.pdf` dated 2025-12-21

## Page 1: Initial Problem Statement

**Date**: 2025-12-21

Thinking about Marin, codebase is kind of a mess, have to somehow convince David that inheritance in configuration files is a bad thing.

Anyway. For cluster management, I'd like to remove the usage of Ray - still use it for local calls but drop it for distributed, at least in the short term. This is complicated by how our whole Ray cluster system works - I can create a per-user cluster easily enough, but then how do I hand off control for it? How do I setup resources on it, etc.

So far I have:

- Main cluster manager allocates devices on all clusters - we have a single dockerfile for all workers
- Workers bootstrap into Ray automatically
- Users get combined Ray cluster

## Page 2: Combined Ray Cluster Questions

What does this combined Ray cluster look like though? If we also require this auto-nonsense, ask how will that work out? I'm not super excited to try to replace Ray in one shot...

We can reuse the token magic, but we'll now have a second part maybe? Danel? What's the flow of control starting from e.g. `ray_run`?

```
ray run exp123.py
  -> Submit job request for 123.py
     {ram: x, disk: y, cpu: z, files}
```

- exp123 is Mr controller & spawns one-off clusters?
- forward logs from those Ray clusters?
- exp123.py -> executor_step -> Launch JobRequest
- Launch -> ??
  - Find appropriate machines in cluster X
  - Start Ray cluster on them for user X
  - Launch user job on that cluster

## Page 3: Library Design Questions

So are we going to have to roll our own RPC library etc to make this a thing? Rust type.

What does our own library give us? We'd start up the application on all cluster workers (or min/max) & ask for mount TPUs where appropriate. Docker boot should be huge or quick - 1 second or so.

Let's see what we want the overall system to look like. Let's start w/ the notebook version - I want to be able to boot & connect to a "job" which runs our process & logs go to a common logging location & are tied to the notebook/controller.

For now, we'll assume:
- Jobs have shared namespace for actors, but gang scheduling for bootstrap
- Namespace comes from executor/top-level
- So they can talk to each other as expected

Unlike Ray, tasks are automatically started on each worker -> for levanter etc.

## Page 4: Fray RPC Interface

So for the Fray RPC we'd have something like:

```python
ctx = fray.rpc.init(namespace="vps2",
                    coordinator="host:port")
ctx.wait()
```

This binds a worker into the pool.

For levanter, we can skip this? Or run it but don't do the wait -> since each call will be able to gang scheduled for us.

I want all basic operations to be accessible via HTTP APIs (in addition to binary versions). The Rust lib is HTTP/2 based std IIUC, or at least trivial to query.

So we'll have these RPCs, roughly:

**cls Fluster**:
- `connect()` -> namespace
- `create_job(clusters/requirements, workspace, username, env, etc.)`
- `delete/poll/pause`
- retries?

Fluster sets up Fray env: cookie, user, workspace, namespace, coordinator(?)

**fray**:
- `put/get/run/init/wait`

## Page 5: Execution Flow

```
ray run ->
  -> cluster.run (coordinator)
     -> boots up initial task on non-preemptible machine
        -> that task runs anywhere w/ CPU & RAM = small
           -> (Q) Who sets coordinator address? Logical for it to come from the executor process.
```

```
cls fluster -> |exec
              |namespace/coord etc
        (pre-process) (train)
```

But how is it communicated to the job launch system? Perhaps fluster "marks" the first task as a coordinator? What if you want to just run a training job by itself?

Probably want to have an explicit:

```python
fray.init()  # in
```

In executor, fluster will & thread the Fray context info into the job requests. This will let us handle both types easily.

## Page 6: Scatter/Gather and Piccolo Tables

Then for the details on the fluster & Fray implementations...

Note Ray has its whole generator shenanigans - can we do something better? Our real usage is w/ Zephyr - what do we want in Zephyr? We really just need to support scatter across a graph of processes.

```
(M) ooo    if M[i] fails, we want to be
(R) o|o    able to replay as needed.
(M) ooo
```

Eventually we want this to snoop to RunGraph(), as that would allow us to easily replay through the whole graph.

More incrementally, we'd ideally like to have the mapper return N streams & thread these to the reducers automatically...

**Or create a Piccolo style table for the reducer namespace**:

```
CreateTable(P=xxhash, W=[R0..Rn])
                      S=100)
```

Round robin across reducers, etc. How would we handle replay in this world? Tag each entry w/ its [source-task, index] & replay?

## Page 7: Table Implementation Strategy

In the short term though, do I want to support Ray style execution? How should I do this? Easiest option is to push everything to GCS & read from there, for the amount of data we're dealing w/ this is probably fine. Yeah let's start simple, not throw user speed, a strategy for the same context to have either memory or GCS is tidy.

We'll move onto a more performant etc system as we go. Then we can always just replay operations until they finish? or

What do we do w/ the current weird generator thing? Is there a way to frame that as an implementation detail of the Ray impl? What would the Fray vs Ray abstraction work like? Could we get a bit closer to what we'd like the Fray interface to be?

E.g. could I implement the Fray table as a Ray generator to coordinator handoff?

```python
task(table_ctx) -> ctx.put(k, v)
```

**Or** a dataflow Fray table could be independent of the Ray/Fray backends. Just always have it dump to disk. Then we pipe this into Fray & read from memory as an optimization.

## Page 8: Implementation Recap

So this lives independently from the Fray context to some extent, like the psf library. You could have a fray.table, it uses an fsspec/obj-store backend & you can read/write partitions, w/ the expected flushing to disk etc. Then our simple retry mechanism is easy - always retry on preemption - b/c the table can easily partition the source tasks as needed.

L000-L000 source dests = a lot of data, but if we use e.g. Vortex we'll instead write one file per source & scan over it.

**To recap**:
- Introduce fray.table
- Port zephyr to use it (`ctx.table`)
  - eventually delete obj_store
- Create cluster impl
- Create Fray RPC layer
  - -> actors, task launch
  - -> That's about it
  - -> No generator etc anymore...

Once we have things implemented, we can go back & retrofit table as a builtin & skip the GCS dump.

**What does that final world look like?**
We'd have a local & RPC backend for our table & task launching service, as well as the cluster execution logic.

So e.g.:
```
fray/
  cluster/
  table/
  job/

fray/
  backend/
    local.py
    per_rpc.py
```

## Page 9: Alternatives Considered

### Alternative: Keep Fray Cluster + Ray Backend

Instead of our own RPC, we could instead still have our Fray cluster start up a Ray backend, booting into a configured Ray environment.

This is attractive as it allows us to incrementally replace our Ray cluster logic w/out having to handle our RPC logic at the same time.

**NB**: This was our initial idea but we've elected to reject it for a few reasons:

- **Additional user complexity** - do track job states in the new world, we'd have a new Ray cluster for each user job - w/ out a public HTTP interface, this means significant magic to tunnel or proxy user connections to track logs or job states.

- **We'd fail to capture significant benefit** from the move, as we'd still be using Ray for the most critical functionality.

### Stick w/ Ray

Now that we have Zephyr & Fray, is Ray all that bad? We've "contained the damage", as it were.

## Page 10: More Alternatives

**Can we bear it now?**

While we feel, while Ray is less painful now that it's encapsulated, it's still a painful experience for job management & stands in the way of meaningful cluster improvements...

### Alternative: Implement Full Ray Semantics

Instead of developing our own table API, we could fully implement the Ray generator & reference counting support.

**We're reluctant to do this for a few reasons:**

- **Hard to get right** - the refcount semantics for Ray are complex & partly broken in Ray. Do we need bug for bug compatibility?

- **Limited upside** - as we've discovered w/ Zephyr, the Ray semantics are kind of crap, do we want to impl this for limited upside?
