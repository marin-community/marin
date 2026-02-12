# Fair Scheduler Design for Iris

## Context

This document reviews Spark's Fair Scheduler properties and explores how similar concepts could be applied to Iris to improve multi-user scheduling and resource fairness. This addresses issue #2762 and relates to #2742 (scheduler progress guarantees).

See:
- Issue #2762: Iris fair scheduler
- Issue #2742: Iris scheduler should guarantee progress
- [Spark Fair Scheduler Documentation](https://spark.apache.org/docs/latest/job-scheduling.html#scheduling-within-an-application)

## Current Iris Scheduler Behavior

Iris currently implements a **first-fit FIFO scheduler** with the following characteristics:

**Scheduling Algorithm** (`lib/iris/src/iris/cluster/controller/scheduler.py`):
- Processes pending tasks in FIFO order
- Uses first-fit assignment: attempts to assign each task to the first available worker with sufficient capacity
- Implements back-pressure via `max_building_tasks_per_worker` (default: 4) to limit concurrent setup operations
- Supports coscheduling: all tasks in a coscheduled job must be assigned atomically to workers in the same group
- Prevents head-of-line blocking: if a large task doesn't fit, smaller tasks behind it can still be scheduled

**Key Properties**:
- **No job prioritization**: All jobs are treated equally regardless of user, submission time, or resource requirements
- **No fairness guarantees**: A single large job can monopolize cluster resources
- **No resource quotas**: Users can submit unlimited work
- **No pool-based scheduling**: All jobs compete in a single queue

**Strengths**:
- Simple and predictable
- Low scheduling overhead
- Deterministic ordering for coscheduled jobs

**Limitations**:
- In multi-user environments, long-running jobs can starve short jobs
- No way to prioritize interactive/development work over batch jobs
- No mechanism to guarantee minimum resources for specific users or job types
- Issue #2742: Dynamic job hierarchies (Ray-style child jobs) can prevent progress for unrelated top-level jobs

## Spark Fair Scheduler Overview

Spark's Fair Scheduler provides **round-robin resource allocation** between jobs, enabling:
- All jobs to receive roughly equal cluster resources over time
- Short jobs submitted during long-running jobs to start immediately
- Multi-user environments with fairness guarantees
- Flexible prioritization via pools and weights

### Core Concepts

#### 1. Scheduling Mode

Two modes available:
- **FIFO** (default in Spark standalone): Tasks from the same job run before tasks from other jobs
- **FAIR**: Tasks from all jobs share resources in round-robin fashion

Configuration:
```scala
conf.set("spark.scheduler.mode", "FAIR")
```

#### 2. Scheduler Pools

**Purpose**: Group jobs with different scheduling policies

**Key Features**:
- Jobs are assigned to named pools (default pool if unspecified)
- Each pool can have its own scheduling mode (FIFO or FAIR)
- Pools compete fairly for resources based on weights
- Within each pool, jobs follow the pool's scheduling mode

**Setting Pool for Jobs**:
```scala
// Assign subsequent jobs in this thread to pool1
sc.setLocalProperty("spark.scheduler.pool", "pool1")
```

**Use Cases**:
- High-priority pool for interactive queries
- Separate pools per user or team
- Production vs. development workloads

#### 3. Pool Properties

Each pool supports three configurable properties:

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| **schedulingMode** | FIFO \| FAIR | FIFO | Controls whether jobs within the pool queue (FIFO) or share resources (FAIR) |
| **weight** | Integer | 1 | Pool's share relative to other pools. Weight of 2 gets 2x resources. Can implement priorities (e.g., weight=1000 for critical jobs) |
| **minShare** | CPU cores | 0 | Minimum guaranteed resources. Fair scheduler satisfies all pools' `minShare` before redistributing extra resources |

#### 4. Configuration via XML

Pools are defined in an XML configuration file:

```xml
<?xml version="1.0"?>
<allocations>
  <pool name="production">
    <schedulingMode>FAIR</schedulingMode>
    <weight>1</weight>
    <minShare>2</minShare>
  </pool>
  <pool name="test">
    <schedulingMode>FIFO</schedulingMode>
    <weight>2</weight>
    <minShare>3</minShare>
  </pool>
</allocations>
```

Configuration path:
```scala
// Local file
conf.set("spark.scheduler.allocation.file", "file:///path/to/file")

// HDFS file
conf.set("spark.scheduler.allocation.file", "hdfs:///path/to/file")
```

#### 5. Resource Allocation Algorithm

Spark's Fair Scheduler implements a two-phase allocation:

1. **Phase 1: Minimum Shares**
   - Allocate resources to satisfy each pool's `minShare`
   - If total `minShare` exceeds cluster capacity, proportionally reduce allocations

2. **Phase 2: Fair Distribution**
   - Distribute remaining resources proportionally by `weight`
   - A pool with weight=2 receives twice as many resources as weight=1
   - Pools that don't need their full allocation release resources for redistribution

**Example**:
- Cluster: 10 CPU cores
- Pool A: minShare=2, weight=1 → receives 2 cores (minShare) + 2 cores (fair share of 8 remaining) = 4 cores
- Pool B: minShare=3, weight=2 → receives 3 cores (minShare) + 5.33 cores (fair share of 8 remaining) ≈ 8 cores

#### 6. Within-Pool Scheduling

After resources are allocated to pools, jobs within each pool are scheduled according to the pool's `schedulingMode`:

- **FIFO Mode**: Jobs queue behind each other; first job receives all pool resources until complete
- **FAIR Mode**: Jobs within the pool share resources fairly (similar to pool-level fair sharing)

#### 7. Additional Features

**JDBC/SQL Scheduling**: For Spark SQL Thrift Server JDBC clients:
```sql
SET spark.sql.thriftserver.scheduler.pool=accounting;
```

**PySpark Considerations**: PySpark doesn't synchronize Python threads with JVM threads by default. Use `pyspark.InheritableThread` to inherit local properties across threads.

## Mapping to Iris

### Key Differences Between Spark and Iris

| Aspect | Spark | Iris |
|--------|-------|------|
| **Execution Model** | Single application with multiple jobs | Multiple independent jobs from different users |
| **Task Granularity** | Fine-grained tasks (seconds to minutes) | Longer-running tasks (minutes to hours) |
| **Resource Model** | CPU cores + memory per task | CPU, memory, GPU/TPU, custom attributes |
| **Scheduling Context** | Within a single Spark application | Across multiple user jobs |
| **Dynamic Hierarchy** | Limited | Ray-style dynamic job creation (issue #2742) |

### Proposed Concepts for Iris Fair Scheduler

#### 1. Job Pools

**Iris Equivalent**: Add pool assignment to job submission

```python
# Job submission with pool assignment
job = iris_client.submit(
    fn=my_task,
    pool="interactive",  # New parameter
)
```

**Implementation**:
- Add `pool: str` field to `JobRequest` proto
- Default pool: "default"
- Pool configuration stored in controller state
- Jobs inherit pool from parent job for Ray-style hierarchies

#### 2. Pool Configuration

**Iris Equivalent**: YAML-based pool configuration

```yaml
pools:
  - name: interactive
    scheduling_mode: FAIR
    weight: 2
    min_share:
      cpu: 10
      memory_gb: 20
      gpu_count: 1

  - name: batch
    scheduling_mode: FIFO
    weight: 1
    min_share:
      cpu: 5
      memory_gb: 10

  - name: production
    scheduling_mode: FAIR
    weight: 10
    min_share:
      cpu: 20
      memory_gb: 40
      gpu_count: 2
```

**Key Differences from Spark**:
- `minShare` is multi-dimensional (CPU, memory, GPU/TPU) instead of just CPU cores
- Support for device types (GPU/TPU) and device variants
- Can specify constraints for minimum shares (e.g., "production pool needs GPU workers")

#### 3. Scheduling Modes

**FIFO Mode** (current behavior):
- Process jobs in submission order within the pool
- First job receives all pool resources until complete or blocked

**FAIR Mode** (new):
- All active jobs in pool receive equal share of pool's resources
- Round-robin assignment: each job gets one task assigned before cycling to next job
- Prevents large jobs from monopolizing pool resources

**Implementation Strategy**:
```python
# In Scheduler.find_assignments()
if pool.scheduling_mode == FAIR:
    # Round-robin: assign one task per job, cycling through jobs
    tasks_to_schedule = round_robin_task_selection(pending_tasks)
else:  # FIFO
    # Current behavior: process tasks in order
    tasks_to_schedule = pending_tasks
```

#### 4. Weight-Based Resource Allocation

**Phase 1: Allocate Minimum Shares**
```python
def allocate_min_shares(pools: list[Pool], total_capacity: ResourceCapacity) -> dict[str, ResourceCapacity]:
    """Allocate minimum shares to each pool.

    Returns: Dict mapping pool_name -> allocated resources
    """
    allocations = {}
    remaining = total_capacity.copy()

    for pool in pools:
        # Allocate up to min_share, limited by remaining capacity
        allocated = ResourceCapacity(
            cpu=min(pool.min_share.cpu, remaining.cpu),
            memory=min(pool.min_share.memory, remaining.memory),
            gpu_count=min(pool.min_share.gpu_count, remaining.gpu_count),
        )
        allocations[pool.name] = allocated
        remaining -= allocated

    return allocations, remaining
```

**Phase 2: Distribute Remaining Resources by Weight**
```python
def allocate_by_weight(pools: list[Pool], remaining: ResourceCapacity, base_allocations: dict) -> dict:
    """Distribute remaining resources proportionally by pool weight."""
    total_weight = sum(p.weight for p in pools)

    for pool in pools:
        share = pool.weight / total_weight
        base_allocations[pool.name] += ResourceCapacity(
            cpu=int(remaining.cpu * share),
            memory=int(remaining.memory * share),
            gpu_count=int(remaining.gpu_count * share),
        )

    return base_allocations
```

#### 5. Per-Pool Task Scheduling

**Modified Scheduling Flow**:
1. Compute resource allocation for each pool (min share + weighted distribution)
2. For each pool (in priority order):
   - Create a pool-specific `SchedulingContext` with allocated capacity
   - Schedule tasks according to pool's scheduling mode (FIFO or FAIR)
   - Track resource usage per pool
3. Optionally: Allow pools to "borrow" unused capacity from other pools

**Example**:
```python
def find_assignments_with_pools(
    self,
    pending_tasks: list[ControllerTask],
    workers: list[ControllerWorker],
    pools: list[Pool],
) -> SchedulingResult:
    """Fair scheduler with pool support."""

    # Phase 1: Compute pool allocations
    total_capacity = compute_total_capacity(workers)
    pool_allocations, remaining = allocate_min_shares(pools, total_capacity)
    pool_allocations = allocate_by_weight(pools, remaining, pool_allocations)

    result = SchedulingResult()

    # Phase 2: Schedule within each pool
    for pool in sorted(pools, key=lambda p: -p.weight):  # Higher weight first
        pool_tasks = [t for t in pending_tasks if get_pool(t) == pool.name]

        if pool.scheduling_mode == FAIR:
            pool_tasks = round_robin_by_job(pool_tasks)

        # Create pool-specific context with allocated capacity
        pool_context = create_pool_context(workers, pool_allocations[pool.name])

        for task in pool_tasks:
            task_result = self.try_schedule_task(task, pool_context)
            if task_result.success:
                result.assignments.append((task, task_result.worker))

    return result
```

#### 6. User Quotas (Addressing Issue #2742)

Issue #2742 notes that dynamic job creation (Ray-style) can prevent progress for unrelated jobs. A quota system addresses this:

**Per-User Quotas**:
```yaml
quotas:
  - user: alice
    max_cpu: 100
    max_memory_gb: 200
    max_gpu_count: 4

  - user: bob
    max_cpu: 50
    max_memory_gb: 100
    max_gpu_count: 2
```

**Implementation**:
- Track resource usage per user (sum of all running tasks)
- When user hits quota, deprioritize their pending tasks
- Apply quota checks before pool allocation

**Algorithm**:
```python
def enforce_quotas(pending_tasks: list[ControllerTask], quotas: dict[str, Quota]) -> list[ControllerTask]:
    """Filter tasks that exceed user quotas to the back of the queue."""

    current_usage = compute_user_usage()

    within_quota = []
    over_quota = []

    for task in pending_tasks:
        user = task.user
        if current_usage[user] < quotas[user]:
            within_quota.append(task)
        else:
            over_quota.append(task)

    # Within-quota tasks scheduled first, then over-quota tasks
    return within_quota + over_quota
```

This ensures:
- Users creating many "cheap" child jobs don't monopolize the cluster
- New top-level submissions from other users can make progress
- Fair sharing across users, not just jobs

#### 7. Job Depth Prioritization (Issue #2742)

Issue #2742 suggests prioritizing jobs depth-first within each user's quota:

**Job Depth Tracking**:
```python
# Add to JobRequest proto
depth: int = 0  # 0 for top-level jobs, 1 for child jobs, etc.
parent_job_id: str | None = None
```

**Scheduling Priority**:
```python
def prioritize_by_depth(tasks: list[ControllerTask]) -> list[ControllerTask]:
    """Sort tasks by job depth (depth-first) to ensure child jobs make progress."""

    # Group by job, sort jobs by depth (ascending), flatten back to tasks
    tasks_by_job = defaultdict(list)
    for task in tasks:
        tasks_by_job[task.job_id].append(task)

    jobs_by_depth = sorted(
        tasks_by_job.items(),
        key=lambda item: (get_job_depth(item[0]), get_job_submission_time(item[0]))
    )

    return [task for job_id, job_tasks in jobs_by_depth for task in job_tasks]
```

**Combined Algorithm** (Quota + Depth):
1. Enforce per-user quotas
2. Within each user's allocation, prioritize by job depth
3. Within each depth level, maintain FIFO order (or FAIR mode if configured)

### Implementation Phases

#### Phase 1: Pool Infrastructure (Minimal Fair Scheduler)

**Goals**:
- Add pool concept to job submission
- Implement YAML-based pool configuration
- Modify scheduler to track pool assignments

**Changes**:
- Add `pool` field to `JobRequest` proto
- Add pool configuration to `ControllerState`
- Update dashboard to show pool assignments

**Testing**:
- Submit jobs to different pools
- Verify pool assignment is tracked and displayed

#### Phase 2: Weight-Based Resource Allocation

**Goals**:
- Implement two-phase allocation (minShare + weight)
- Allocate resources to pools before scheduling tasks

**Changes**:
- Implement `allocate_min_shares()` and `allocate_by_weight()`
- Modify `Scheduler.find_assignments()` to compute pool allocations
- Add metrics/logging for pool resource usage

**Testing**:
- Configure pools with different weights and minShares
- Verify resources are allocated according to configuration
- Verify pools can't exceed their allocation

#### Phase 3: Fair Scheduling Mode

**Goals**:
- Implement FAIR mode within pools (round-robin job scheduling)

**Changes**:
- Add `schedulingMode` to pool configuration
- Implement `round_robin_by_job()` task selection
- Modify scheduler to honor pool scheduling mode

**Testing**:
- Submit multiple jobs to a FAIR pool
- Verify tasks are interleaved across jobs
- Verify FIFO pools maintain current behavior

#### Phase 4: User Quotas

**Goals**:
- Implement per-user resource quotas
- Deprioritize tasks from users over quota

**Changes**:
- Add quota configuration (YAML or database)
- Track per-user resource usage
- Implement quota enforcement in scheduler

**Testing**:
- Configure user quotas
- Submit jobs exceeding quota
- Verify over-quota jobs are deprioritized

#### Phase 5: Job Depth Prioritization

**Goals**:
- Track job hierarchy (parent/child relationships)
- Prioritize child jobs over unrelated top-level jobs

**Changes**:
- Add `depth` and `parent_job_id` to `JobRequest`
- Implement depth-first task ordering
- Integrate with quota system

**Testing**:
- Create dynamic job hierarchies (Ray-style)
- Verify child jobs are scheduled before other top-level jobs
- Verify quotas limit total user resources across hierarchy

## Open Questions

1. **Pool Assignment Strategy**: Should pools be specified at job submission time, or automatically assigned based on user/job properties?

2. **Resource Borrowing**: Should pools be allowed to use unused capacity from other pools? If so, how to reclaim resources when the original pool needs them?

3. **Preemption**: Spark doesn't support preemption. Should Iris support preempting tasks from over-quota users or low-priority pools?

4. **Multi-Dimensional minShare**: How to handle minShare when some dimensions are satisfied but others aren't? (e.g., pool needs 10 CPUs and 2 GPUs, but only 1 GPU is available)

5. **Coscheduling and Pools**: How do coscheduled jobs interact with pool allocations? Should all tasks in a coscheduled job come from the same pool?

6. **Dynamic Pool Configuration**: Should pools be configurable at runtime, or require controller restart?

7. **Dashboard Integration**: What pool/quota metrics should be surfaced in the dashboard?

## Related Work

- **Kubernetes Fair Scheduler**: Not part of core Kubernetes; typically implemented via admission controllers and resource quotas
- **YARN Fair Scheduler**: Similar to Spark; uses hierarchical queues with min/max resources
- **Mesos DRF (Dominant Resource Fairness)**: Multi-resource fairness algorithm; allocates based on each user's "dominant resource"
- **Slurm Fair Share**: Uses decay factor to compute historical usage; users with less historical usage get priority

## References

- [Spark Fair Scheduler Documentation](https://spark.apache.org/docs/latest/job-scheduling.html#scheduling-within-an-application)
- [Hadoop Fair Scheduler Guide](https://hadoop.apache.org/docs/stable/hadoop-yarn/hadoop-yarn-site/FairScheduler.html)
- [Dominant Resource Fairness Paper](https://cs.stanford.edu/~matei/papers/2011/nsdi_drf.pdf)
- Iris Issue #2762: Fair scheduler
- Iris Issue #2742: Scheduler should guarantee progress
