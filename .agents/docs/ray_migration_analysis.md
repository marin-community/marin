# Ray Migration Analysis - Marin Codebase

**Date:** 2025-10-03
**Purpose:** Understand Ray usage patterns and develop migration strategy
**Scope:** Complete analysis of `src/`, `experiments/`, and `submodules/levanter/src/`

---

## Executive Summary

The Marin codebase has **deep, mission-critical dependence** on Ray across 90+ files. Ray serves as the core infrastructure for:
- Distributed data processing (80+ files)
- RL training & rollout generation (complex actor-based patterns)
- TPU multi-slice orchestration (1400+ lines of custom code)
- Workflow orchestration (Executor framework)

**Key Finding:** Ray usage falls into two distinct categories with **fundamentally different requirements:**

1. **Batch Processing** (80+ files): Stateless map-reduce tasks, could migrate to simpler alternatives
2. **RL Training** (10+ files): Stateful actors with tight coupling, requires sophisticated replacement

**Migration Complexity:** High - estimated 5-6 months with dedicated engineering effort

---

## 1. Quantitative Analysis

### 1.1 Ray API Usage Frequency

| API / Feature | Count | Files | Notes |
|--------------|-------|-------|-------|
| `@ray.remote` decorators | ~140 | 84 | Task & actor definitions |
| `.remote()` calls | 268 | 77 | Task/actor invocations |
| `ray.get()` | 148 | 74 | Blocking result retrieval |
| `ray.wait()` | 37 | 27 | Async waiting with backpressure |
| `ray.put()` | 1-2 | 1-2 | Object store writes (rarely used directly) |
| `ray.init()` | 3 | 3 | Minimal - mostly connect to existing cluster |
| Ray actors | ~15 | 8 | Long-lived stateful services |
| Runtime environments | 93 | 25 | Pip dependencies, env vars |
| Resource specifications | 133 | 57 | Memory/CPU/GPU/TPU constraints |
| PlacementGroups | 1 | 1 | For tensor parallelism |
| NodeAffinityScheduling | Heavy | Levanter | TPU host affinity |
| Job submissions | 7 | 2 | Via JobSubmissionClient |

### 1.2 Files by Category

**Data Processing (Batch):**
- Transform: 24 files
- Crawl: 32 files
- Download: 12 files
- Processing: 12 files
- **Total: ~80 files**

**RL/Training:**
- RL workers: 4 files
- Weight transfer: 3 files
- Training: 2 files
- Curriculum: 1 file
- **Total: ~10 files**

**Infrastructure:**
- Executor framework: 2 files
- Cluster management: 3 files
- Utilities: 4 files
- Levanter TPU: 4 files
- **Total: ~13 files**

---

## 2. Current Encapsulation State

### 2.1 Well Encapsulated ✅

These components have clean abstractions and clearer migration paths:

#### **Executor Framework** (`src/marin/execution/executor.py`)
- **Abstraction:** DAG-based workflow orchestration
- **Ray coupling:** Minimal - only `@ray.remote` and `.remote()`
- **Interface:**
  ```python
  ExecutorFunction = Callable | ray.remote_function.RemoteFunction | None
  ```
- **Migration difficulty:** Low - swap RemoteFunction type
- **Critical:** Yes - workflow orchestration is core feature

#### **StatusActor** (`src/marin/execution/status_actor.py`)
- **Abstraction:** Distributed lock manager for output paths
- **Ray coupling:** Single actor definition
- **Interface:**
  - `get_task_id_with_lock(output_path) -> str`
  - `get_lock_by_replacing_task_id(...) -> bool`
- **Migration difficulty:** Low - any distributed KV store with CAS
- **Alternative:** Redis, etcd, Firestore

#### **Simple Backpressure** (`src/marin/core/runtime.py:243`)
- **Abstraction:** Ordered task execution with concurrency control
- **Ray coupling:** Uses `ray.wait()` internally
- **Key property:** Returns results in submission order (not completion order)
- **Migration difficulty:** Low - can reimplement with async/await
- **Used by:** All data processing pipelines

### 2.2 Poorly Encapsulated ❌

These leak Ray abstractions throughout the codebase:

#### **Direct `@ray.remote` Usage Everywhere**
- **Problem:** ~140 instances across 84 files
- **No abstraction layer** - each file directly imports Ray
- **Example pattern (appears in 80+ files):**
  ```python
  import ray

  @ray.remote(memory=4 * 1024 * 1024 * 1024, num_cpus=2)
  def my_transform_task(input_path, output_path):
      # Process file
      pass
  ```
- **Migration impact:** Must touch every file

#### **Resource Specifications**
- **Problem:** Ray-specific syntax scattered everywhere
- **133 occurrences** across 57 files
- **No unified abstraction**
- **Range:** Memory from 512MB to 350GB, CPUs from 0.01 to 16
- **Examples:**
  ```python
  @ray.remote(memory=1 * 1024 * 1024 * 1024)  # 1GB
  @ray.remote(memory=350 * 1024 * 1024 * 1024)  # 350GB
  @ray.remote(resources={"TPU": 4, "TPU-v4-8-head": 1})
  ```

#### **Runtime Environments**
- **Problem:** 93 instances of Ray-specific `runtime_env` dicts
- **Some abstraction** exists in `build_runtime_env_for_packages()` but underutilized
- **Example:**
  ```python
  runtime_env={"pip": ["fastparquet", "scikit-learn"]}
  runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu"}}
  ```

#### **Object References & `.remote()` Calls**
- **Problem:** 268 `.remote()` calls, no abstraction
- **Direct dependency** on Ray's execution model
- **Ordering semantics** baked into custom code
- **Example:**
  ```python
  futures = [task.remote(arg) for arg in args]
  results = ray.get(futures)
  ```

---

## 3. Batch Processing vs RL Training - Architectural Differences

### 3.1 Batch Processing Pattern

**Used in:** Data transforms, crawling, downloads, tokenization (~80 files)

**Characteristics:**
- **Stateless tasks** - each task is completely independent
- **Map-reduce style** - apply same function to many inputs
- **Finite workload** - known set of files/URLs to process
- **No inter-task communication** - results written to GCS/storage
- **Fault tolerance via retries** - idempotent operations
- **Latency tolerant** - seconds to minutes per task acceptable

**Typical Pattern:**
```python
@ray.remote(memory=4 * 1024 * 1024 * 1024, max_retries=5)
def process_file(input_path, output_path):
    data = read_from_gcs(input_path)
    transformed = apply_transform(data)
    write_to_gcs(output_path, transformed)
    return True

# Execute with concurrency control
files = list_gcs_files(input_dir)
file_pairs = [(f, compute_output_path(f)) for f in files]

responses = simple_backpressure(
    process_file,
    iter(file_pairs),
    max_in_flight=1000,
    fetch_local=True
)
```

**What's needed from distributed system:**
- Task scheduling with resource limits (CPU, memory)
- Concurrency control / backpressure
- Automatic retries on failure
- Progress tracking
- **NOT needed:** actors, state management, low-latency communication

**Migration candidates:** Dask, Prefect, Kubernetes Jobs, Modal

---

### 3.2 RL Training Pattern

**Used in:** Rollout workers, training workers, weight transfer, curriculum (~10 files)

**Fundamentally different architecture:**

#### **Long-Lived Stateful Services**

**RolloutWorker** (`src/marin/rl/rollout_worker.py`):
```python
class RolloutWorker:
    # Stateful components held in memory
    _inference_server: InferenceServer      # HTTP server on port 8000+
    _policy_model: LmModel                  # JAX model (GBs of weights)
    _transfer_client: WeightTransferClient  # Arrow Flight client
    _rollout_writer: RolloutWriter          # Continuous stream writer
    _environments: dict[str, MarinEnv]      # Game/task environments

    def run(self):
        # Infinite loop - never returns
        while self._running:
            # 1. Poll for new weights from training worker
            new_weights = self._transfer_client.check_for_update()
            if new_weights:
                self._update_model(new_weights)

            # 2. Get curriculum lesson
            lesson = ray.get(self.curriculum_actor.get_lesson.remote())

            # 3. Generate rollouts using inference server
            rollouts = self._generate_rollouts(lesson)

            # 4. Write to storage for training worker
            self._rollout_writer.write(rollouts)

            # 5. Report stats back to curriculum
            self.curriculum_actor.report_stats.remote(stats)
```

**TrainWorker** (`src/marin/rl/train_worker.py`):
```python
class TrainWorker:
    # Stateful training state
    transfer_server: WeightTransferServer  # Ray actor serving weights
    replay_buffer: ReplayBuffer            # In-memory rollout buffer
    model: LmModel                         # Training model
    optimizer_state: OptimizerState        # Optimizer state

    def train(self):
        # Infinite stream of data
        for batch in self.data_loader:  # Never ends
            # 1. Train on batch
            loss, new_model = train_step(batch, self.model)

            # 2. Every N steps, serve weights to rollout workers
            if step % sync_interval == 0:
                self.transfer_server.serve_weights(step, new_model)

            # 3. Log metrics
            self.tracker.log({"loss": loss}, step=step)
```

#### **Actor-Based Inter-Service Communication**

**WeightTransferServer** (`src/marin/rl/weight_transfer/arrow_flight.py:204`):
```python
@ray.remote(num_cpus=0)
class ArrowFlightServerActor:
    """
    Zero-copy weight serving via Arrow Flight protocol.
    Lives in training worker's process, serves to rollout workers.
    """
    def __init__(self):
        self._current_weights = {}  # step -> model weights
        self._server = FlightServer(...)  # gRPC server

    def serve_weights(self, step: int, weights):
        """Called by train worker to publish new weights."""
        self._current_weights[step] = weights
        # Weights stay in memory, served via Flight RPC

    def get_metrics(self) -> TransferMetrics:
        """Returns transfer statistics."""
        return self._metrics
```

**WeightTransferClient** (in rollout worker):
```python
class WeightTransferClient:
    def check_for_update(self) -> tuple[int, ModelWeights] | None:
        """Poll server for new weights. Returns None if no update."""
        # Arrow Flight RPC call to server actor
        latest_step = self._flight_client.get_latest_step()
        if latest_step > self._current_step:
            weights = self._flight_client.get_weights(latest_step)
            return latest_step, weights
        return None
```

**CurriculumActor** (`src/marin/rl/curriculum.py`):
```python
@ray.remote
class CurriculumActor:
    """
    Named actor shared across both rollout AND training workers.
    Coordinates curriculum progression.
    """
    def __init__(self):
        self._lessons: dict[str, Lesson] = load_lessons()
        self._stats: dict[str, list[RolloutStats]] = {}
        self._current_lesson_id = "lesson_1"

    def get_lesson(self) -> Lesson:
        """Called by rollout workers to get current task."""
        return self._lessons[self._current_lesson_id]

    def report_stats(self, stats: RolloutStats):
        """Called by rollout workers to report performance."""
        self._stats[stats.lesson_id].append(stats)

        # Maybe advance curriculum based on stats
        if self._should_advance():
            self._current_lesson_id = self._next_lesson()

    def save_checkpoint(self, path: str):
        """Called by train worker periodically."""
        # Persist curriculum state
        pass
```

#### **Coordination Pattern**

```
┌─────────────────┐         ┌─────────────────┐
│ TrainWorker     │         │ RolloutWorker   │
│                 │         │   (many)        │
│  ┌──────────┐   │         │                 │
│  │Transfer  │◄──┼─────────┼──Poll weights   │
│  │Server    │   │ Arrow   │                 │
│  │(Actor)   │   │ Flight  │                 │
│  └──────────┘   │         │                 │
│       │         │         │       │         │
│       ▼         │         │       ▼         │
│  Serve weights  │         │  Update model   │
│       │         │         │       │         │
└───────┼─────────┘         └───────┼─────────┘
        │                           │
        │    ┌──────────────┐       │
        │    │ Curriculum   │       │
        └───►│ Actor        │◄──────┘
             │ (Named)      │
             └──────────────┘
                Get lesson
                Report stats
```

#### **Key Differences from Batch Processing**

| Aspect | Batch Processing | RL Training |
|--------|------------------|-------------|
| **Execution Model** | Finite tasks (seconds to minutes) | Infinite loops (days) |
| **State** | Stateless | Stateful (models, buffers, servers) |
| **Communication** | None (via storage) | Direct actor-to-actor (RPC) |
| **Latency Requirements** | Tolerant (minutes) | Sensitive (milliseconds for weight sync) |
| **Failure Model** | Retry individual task | Restart entire service, restore state |
| **Resource Lifecycle** | Allocate per-task | Long-lived allocation (hours/days) |
| **Coordination** | Independent | Tightly coupled (rollout ↔ training ↔ curriculum) |
| **Data Transfer** | Write to GCS | Zero-copy shared memory (Arrow Flight) |
| **Discovery** | None | Named actors / service discovery |

---

## 4. TPU Infrastructure - Why It's Terrible

### 4.1 The Problem

**File:** `submodules/levanter/src/levanter/infra/ray_tpu.py` (1,405 lines)

This is a **massive workaround** for Ray's poor TPU support. You've built:
- Custom TPU topology management
- Preemption detection & handling
- Multi-slice coordination
- Resource pool management
- Lockfile cleanup hacks

**Why it exists:** Ray has no native understanding of:
- TPU topology (chips, hosts, slices)
- TPU preemption semantics
- Multi-slice networking (MEGASCALE)
- libtpu process model

### 4.2 Over-Engineered Actor Hierarchy

**3 levels of actors** for simple resource allocation:

```python
SlicePoolManager (Python class)
  └─ manages pool of...
     └─ SliceActor (Ray actor, 1 per TPU slice)
           - Has TPU-{type}-head resource
           - Manages pool of...
              └─ TPUHostActor (Ray actor, 1 per VM)
                    - Has TPU chips + slice_name resource
                    - Schedules...
                       └─ User task (Ray remote function)
                             - Scheduled via NodeAffinitySchedulingStrategy
```

**Each level implements:**
- Health checking (`healthy()` remote method)
- Teardown (`teardown()` remote method)
- Error handling & classification
- Retry logic
- State tracking

### 4.3 Preemption Detection Heuristics

**Because Ray doesn't expose preemption status, you guess from errors:**

```python
def _handle_ray_error(e: RayError):
    # Try to infer preemption from error type
    if isinstance(e, NodeDiedError):
        return TpuPreempted(e)
    elif isinstance(e, ActorUnavailableError):
        return TpuPreempted(e)
    elif isinstance(e, WorkerCrashedError):
        return TpuPreempted(e)
    elif isinstance(e, RaySystemError):
        return TpuRunError(e)  # Maybe not preemption?
    elif isinstance(e, RayTaskError):
        # Poll GCP API directly
        if get_current_tpu_is_preempted():
            return TpuPreempted(e)
        # Or check error message
        if "timed out" in str(e):
            return TpuPreempted(e)
        return TpuRunError(e)
```

**Real solution:** Query TPU status directly via GCP API, don't infer from Ray errors.

### 4.4 Manual Resource Tracking

**You manually track which TPU chip is on which node:**

```python
@dataclass(frozen=True)
class TPUHostInfo:
    slice_name: str      # e.g., "my-tpu-slice"
    worker_index: int    # Which VM in the slice (0-7 for v4-8)
    node_id: str        # Ray node ID
    num_tpus: int       # Chips on this VM (4 for v4-8)

# Later, manually schedule to specific host
scheduling_strategy=NodeAffinitySchedulingStrategy(
    node_id=host_info.node_id,
    soft=False
)
resources={"TPU": host_info.num_tpus}
```

**Why:** Ray doesn't understand TPU topology, so you build a parallel tracking system.

### 4.5 Libtpu Lockfile Hacks

```python
def _hacky_remove_tpu_lockfile():
    """
    libtpu only allows one process to access the TPU at a time.
    Ray's long-running daemon means lockfile persists across tasks.
    """
    if os.path.exists("/tmp/libtpu_lockfile"):
        try:
            os.unlink("/tmp/libtpu_lockfile")
        except PermissionError:
            logger.warning("Failed to remove lockfile")
            try:
                os.system("sudo rm /tmp/libtpu_lockfile")  # !!!
            except Exception:
                pass
```

**Workaround:** Use `max_calls=1` on all TPU tasks to force process restarts.

**Impact:** Overhead from process creation, potential resource leaks.

### 4.6 Multislice Coordination

**Manual MEGASCALE environment variable injection:**

```python
def _multislice_info_to_env_vars(multislice: MultisliceInfo) -> dict[str, str]:
    return {
        "MEGASCALE_COORDINATOR_ADDRESS": f"{multislice.coordinator_ip}:{multislice.port}",
        "MEGASCALE_NUM_SLICES": str(multislice.num_slices),
        "MEGASCALE_SLICE_ID": str(multislice.slice_id),
        "MEGASCALE_PORT": str(multislice.port),
    }
```

**You manage:**
- Coordinator IP discovery (use first slice's IP)
- Slice ID assignment (0-indexed)
- Port allocation
- Passing to Ray runtime_env

**GKE alternative:** This is all automatic with native TPU Podslices.

### 4.7 Flex Multislice Complexity

**Support for variable slice counts** (e.g., "give me 2, 4, or 8 slices - whatever is available"):

```python
def scale_multislice(self, num_slices: int | Sequence[int]) -> None:
    if isinstance(num_slices, int):
        self._scale_actor_pool(num_slices)
        return

    # Try to get maximum
    sorted_valid_sizes = sorted(num_slices)
    max_valid_size = sorted_valid_sizes[-1]

    try:
        self._scale_actor_pool(max_valid_size)
    except Exception:
        # Failed to get max, try next smaller size
        feasible_sizes = [s for s in sorted_valid_sizes if s <= current_size]
        max_feasible_size = feasible_sizes[-1]
        self._scale_actor_pool(max_feasible_size)
```

**With periodic checks to scale up:**
```python
def check_should_scale_up_multislice(self, num_slices: Sequence[int]) -> bool:
    # Every 3 hours, check if we can get more slices
    if current_time - last_check < 3 * 60 * 60:
        return False

    # Try to acquire more slices
    # If we get them, keep them
    # If not, release them and go back to previous size
```

**Why this is complex:**
- Ray doesn't understand spot/preemptible resources
- You're building your own capacity management
- GKE's autoscaling could do this

### 4.8 What You Should Do Instead

**Use GKE native TPU support:**

```yaml
# Kubernetes Job for TPU training
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job
spec:
  template:
    spec:
      restartPolicy: OnFailure
      nodeSelector:
        cloud.google.com/gke-tpu-topology: 4x4  # Multislice topology
      containers:
      - name: trainer
        image: marin-train:latest
        resources:
          limits:
            google.com/tpu: "16"  # GKE handles allocation
        env:
        - name: TPU_WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['cloud.google.com/gke-tpu-worker-id']
        - name: TPU_WORKER_HOSTNAMES
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['cloud.google.com/gke-tpu-worker-hostnames']
```

**GKE handles:**
- ✅ TPU topology (it knows v4-8 = 2 VMs with 4 chips each)
- ✅ Multislice coordination (MEGASCALE env vars set automatically)
- ✅ Preemption (restart policy handles it)
- ✅ Host affinity (pods are placed correctly)
- ✅ Lockfile issues (container restarts clean up)
- ✅ Resource allocation (no manual tracking)

**Your code becomes:**
```python
# Just use JAX - GKE already set up topology
import jax
devices = jax.devices("tpu")
mesh = Mesh(devices, axis_names=("data", "model"))
# That's it!
```

**Result:** Delete 1,405 lines of TPU infrastructure code.

---

## 5. Migration Strategy

### 5.1 Guiding Principles

1. **Different tools for different jobs** - don't force one solution for batch + RL
2. **Abstraction layer first** - hide Ray before replacing it
3. **Gradual migration** - keep existing Ray working during transition
4. **Fix TPU first** - biggest pain point, clearest win
5. **Validate continuously** - run dual backends in parallel

### 5.2 Recommended Backend Choices

#### **For Batch Processing (80+ files):**

**Recommendation: Dask**

**Why:**
- Familiar API (similar to Ray)
- Mature, stable, widely used
- Good for data pipelines
- Kubernetes native deployment
- Simple compared to Ray

**Example:**
```python
# Current Ray pattern
@ray.remote(memory=4 * 1024 * 1024 * 1024)
def process_file(input_path, output_path):
    ...

futures = [process_file.remote(f) for f in files]
results = ray.get(futures)

# Dask equivalent
from dask.distributed import Client

@dask.delayed
def process_file(input_path, output_path):
    ...

futures = [process_file(f) for f in files]
results = client.compute(futures)
```

**Alternatives considered:**
- **Prefect/Airflow:** Too high-level, not for low-level task execution
- **Kubernetes Jobs:** Too much overhead, no shared memory
- **Modal:** Vendor lock-in, cost concerns

#### **For RL Training (~10 files):**

**Recommendation: Kubernetes StatefulSets + gRPC**

**Why:**
- Native TPU support in GKE
- Service discovery for named actors
- Health checks & restarts built-in
- Full control over communication
- No Ray overhead

**Architecture:**
```
Kubernetes Services:
├─ train-worker (StatefulSet)
│  └─ Serves weights via gRPC
├─ rollout-worker (StatefulSet, replicas=N)
│  └─ Polls train-worker for weights
└─ curriculum-service (Deployment)
   └─ Coordinates both via gRPC
```

**Example migration:**
```python
# Current: Ray actor
curriculum_actor = ray.get_actor("curriculum_actor")
lesson = ray.get(curriculum_actor.get_lesson.remote())

# After: gRPC service
import grpc
channel = grpc.insecure_channel('curriculum-service:50051')
stub = CurriculumServiceStub(channel)
lesson = stub.GetLesson(GetLessonRequest())
```

**Alternatives considered:**
- **Keep Ray for RL only:** Still have to maintain Ray cluster
- **Custom services without K8s:** More work, no autoscaling/health checks

#### **For TPU Training:**

**Recommendation: GKE Native TPU Podslices**

**Why:**
- First-class TPU support
- Automatic topology management
- Built-in preemption handling
- MEGASCALE coordination automatic
- Delete 1,405 lines of workaround code

**Migration:** Replace `run_on_pod_resumable()` with Kubernetes Job, use JAX directly.

---

### 5.3 Migration Phases

#### **Phase 1: Add Abstraction Layer (2-4 weeks)**

**Goal:** Hide Ray behind interfaces without changing behavior.

**Deliverables:**
- Create `src/marin/distributed/` module with:
  - `ResourceSpec` - platform-agnostic resource requirements
  - `DistributedTask` - abstract task interface
  - `Actor` - abstract actor interface
  - `execute_with_backpressure()` - abstract concurrency control
  - `get()`, `wait()` - abstract futures API

- Migrate 5-10 representative files as proof-of-concept
- All existing code still works (Ray backend)

**Success criteria:** Can swap backends by changing one module.

---

#### **Phase 2: Migrate Batch Processing (4-6 weeks)**

**Goal:** Move 80+ data processing files to Dask.

**Steps:**
1. Set up Dask cluster on GKE
2. Implement Dask backend for `DistributedTask`
3. Migrate file by file:
   - Transform files (24)
   - Crawl files (32)
   - Download files (12)
   - Processing files (12)
4. Migrate Executor framework to use Dask
5. Run both backends in parallel for validation
6. Cutover to Dask, deprecate Ray for batch

**Success criteria:** All batch processing on Dask, RL still on Ray.

**Risk:** Lower than RL migration, tasks are independent.

---

#### **Phase 3: Design RL Service Architecture (3-4 weeks)**

**Goal:** Spec out gRPC services to replace Ray actors.

**Deliverables:**
- gRPC `.proto` definitions:
  - `TrainWorkerService` (serves weights, receives metrics)
  - `RolloutWorkerService` (receives weights, generates rollouts)
  - `CurriculumService` (coordinates lessons, receives stats)
  - `WeightTransferService` (replaces Arrow Flight actor)

- Kubernetes manifests:
  - StatefulSets for train/rollout workers
  - Services for discovery
  - ConfigMaps for shared config

- State management design:
  - Where does curriculum state live?
  - How do we checkpoint?
  - How do we handle failures?

**Success criteria:** Complete design doc, reviewed and approved.

**Risk:** Medium - need to nail the architecture before building.

---

#### **Phase 4: Implement RL Services (6-8 weeks)**

**Goal:** Replace Ray actors with K8s + gRPC.

**Steps:**
1. Implement gRPC services
2. Build Docker images
3. Deploy to GKE
4. Migrate weight transfer (Arrow Flight → gRPC streaming)
5. Test on small RL workload
6. Gradually scale up
7. Run dual backends for validation
8. Cutover to K8s+gRPC

**Success criteria:** RL training on K8s, Ray completely removed.

**Risk:** High - RL is complex, tight coupling, state management.

---

#### **Phase 5: Migrate TPU Code (2-3 weeks)**

**Goal:** Replace custom Ray TPU code with GKE native.

**Steps:**
1. Update Levanter to use GKE TPU Podslices
2. Remove `levanter/infra/ray_tpu.py` (1,405 lines)
3. Update training jobs to use Kubernetes Jobs
4. Test multislice training on GKE
5. Remove all Ray TPU dependencies

**Success criteria:** TPU training uses native GKE, 1400+ lines deleted.

**Risk:** Low - GKE TPU is more mature than Ray's TPU support.

---

#### **Phase 6: Deprecate Ray (1-2 weeks)**

**Goal:** Complete Ray removal.

**Steps:**
1. Remove Ray from `pyproject.toml`
2. Delete `src/marin/distributed/ray_backend.py`
3. Delete Ray cluster infrastructure
4. Update documentation
5. Archive Ray-related code

**Success criteria:** No Ray imports anywhere, cluster decommissioned.

---

### 5.4 Timeline & Effort

| Phase | Duration | Effort (eng-weeks) | Risk |
|-------|----------|-------------------|------|
| 1. Abstraction Layer | 2-4 weeks | 2-3 weeks | Low |
| 2. Batch Migration | 4-6 weeks | 4-5 weeks | Low |
| 3. RL Design | 3-4 weeks | 2-3 weeks | Medium |
| 4. RL Implementation | 6-8 weeks | 6-8 weeks | High |
| 5. TPU Migration | 2-3 weeks | 2-3 weeks | Low |
| 6. Ray Deprecation | 1-2 weeks | 1 week | Low |
| **Total** | **18-27 weeks** | **17-23 weeks** | |

**Assumptions:**
- 1 dedicated senior engineer full-time
- Can run dual backends for validation
- No major architectural changes required
- GKE already available

**Risks:**
- RL migration could take longer (most complex)
- Unforeseen issues with weight transfer latency
- Team learning curve on gRPC/K8s
- Production incidents during migration

---

## 6. Key Recommendations

### Do This ✅

1. **Start with abstraction layer**
   - Lowest risk, immediate value
   - Enables gradual migration
   - Can pause at any point

2. **Migrate batch processing first**
   - 80+ files, but mostly independent
   - Clear win with Dask
   - Reduces Ray cluster load

3. **Fix TPU code with GKE native**
   - Biggest pain point
   - Delete 1,405 lines of workarounds
   - GKE TPU is mature and reliable

4. **Keep Executor framework**
   - Critical feature
   - Just swap backend
   - Don't reinvent workflow orchestration

5. **Use different tools for batch vs RL**
   - They have fundamentally different requirements
   - Don't force one solution

### Don't Do This ❌

1. **Don't migrate RL first**
   - Most complex, highest risk
   - Tight coupling, state management
   - Do batch processing first to learn

2. **Don't big-bang rewrite**
   - Keep Ray working during migration
   - Gradual cutover with validation
   - Fallback plan if issues arise

3. **Don't underestimate RL complexity**
   - Actors, state, coordination are hard
   - Weight transfer latency is critical
   - Allocate 6-8 weeks for implementation

4. **Don't try to use one tool for everything**
   - Batch ≠ RL in requirements
   - Dask good for batch, not for actors
   - K8s+gRPC good for RL, overkill for batch

5. **Don't keep custom TPU code**
   - Ray's TPU support won't get better
   - GKE native is the right solution
   - Delete the hacks

---

## 7. Open Questions

Before starting migration, answer these:

### Technical

1. **Do you already have GKE?**
   - Required for TPU migration
   - Needed for K8s-based RL services
   - What's the setup timeline?

2. **What's your RL training scale?**
   - How many concurrent rollout workers?
   - How often are weights transferred?
   - What's acceptable latency for weight sync?

3. **Can you tolerate higher latency for batch?**
   - Dask may be slower than Ray for some workloads
   - Is seconds to minutes per task acceptable?
   - What's your SLA for data processing?

4. **Where does state live in RL?**
   - Curriculum state persistence strategy?
   - Replay buffer sizing and lifecycle?
   - Checkpoint storage (GCS vs local)?

5. **How do you handle preemption today?**
   - What's your retry strategy?
   - How much state do you lose?
   - What's your checkpoint frequency?

### Organizational

6. **Who owns this migration?**
   - Need dedicated senior engineer for 5-6 months
   - Backup/support from team?
   - On-call during migration?

7. **Can you run dual backends temporarily?**
   - For validation and rollback
   - Doubles infrastructure cost during transition
   - How long can you afford this?

8. **What's the urgency?**
   - Is Ray causing immediate pain?
   - Or is this technical debt cleanup?
   - What's the business driver?

9. **What happens if migration fails?**
   - Fallback plan?
   - How much sunk cost is acceptable?
   - Can you roll back?

10. **What's your risk tolerance?**
    - Can you afford downtime during cutover?
    - What if RL training is broken for a week?
    - What's the blast radius of failure?

---

## 8. Next Steps

### Immediate (Next 2 weeks)

1. **Answer open questions** - get team alignment
2. **Estimate resource availability** - who can work on this?
3. **Set up Dask proof-of-concept** - validate it works for your workload
4. **Design abstraction layer API** - sketch out interfaces
5. **Pick 5 batch files to migrate** - simple, representative samples

### Short-term (Next month)

6. **Implement abstraction layer** - Phase 1 complete
7. **Migrate first batch of files** - validate approach
8. **Measure performance** - Dask vs Ray comparison
9. **Design RL gRPC services** - start architecture work
10. **Investigate GKE TPU setup** - what's required?

### Medium-term (Next quarter)

11. **Migrate all batch processing** - Phase 2 complete
12. **Build RL services** - Phase 4 in progress
13. **Test TPU migration** - Phase 5 validation
14. **Plan Ray deprecation** - timeline for shutdown

---

## 9. Conclusion

**Ray migration is feasible but non-trivial.** Estimated 5-6 months with dedicated engineering effort.

**Key insights:**
- Ray usage falls into two distinct categories (batch vs RL) with different requirements
- Batch processing (80 files) is easier to migrate → Dask recommended
- RL training (10 files) is complex due to actors/state → K8s+gRPC recommended
- TPU code (1,405 lines) should be replaced with GKE native → biggest win
- Abstraction layer enables gradual, low-risk migration

**Success depends on:**
- Dedicated engineering resources
- Ability to run dual backends during transition
- GKE availability for TPU and RL services
- Team buy-in on splitting batch vs RL solutions
- Willingness to invest in gRPC service infrastructure

**Biggest wins:**
1. Delete 1,405 lines of TPU workarounds
2. Simpler infrastructure (Dask < Ray for batch)
3. Native GKE TPU support (better than Ray)
4. Full control over RL services (vs Ray actors)
5. Reduced vendor lock-in (Dask + K8s + gRPC all open source)

**Biggest risks:**
1. RL migration complexity (6-8 weeks, state management)
2. Weight transfer latency with gRPC (vs Ray shared memory)
3. Team learning curve (gRPC, K8s, Dask)
4. Unforeseen integration issues
5. Migration fatigue (80+ files to touch)

**Recommendation:** Proceed with Phase 1 (abstraction layer) immediately. It's low-risk, high-value, and enables all future work. Re-evaluate after Phase 1 complete based on learnings.

---

**Document version:** 1.0
**Last updated:** 2025-10-03
**Author:** Analysis based on codebase deep dive
**Next review:** After Phase 1 completion
