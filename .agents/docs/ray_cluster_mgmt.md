
     Core Insight: API Design Shift

     Current Fray API (wrong abstraction):
     def my_function(ctx: JobContext):
         ...
     cluster.create_job(my_function, RuntimeEnv())  # ❌ Can't serialize arbitrary functions

     Revised Fray API (matches reality):
     # Option 1: Script-based (like ray_run.py does today)
     cluster.create_job(
         entrypoint="python train.py --config config.yaml",
         env=RuntimeEnv(package_requirements=["torch", "jax"])
     )

     # Option 2: Docker-based (future)
     cluster.create_job(
         entrypoint="python train.py",
         env=RuntimeEnv(container_image="gcr.io/my-project/trainer:v1")
     )

     Why this is correct:
     1. ✅ Matches Ray's actual JobSubmissionClient API
     2. ✅ Works with TPUs (already using this pattern in ray_tpu.py)
     3. ✅ Supports Docker workflows (roadmap)
     4. ✅ No function serialization complexity
     5. ✅ Clear separation: create_job (cluster-level) vs create_task (in-cluster)

     Architecture: Two Tiers of Execution

     Tier 1: Cluster Job Submission (create_job)

     Purpose: Submit standalone jobs to a Ray cluster
     Entrypoint: Shell command or Docker image
     Use case: Training runs, batch processing, CI/CD
     Backend support:
     - Ray: ✅ Native via JobSubmissionClient
     - In-memory: ⚠️ Limited (spawn subprocess? or mark unsupported)

     Tier 2: In-Cluster Task Execution (create_task, create_actor)

     Purpose: Programmatic task scheduling within a job
     Entrypoint: Python functions (already working)
     Use case: Dynamic workloads, TPU coordination, parallel processing
     Backend support:
     - Ray: ✅ Working (our recent fixes)
     - In-memory: ✅ Working

     Implementation Plan

     Phase 1: Update Fray Types (Breaking Change)

     Files to modify:
     - lib/fray/src/fray/types.py
     - lib/fray/src/fray/cluster.py

     Changes:
     # Before (types.py)
     EntryPoint = Callable[["JobContext"], None]

     # After (types.py)
     EntryPoint = str  # Shell command like "python script.py"

     # Or support both:
     EntryPoint = str | Callable[["JobContext"], None]

     Update ClusterContext interface:
     class ClusterContext(ABC):
         @abstractmethod
         def create_job(self, entrypoint: str, env: RuntimeEnv) -> str:
             """
             Submit a job to the cluster.

             Args:
                 entrypoint: Shell command to execute (e.g., "python train.py --config foo.yaml")
                 env: Runtime environment (packages, env vars, container, etc.)

             Returns:
                 job_id: Unique identifier for the job

             Example:
                 cluster = RayClusterContext(dashboard_address="http://ray:8265")
                 job_id = cluster.create_job(
                     entrypoint="python train.py --batch-size 32",
                     env=RuntimeEnv(
                         package_requirements=["torch", "wandb"],
                         env={"WANDB_API_KEY": "xxx"}
                     )
                 )
             """
             pass

     Phase 2: Implement Ray Backend (Simple - Just Wrapper)

     Files to modify:
     - lib/fray/src/fray/backend/ray/ray_cluster.py

     Implementation (straightforward wrapper over existing API):
     from ray.job_submission import JobSubmissionClient

     class RayClusterContext(ClusterContext):
         def __init__(self, address: str | None = None, dashboard_address: str | None = None):
             """
             Initialize Ray cluster context.

             Args:
                 address: Ray cluster address for ray.init() (None for local)
                 dashboard_address: Dashboard URL for job submission (defaults to address)
             """
             self._address = address
             self._dashboard_address = dashboard_address or address

             # Initialize Ray runtime (for run_on_tpu, etc.)
             if not ray.is_initialized():
                 ray.init(address=address)

             # Create job submission client (for create_job, list_jobs, delete_job)
             self._job_client = JobSubmissionClient(self._dashboard_address)

         def create_job(self, entrypoint: str, env: RuntimeEnv) -> str:
             """Submit job using Ray JobSubmissionClient."""
             # Build Ray runtime_env from Fray RuntimeEnv
             runtime_env = self._build_ray_runtime_env(env)

             # Add current working directory
             runtime_env["working_dir"] = os.getcwd()
             runtime_env["config"] = {"setup_timeout_seconds": 1800}

             # Generate job ID
             job_id = f"fray-{time.time_ns()}"

             # Submit (exactly like ray_run.py does)
             job_id = self._job_client.submit_job(
                 entrypoint=entrypoint,
                 runtime_env=runtime_env,
                 submission_id=job_id,
             )

             return job_id

         def list_jobs(self) -> list[dict[str, Any]]:
             """List all jobs."""
             jobs = []
             for job_id in self._job_client.list_jobs():
                 info = self._job_client.get_job_info(job_id)
                 jobs.append({
                     "id": job_id,
                     "status": str(info.status),
                     "submission_time": info.submission_time,
                 })
             return jobs

         def delete_job(self, job_id: str) -> None:
             """Stop/delete a job (same as ray_run.py --auto-stop)."""
             self._job_client.stop_job(job_id)

         def _build_ray_runtime_env(self, env: RuntimeEnv) -> dict:
             """Convert Fray RuntimeEnv to Ray runtime_env dict."""
             ray_env = {}

             if env.package_requirements:
                 ray_env["pip"] = env.package_requirements

             if env.env:
                 ray_env["env_vars"] = env.env

             # Future: support container_image
             # if hasattr(env, "container_image"):
             #     ray_env["container"] = {"image": env.container_image}

             return ray_env

     That's it! No function serialization, no complex wrappers. Just a thin layer over Ray's proven API.

     Phase 3: Update/Simplify In-Memory Backend

     Files to modify:
     - lib/fray/src/fray/backend/in_memory.py

     Options:

     Option A: Mark as unsupported (recommended)
     class LocalClusterContext(ClusterContext):
         def create_job(self, entrypoint: str, env: RuntimeEnv) -> str:
             raise NotImplementedError(
                 "In-memory backend doesn't support create_job with string entrypoints. "
                 "Use RayClusterContext for job submission, or create_task for in-process execution."
             )

     Option B: Spawn subprocess (if needed for testing)
     def create_job(self, entrypoint: str, env: RuntimeEnv) -> str:
         """Run entrypoint as subprocess for testing."""
         job_id = f"local-job-{self._job_counter}"
         self._job_counter += 1

         # Spawn subprocess
         def run_subprocess():
             subprocess.run(entrypoint, shell=True, env=env.env)

         thread = threading.Thread(target=run_subprocess, daemon=True)
         thread.start()

         self._jobs[job_id] = {"id": job_id, "status": "running"}
         return job_id

     Recommendation: Option A. The in-memory backend is for unit testing tasks/actors, not cluster job orchestration.

     Phase 4: Move ray_deps.py to Fray (Optional Improvement)

     Files to create:
     - lib/fray/src/fray/backend/ray/ray_deps.py

     Copy from: src/marin/run/ray_deps.py

     Why: Useful utility for building runtime environments with UV dependencies. Can be used by Marin's ray_run.py CLI.

     Migration:
     # Marin code can import from Fray
     from fray.backend.ray.ray_deps import build_runtime_env_for_packages

     runtime_env = build_runtime_env_for_packages(
         extra=["tpu"],
         env_vars={"WANDB_PROJECT": "my-project"}
     )

     Phase 5: Update Tests

     Files to modify:
     - lib/fray/tests/test_backend.py

     Test strategy:

     For Ray backend:
     def test_create_job_with_script(ray_cluster):
         """Test job submission with script entrypoint."""
         cluster = RayClusterContext()

         # Create a simple test script
         script_path = "/tmp/test_script.py"
         Path(script_path).write_text("""
     import sys
     print("Job executed successfully")
     sys.exit(0)
         """)

         job_id = cluster.create_job(
             entrypoint=f"python {script_path}",
             env=RuntimeEnv()
         )

         # Wait for completion
         time.sleep(5)

         # Verify job completed
         jobs = cluster.list_jobs()
         job = next(j for j in jobs if j["id"] == job_id)
         assert "SUCCEEDED" in job["status"]

     def test_delete_job(ray_cluster):
         """Test job deletion."""
         cluster = RayClusterContext()

         # Submit long-running job
         script_path = "/tmp/long_job.py"
         Path(script_path).write_text("""
     import time
     time.sleep(100)
         """)

         job_id = cluster.create_job(
             entrypoint=f"python {script_path}",
             env=RuntimeEnv()
         )

         time.sleep(2)  # Let it start

         # Delete should stop it
         cluster.delete_job(job_id)

         time.sleep(1)

         # Verify stopped
         jobs = cluster.list_jobs()
         job = next(j for j in jobs if j["id"] == job_id)
         assert "STOP" in job["status"]

     For in-memory backend:
     def test_in_memory_create_job_not_supported():
         """In-memory backend doesn't support script-based jobs."""
         cluster = LocalClusterContext()

         with pytest.raises(NotImplementedError, match="doesn't support create_job"):
             cluster.create_job("python script.py", RuntimeEnv())

     Phase 6: Documentation

     Files to create:
     - .agents/docs/ray_cluster_mgmt.md
     - Update lib/fray/README.md

     Key documentation points:

     1. Two-tier execution model:
       - Cluster jobs (create_job): Submit scripts/containers
       - In-cluster tasks (create_task): Dynamic Python functions
     2. When to use which:
       - Use create_job: Training runs, batch jobs, CI/CD pipelines
       - Use create_task: Dynamic parallelism, TPU coordination, interactive workflows
     3. Migration from function-based to script-based:
     # Old pattern (doesn't work for Ray)
     def my_job(ctx):
         ...
     cluster.create_job(my_job, env)

     # New pattern (works everywhere)
     # 1. Put code in script
     with open("my_job.py", "w") as f:
         f.write("""
     from fray import get_job_context

     ctx = get_job_context()
     ref = ctx.create_task(lambda: 42)
     print(ctx.get(ref))
         """)

     # 2. Submit script
     cluster.create_job("python my_job.py", env)
     4. Example: Marin-style usage
     # This is how marin/run/ray_run.py already works!
     cluster = RayClusterContext(dashboard_address="http://ray:8265")

     job_id = cluster.create_job(
         entrypoint="python experiments/train_llama.py --config configs/70b.yaml",
         env=RuntimeEnv(
             package_requirements=["torch", "transformers"],
             env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
         )
     )

     Testing Strategy

     Unit Tests

     test_ray_cluster_context_initialization()
     test_create_job_returns_job_id()
     test_list_jobs_returns_list()
     test_delete_job_stops_running_job()
     test_build_ray_runtime_env_from_fray_env()

     Integration Tests

     test_submit_simple_script_job()
     test_submit_job_with_packages()
     test_submit_job_with_env_vars()
     test_job_lifecycle_submit_wait_delete()
     test_multiple_concurrent_jobs()

     Manual Tests

     1. Submit to real Ray cluster
     2. Verify logs in Ray dashboard
     3. Test job cancellation
     4. Verify TPU resource allocation (if available)

     Migration Impact

     Breaking Changes

     ⚠️ create_job signature changes from:
     create_job(entry_point: Callable, env: RuntimeEnv)
     To:
     create_job(entrypoint: str, env: RuntimeEnv)

     Who's affected?

     Looking at the codebase:
     - test_backend.py: Tests currently skipped for Ray, need updates
     - Marin's ray_run.py: Already uses string entrypoints ✅ (no changes needed!)
     - TPU workflows: Use run_on_tpu() not create_job() ✅ (no changes needed!)

     Impact: Minimal - The broken API (function-based create_job) isn't being used in production.

     Migration Path

     # If anyone was using the old API (unlikely):
     # OLD:
     def my_job(ctx):
         ctx.create_task(...)

     cluster.create_job(my_job, env)

     # NEW: Extract to script
     # my_job.py:
     from fray import get_job_context
     ctx = get_job_context()
     ctx.create_task(...)

     # Submit:
     cluster.create_job("python my_job.py", env)

     Timeline

     - Phase 1 (Types): 1 hour
     - Phase 2 (Ray impl): 2-3 hours (it's mostly copying ray_run.py logic)
     - Phase 3 (In-memory): 30 min (mark NotImplementedError)
     - Phase 4 (ray_deps move): 1 hour (optional)
     - Phase 5 (Tests): 2-3 hours
     - Phase 6 (Docs): 1-2 hours

     Total: 8-11 hours (vs 15-20 hours with function serialization!)

     Success Criteria

     ✅ Ray backend can submit jobs with shell command entrypoints
     ✅ Jobs appear in Ray dashboard with proper status
     ✅ delete_job() stops running jobs
     ✅ Tests pass for Ray backend
     ✅ Marin's ray_run.py can optionally use Fray (backward compatible)
     ✅ Documentation clearly explains two-tier model
     ✅ No regression in run_on_tpu() functionality

     Why This Is Better

     Compared to function serialization:

     1. Simpler: No cloudpickle, no wrapper scripts, no closure handling
     2. Proven: Marin already does this in production
     3. Flexible: Easy to add Docker support later
     4. Debuggable: Scripts are inspectable, functions in pickles are not
     5. Portable: Works across Python versions, environments

     Aligns with real-world usage:

     - ML training: "python train.py --config foo.yaml"
     - Data pipelines: "python etl.py --input s3://..."
     - CI/CD: "python run_tests.py --suite integration"

     Not: "Let me serialize this closure I defined in my Jupyter notebook"
  ⎿
