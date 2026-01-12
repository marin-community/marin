# Implementation Plan: Worker Logs and Job Monitoring

## Overview

Add comprehensive job monitoring and log viewing to the Fluster worker dashboard, including:
- Real-time resource metrics (memory, CPU, processes, disk)
- Build log capture from Docker/UV
- Detailed job view with tabbed logs (stdout, stderr, build)
- Clickable UI to navigate from job list to detailed status

## Current Architecture

### Existing Components
- **WorkerDashboard** (dashboard.py) - Starlette HTTP server with REST API + Web UI + Connect RPC
- **WorkerServiceImpl** (service.py) - RPC handler delegating to JobManager
- **JobManager** (manager.py) - Job lifecycle orchestration (PENDING → BUILDING → RUNNING → terminal)
- **DockerRuntime** (runtime.py) - Container execution with resource limits
- **ImageBuilder** (builder.py) - Docker image building with BuildKit cache

### Current State
- ✓ ResourceUsage proto field exists but **unpopulated**
- ✓ STDOUT/STDERR written to workdir files
- ✗ No BUILD logs captured
- ✗ No resource monitoring during execution
- ✗ Log timestamps incorrect (use read time, not write time)
- ✗ UI shows only basic job list, no detailed view

## Design Decisions

1. **Polling frequency**: 5 seconds (matches dashboard refresh, balances overhead)
2. **Storage**: Current + peak metrics only (avoid unbounded memory growth)
3. **Log timestamps**: Use Docker library's native timestamp support (`container.logs(timestamps=True)`)
4. **Build logs**: Capture Docker build + UV output to BUILD_LOG file
5. **Disk usage**: Track job workdir size (not container filesystem)

## Implementation Phases

### Phase 1: Protocol Buffer Extensions

**File**: `src/fluster/proto/cluster.proto`

Extend ResourceUsage with new fields:
```protobuf
message ResourceUsage {
  int64 memory_mb = 1;           // Current memory usage
  int64 memory_peak_mb = 2;      // NEW: Peak memory
  int64 disk_mb = 3;             // Workdir disk usage
  int32 cpu_millicores = 4;      // CPU millicores
  int32 cpu_percent = 5;         // NEW: CPU percentage 0-100
  int32 process_count = 6;       // NEW: Number of processes
}

message BuildMetrics {
  int64 build_started_ms = 1;
  int64 build_finished_ms = 2;
  bool from_cache = 3;
  string image_tag = 4;
}

message JobStatus {
  // ... existing fields ...
  ResourceUsage resource_usage = 8;  // Now populated
  string status_message = 9;
  BuildMetrics build_metrics = 10;   // NEW
}

message LogEntry {
  int64 timestamp_ms = 1;
  string source = 2;  // "stdout", "stderr", "build"
  string data = 3;
}
```

**Action**: Run `uv run buf generate` to regenerate Python code.

---

### Phase 2: Job State Enhancement

**File**: `src/fluster/cluster/worker/worker_types.py`

Add resource tracking fields to Job dataclass:
```python
@dataclass(kw_only=True)
class Job:
    # ... existing fields ...

    # Resource tracking (NEW)
    current_memory_mb: int = 0
    peak_memory_mb: int = 0
    current_cpu_percent: int = 0
    process_count: int = 0
    disk_mb: int = 0

    # Build tracking (NEW)
    build_started_ms: int | None = None
    build_finished_ms: int | None = None
    build_from_cache: bool = False
    image_tag: str = ""

    # Background tasks (NEW)
    stats_task: asyncio.Task | None = None
    log_streamer_task: asyncio.Task | None = None

    def to_proto(self) -> cluster_pb2.JobStatus:
        """Convert job to JobStatus proto (populate ResourceUsage)."""
        return cluster_pb2.JobStatus(
            # ... existing fields ...
            resource_usage=cluster_pb2.ResourceUsage(
                memory_mb=self.current_memory_mb,
                memory_peak_mb=self.peak_memory_mb,
                disk_mb=self.disk_mb,
                cpu_millicores=self.current_cpu_percent * 10,
                cpu_percent=self.current_cpu_percent,
                process_count=self.process_count,
            ),
            build_metrics=cluster_pb2.BuildMetrics(
                build_started_ms=self.build_started_ms or 0,
                build_finished_ms=self.build_finished_ms or 0,
                from_cache=self.build_from_cache,
                image_tag=self.image_tag,
            ),
        )
```

---

### Phase 3: Resource Monitoring Module

**New File**: `src/fluster/cluster/worker/stats.py`

Create Docker stats collection utilities using Docker Python library:
```python
"""Docker container statistics collection."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import docker


@dataclass
class ContainerStats:
    """Parsed Docker container statistics."""
    memory_mb: int
    cpu_percent: int  # 0-100
    process_count: int
    available: bool  # False if container stopped


async def collect_container_stats(docker_client: docker.DockerClient, container_id: str) -> ContainerStats:
    """Collect resource usage from Docker container using Docker library.

    Uses container.stats(decode=True, stream=False) for single snapshot.

    Args:
        docker_client: Docker client instance
        container_id: Container ID or name

    Returns:
        ContainerStats with current resource usage
    """
    try:
        container = docker_client.containers.get(container_id)

        # Get single stats snapshot (stream=False returns one sample)
        stats = await asyncio.to_thread(
            lambda: container.stats(decode=True, stream=False)
        )

        # Parse memory usage
        memory_stats = stats.get("memory_stats", {})
        memory_usage_bytes = memory_stats.get("usage", 0)
        memory_mb = int(memory_usage_bytes / (1024 * 1024))

        # Parse CPU usage
        cpu_stats = stats.get("cpu_stats", {})
        precpu_stats = stats.get("precpu_stats", {})

        cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - \
                   precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
        system_delta = cpu_stats.get("system_cpu_usage", 0) - \
                      precpu_stats.get("system_cpu_usage", 0)

        num_cpus = cpu_stats.get("online_cpus", 1)
        cpu_percent = 0
        if system_delta > 0 and cpu_delta > 0:
            cpu_percent = int((cpu_delta / system_delta) * num_cpus * 100.0)

        # Parse process count
        pids_stats = stats.get("pids_stats", {})
        process_count = pids_stats.get("current", 0)

        return ContainerStats(
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            process_count=process_count,
            available=True,
        )
    except (docker.errors.NotFound, docker.errors.APIError):
        # Container not found or stopped
        return ContainerStats(0, 0, 0, available=False)


async def collect_workdir_size_mb(workdir: Path) -> int:
    """Calculate workdir size in MB using du command."""
    if not workdir or not workdir.exists():
        return 0

    proc = await asyncio.create_subprocess_exec(
        "du", "-sm", str(workdir),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()

    if proc.returncode == 0:
        return int(stdout.decode().split("\t")[0])
    return 0
```

**Test File**: `tests/cluster/worker/test_stats.py`
- Integration test for collect_container_stats() with invalid container
- Test for collect_workdir_size_mb()
- Mock docker client for unit tests

---

### Phase 4: Log Streaming with Docker Library

**File**: `src/fluster/cluster/worker/manager.py`

Add dependency and log streaming background task:
```python
import docker  # Add to imports

class JobManager:
    def __init__(self, ...):
        # ... existing init ...
        self._docker_client = docker.from_env()

    async def _execute_job(self, job: Job) -> None:
        """Execute job with stats collection and log streaming."""
        async with self._semaphore:
            try:
                # Phase 1: Download bundle
                job.transition_to(cluster_pb2.JOB_STATE_BUILDING, message="downloading bundle")
                job.started_at_ms = int(time.time() * 1000)
                bundle_path = await self._bundle_cache.get_bundle(...)

                # Phase 2: Build image with log capture
                job.transition_to(cluster_pb2.JOB_STATE_BUILDING, message="building image")
                job.build_started_ms = int(time.time() * 1000)

                build_result = await self._image_builder.build(
                    bundle_path=bundle_path,
                    # ... other args ...
                    build_log_file=str(job.workdir / "BUILD_LOG"),  # NEW
                )

                job.build_finished_ms = int(time.time() * 1000)
                job.build_from_cache = build_result.from_cache
                job.image_tag = build_result.image_tag

                # Phase 3: Run container
                job.transition_to(cluster_pb2.JOB_STATE_RUNNING)
                result = await self._runtime.run(config)
                job.container_id = result.container_id

                # Start background tasks (NEW)
                job.stats_task = asyncio.create_task(self._collect_stats_loop(job))
                job.log_streamer_task = asyncio.create_task(self._stream_logs(job))

                # Wait for completion
                await result.wait_task
                job.exit_code = result.exit_code

                # Transition to terminal state
                if result.exit_code == 0:
                    job.transition_to(cluster_pb2.JOB_STATE_SUCCEEDED)
                else:
                    job.transition_to(cluster_pb2.JOB_STATE_FAILED)

            except asyncio.CancelledError:
                job.transition_to(cluster_pb2.JOB_STATE_KILLED)
                raise
            except Exception as e:
                job.transition_to(cluster_pb2.JOB_STATE_FAILED, error=repr(e))
            finally:
                # Stop background tasks (NEW)
                for task in [job.stats_task, job.log_streamer_task]:
                    if task and not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                # ... existing cleanup ...

    async def _collect_stats_loop(self, job: Job) -> None:
        """Poll Docker stats every 5 seconds during execution."""
        while job.status == cluster_pb2.JOB_STATE_RUNNING:
            try:
                if job.container_id:
                    stats = await collect_container_stats(self._docker_client, job.container_id)
                    if stats.available:
                        job.current_memory_mb = stats.memory_mb
                        job.current_cpu_percent = stats.cpu_percent
                        job.process_count = stats.process_count

                        # Update peak memory
                        if stats.memory_mb > job.peak_memory_mb:
                            job.peak_memory_mb = stats.memory_mb

                # Collect disk usage
                if job.workdir:
                    job.disk_mb = await collect_workdir_size_mb(job.workdir)

            except Exception:
                # Don't fail job on stats collection errors
                pass

            await asyncio.sleep(5.0)

    async def _stream_logs(self, job: Job) -> None:
        """Stream container logs to files with timestamps using Docker library."""
        if not job.container_id or not job.workdir:
            return

        try:
            container = self._docker_client.containers.get(job.container_id)

            # Open log files
            stdout_file = (job.workdir / "STDOUT").open("w")
            stderr_file = (job.workdir / "STDERR").open("w")

            # Stream stdout
            async def stream_stdout():
                for line in container.logs(
                    stdout=True, stderr=False, timestamps=True, stream=True
                ):
                    stdout_file.write(line.decode())
                    stdout_file.flush()

            # Stream stderr
            async def stream_stderr():
                for line in container.logs(
                    stdout=False, stderr=True, timestamps=True, stream=True
                ):
                    stderr_file.write(line.decode())
                    stderr_file.flush()

            # Run both streams concurrently
            await asyncio.gather(
                asyncio.to_thread(stream_stdout),
                asyncio.to_thread(stream_stderr),
            )

        except Exception:
            # Container may have stopped
            pass
        finally:
            stdout_file.close()
            stderr_file.close()

    async def get_logs(self, job_id: str, start_line: int = 0) -> list[cluster_pb2.LogEntry]:
        """Get logs with proper timestamp parsing."""
        job = self._jobs.get(job_id)
        if not job or not job.workdir:
            return []

        logs = []

        def read_log_file(filepath: Path, source: str) -> None:
            if not filepath.exists():
                return

            try:
                for line in filepath.read_text().splitlines():
                    if not line:
                        continue

                    # Docker logs format: "2024-01-12T10:30:45.123456789Z message"
                    timestamp_ms = int(time.time() * 1000)  # fallback
                    data = line

                    if " " in line:
                        ts_str, data = line.split(" ", 1)
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            timestamp_ms = int(dt.timestamp() * 1000)
                        except ValueError:
                            data = line  # Keep full line if parse fails

                    logs.append(
                        cluster_pb2.LogEntry(
                            timestamp_ms=timestamp_ms,
                            source=source,
                            data=data,
                        )
                    )
            except Exception:
                pass

        # Read all log sources
        read_log_file(job.workdir / "BUILD_LOG", "build")
        read_log_file(job.workdir / "STDOUT", "stdout")
        read_log_file(job.workdir / "STDERR", "stderr")

        # Sort by timestamp
        logs.sort(key=lambda e: e.timestamp_ms)

        return logs[start_line:]
```

---

### Phase 5: Build Log Capture

**File**: `src/fluster/cluster/worker/builder.py`

Modify ImageBuilder to capture build output:
```python
class ImageBuilder:
    async def build(
        self,
        bundle_path: Path,
        base_image: str,
        extras: list[str],
        job_id: str,
        deps_hash: str,
        build_log_file: str | None = None,  # NEW
    ) -> BuildResult:
        """Build Docker image with optional build log capture."""
        image_tag = f"{self._registry}/fluster-job-{job_id}:{deps_hash[:8]}"

        # Check cache
        if await self._image_exists(image_tag):
            if build_log_file:
                Path(build_log_file).write_text(
                    f"{self._timestamp()} Using cached image: {image_tag}\n"
                )
            return BuildResult(image_tag, deps_hash, 0, from_cache=True)

        # Build with log capture
        start = time.time()
        await self._docker_build(bundle_path, dockerfile, image_tag, build_log_file)
        build_time_ms = int((time.time() - start) * 1000)

        await self._evict_old_images()
        return BuildResult(image_tag, deps_hash, build_time_ms, from_cache=False)

    async def _docker_build(
        self, context: Path, dockerfile: str, tag: str, build_log_file: str | None = None
    ) -> None:
        """Run docker build with optional log capture."""
        dockerfile_path = context / "Dockerfile.fluster"
        dockerfile_path.write_text(dockerfile)

        log_file = None
        if build_log_file:
            Path(build_log_file).parent.mkdir(parents=True, exist_ok=True)
            log_file = open(build_log_file, "w")
            log_file.write(f"{self._timestamp()} Starting Docker build\n")
            log_file.write(f"{self._timestamp()} Image: {tag}\n")

        try:
            cmd = [
                "docker", "build", "-f", str(dockerfile_path), "-t", tag,
                "--build-arg", "BUILDKIT_INLINE_CACHE=1", str(context)
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env={**os.environ, "DOCKER_BUILDKIT": "1"},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            # Stream output to log
            if log_file:
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    log_file.write(f"{self._timestamp()} {line.decode()}")
                    log_file.flush()
            else:
                await proc.communicate()

            if proc.returncode != 0:
                if log_file:
                    log_file.write(f"{self._timestamp()} Build failed: {proc.returncode}\n")
                raise RuntimeError(f"Docker build failed: {proc.returncode}")

            if log_file:
                log_file.write(f"{self._timestamp()} Build completed\n")

        finally:
            if log_file:
                log_file.close()
            dockerfile_path.unlink(missing_ok=True)

    def _timestamp(self) -> str:
        """Generate ISO timestamp for logs."""
        from datetime import datetime
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
```

---

### Phase 6: Dashboard Updates

**File**: `src/fluster/cluster/worker/dashboard.py`

Update dashboard to expose new job detail page:
```python
class WorkerDashboard:
    def _create_app(self) -> Starlette:
        routes = [
            Route("/", self._dashboard),
            Route("/job/{job_id}", self._job_detail_page),  # NEW
            Route("/api/stats", self._stats),
            Route("/api/jobs", self._list_jobs),
            Route("/api/jobs/{job_id}", self._get_job),
            Route("/api/jobs/{job_id}/logs", self._get_logs),
            Mount(rpc_app.path, app=rpc_app),
        ]
        return Starlette(routes=routes)

    async def _job_detail_page(self, request: Request) -> HTMLResponse:
        """Serve detailed job view page."""
        job_id = request.path_params["job_id"]
        return HTMLResponse(JOB_DETAIL_HTML.replace("{{job_id}}", job_id))

    async def _list_jobs(self, request: Request) -> JSONResponse:
        """List all jobs with resource metrics (UPDATE EXISTING)."""
        ctx = FakeRequestContext()
        response = await self._service.list_jobs(cluster_pb2.ListJobsRequest(), ctx)

        jobs = []
        for job in response.jobs:
            jobs.append({
                "job_id": job.job_id,
                "status": self._status_name(job.state),
                "started_at": job.started_at_ms if job.started_at_ms else None,
                "finished_at": job.finished_at_ms if job.finished_at_ms else None,
                "exit_code": job.exit_code if job.exit_code else None,
                "error": job.error if job.error else None,
                "ports": dict(job.ports),
                # Add resource metrics (NEW)
                "memory_mb": job.resource_usage.memory_mb,
                "memory_peak_mb": job.resource_usage.memory_peak_mb,
                "cpu_percent": job.resource_usage.cpu_percent,
                "process_count": job.resource_usage.process_count,
                "disk_mb": job.resource_usage.disk_mb,
                # Add build metrics (NEW)
                "build_from_cache": job.build_metrics.from_cache,
                "build_image_tag": job.build_metrics.image_tag,
            })

        return JSONResponse(jobs)

    async def _get_job(self, request: Request) -> JSONResponse:
        """Get single job details with metrics (UPDATE EXISTING)."""
        job_id = request.path_params["job_id"]
        ctx = FakeRequestContext()

        try:
            job = await self._service.get_job_status(
                cluster_pb2.GetStatusRequest(job_id=job_id), ctx
            )
        except Exception:
            return JSONResponse({"error": "Not found"}, status_code=404)

        return JSONResponse({
            "job_id": job.job_id,
            "status": self._status_name(job.state),
            "started_at": job.started_at_ms if job.started_at_ms else None,
            "finished_at": job.finished_at_ms if job.finished_at_ms else None,
            "exit_code": job.exit_code if job.exit_code else None,
            "error": job.error if job.error else None,
            "ports": dict(job.ports),
            # Add resource metrics (NEW)
            "resources": {
                "memory_mb": job.resource_usage.memory_mb,
                "memory_peak_mb": job.resource_usage.memory_peak_mb,
                "cpu_percent": job.resource_usage.cpu_percent,
                "disk_mb": job.resource_usage.disk_mb,
                "process_count": job.resource_usage.process_count,
            },
            # Add build metrics (NEW)
            "build": {
                "started_ms": job.build_metrics.build_started_ms,
                "finished_ms": job.build_metrics.build_finished_ms,
                "duration_ms": (
                    job.build_metrics.build_finished_ms - job.build_metrics.build_started_ms
                    if job.build_metrics.build_started_ms else 0
                ),
                "from_cache": job.build_metrics.from_cache,
                "image_tag": job.build_metrics.image_tag,
            },
        })

    async def _get_logs(self, request: Request) -> JSONResponse:
        """Get job logs (UPDATE EXISTING to support source filtering)."""
        job_id = request.path_params["job_id"]
        tail = request.query_params.get("tail")
        source = request.query_params.get("source")  # NEW: optional source filter

        start_line = -int(tail) if tail else 0

        ctx = FakeRequestContext()
        log_filter = cluster_pb2.FetchLogsFilter(start_line=start_line)

        try:
            response = await self._service.fetch_logs(
                cluster_pb2.FetchLogsRequest(job_id=job_id, filter=log_filter), ctx
            )
        except Exception:
            return JSONResponse({"error": "Not found"}, status_code=404)

        logs = [
            {
                "timestamp": entry.timestamp_ms,
                "source": entry.source,
                "data": entry.data,
            }
            for entry in response.logs
        ]

        # Filter by source if specified (NEW)
        if source:
            logs = [log for log in logs if log["source"] == source]

        return JSONResponse(logs)
```

**Key Changes:**
- No new REST endpoints - reuse existing `/api/jobs` and `/api/jobs/{job_id}` and `/api/jobs/{job_id}/logs`
- Extended existing endpoints to include resource and build metrics from proto
- Added optional `?source=stdout|stderr|build` query param to logs endpoint
- All data comes from proto messages returned by RPC methods
- Dashboard just converts proto→JSON systematically

---

### Phase 7: UI Enhancements

**File**: `src/fluster/cluster/worker/dashboard.py`

Update HTML templates:

**Main Dashboard** - Add clickable job IDs and metrics columns:
```python
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Fluster Worker</title>
  <style>
    body { font-family: sans-serif; margin: 20px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #4CAF50; color: white; }
    tr:nth-child(even) { background-color: #f2f2f2; }
    .status-running { color: blue; font-weight: bold; }
    .status-succeeded { color: green; }
    .status-failed { color: red; font-weight: bold; }
    .job-link { color: #0066cc; cursor: pointer; text-decoration: underline; }
  </style>
</head>
<body>
  <h1>Fluster Worker Dashboard</h1>
  <div id="stats"></div>
  <h2>Jobs</h2>
  <table id="jobs">
    <tr>
      <th>ID</th><th>Status</th><th>Memory</th><th>CPU</th><th>Processes</th>
      <th>Started</th><th>Finished</th><th>Error</th>
    </tr>
  </table>
  <script>
    async function refresh() {
      const stats = await fetch('/api/stats').then(r => r.json());
      document.getElementById('stats').innerHTML =
        `<b>Running:</b> ${stats.running} | <b>Pending:</b> ${stats.pending} | ` +
        `<b>Building:</b> ${stats.building} | <b>Completed:</b> ${stats.completed}`;

      const jobs = await fetch('/api/jobs').then(r => r.json());
      const tbody = jobs.map(j => {
        const started = j.started_at ? new Date(j.started_at).toLocaleString() : '-';
        const finished = j.finished_at ? new Date(j.finished_at).toLocaleString() : '-';
        const shortId = j.job_id.slice(0, 8);

        return `<tr>
          <td><a href="/job/${j.job_id}" target="_blank" class="job-link">${shortId}...</a></td>
          <td class="status-${j.status}">${j.status}</td>
          <td>${j.memory_mb || 0} / ${j.memory_peak_mb || 0} MB</td>
          <td>${j.cpu_percent || 0}%</td>
          <td>${j.process_count || 0}</td>
          <td>${started}</td>
          <td>${finished}</td>
          <td>${j.error || '-'}</td>
        </tr>`;
      }).join('');

      document.getElementById('jobs').innerHTML =
        '<tr><th>ID</th><th>Status</th><th>Memory</th><th>CPU</th><th>Processes</th>' +
        '<th>Started</th><th>Finished</th><th>Error</th></tr>' + tbody;
    }

    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""
```

**Job Detail Page** - Tabbed log viewer with metrics:
```python
JOB_DETAIL_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Job {{job_id}} - Fluster Worker</title>
  <style>
    body { font-family: sans-serif; margin: 20px; }
    .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
    .status-running { color: blue; font-weight: bold; }
    .status-succeeded { color: green; font-weight: bold; }
    .status-failed { color: red; font-weight: bold; }
    .tabs { display: flex; border-bottom: 2px solid #4CAF50; margin-bottom: 0; }
    .tab { padding: 10px 20px; cursor: pointer; background: #f0f0f0; margin-right: 2px;
           border: 1px solid #ddd; border-bottom: none; border-radius: 5px 5px 0 0; }
    .tab.active { background: #4CAF50; color: white; border-color: #4CAF50; }
    .tab-content { display: none; padding: 15px; border: 1px solid #ddd;
                   max-height: 500px; overflow-y: auto; background: #f9f9f9; }
    .tab-content.active { display: block; }
    .log-line { font-family: monospace; font-size: 12px; padding: 2px 0; white-space: pre-wrap; }
    .log-stdout { color: #000; }
    .log-stderr { color: #d32f2f; }
    .log-build { color: #1976d2; }
    .metrics { display: flex; gap: 30px; flex-wrap: wrap; }
    .metric { text-align: center; }
    .metric-label { font-weight: bold; color: #666; font-size: 14px; }
    .metric-value { font-size: 32px; color: #4CAF50; margin: 5px 0; }
    .metric-unit { font-size: 14px; color: #999; }
  </style>
</head>
<body>
  <h1>Job Details: <code id="job-id"></code></h1>
  <a href="/">← Back to Dashboard</a>

  <div class="section">
    <h2>Status: <span id="status"></span></h2>
    <div id="status-details"></div>
  </div>

  <div class="section">
    <h2>Resource Usage</h2>
    <div class="metrics" id="resources"></div>
  </div>

  <div class="section">
    <h2>Build Information</h2>
    <div id="build-info"></div>
  </div>

  <div class="section">
    <h2>Logs</h2>
    <div class="tabs">
      <div class="tab active" data-tab="all">ALL</div>
      <div class="tab" data-tab="stdout">STDOUT</div>
      <div class="tab" data-tab="stderr">STDERR</div>
      <div class="tab" data-tab="build">BUILD</div>
    </div>
    <div id="log-all" class="tab-content active"></div>
    <div id="log-stdout" class="tab-content"></div>
    <div id="log-stderr" class="tab-content"></div>
    <div id="log-build" class="tab-content"></div>
  </div>

  <script>
    const jobId = "{{job_id}}";

    async function refresh() {
      // Fetch job status with metrics
      const job = await fetch(`/api/jobs/${jobId}`).then(r => r.json());
      document.getElementById('job-id').textContent = jobId;
      document.getElementById('status').innerHTML =
        `<span class="status-${job.status}">${job.status}</span>`;

      const started = job.started_at ? new Date(job.started_at).toLocaleString() : '-';
      const finished = job.finished_at ? new Date(job.finished_at).toLocaleString() : '-';
      document.getElementById('status-details').innerHTML = `
        <p><b>Started:</b> ${started}</p>
        <p><b>Finished:</b> ${finished}</p>
        <p><b>Exit Code:</b> ${job.exit_code !== null ? job.exit_code : '-'}</p>
        <p><b>Error:</b> ${job.error || '-'}</p>
        <p><b>Ports:</b> ${JSON.stringify(job.ports)}</p>
      `;

      // Resource metrics (from job.resources)
      document.getElementById('resources').innerHTML = `
        <div class="metric">
          <div class="metric-label">Memory</div>
          <div class="metric-value">${job.resources.memory_mb}</div>
          <div class="metric-unit">MB (Peak: ${job.resources.memory_peak_mb} MB)</div>
        </div>
        <div class="metric">
          <div class="metric-label">CPU</div>
          <div class="metric-value">${job.resources.cpu_percent}</div>
          <div class="metric-unit">%</div>
        </div>
        <div class="metric">
          <div class="metric-label">Processes</div>
          <div class="metric-value">${job.resources.process_count}</div>
          <div class="metric-unit">running</div>
        </div>
        <div class="metric">
          <div class="metric-label">Disk</div>
          <div class="metric-value">${job.resources.disk_mb}</div>
          <div class="metric-unit">MB</div>
        </div>
      `;

      // Build metrics (from job.build)
      const buildDuration = job.build.duration_ms > 0
        ? (job.build.duration_ms / 1000).toFixed(2) + 's'
        : '-';
      document.getElementById('build-info').innerHTML = `
        <p><b>Image:</b> <code>${job.build.image_tag || '-'}</code></p>
        <p><b>Build Time:</b> ${buildDuration}</p>
        <p><b>From Cache:</b> ${job.build.from_cache ? 'Yes ✓' : 'No'}</p>
      `;

      // Fetch logs
      const allLogs = await fetch(`/api/jobs/${jobId}/logs`).then(r => r.json());

      // Separate by source
      const stdoutLogs = allLogs.filter(l => l.source === 'stdout');
      const stderrLogs = allLogs.filter(l => l.source === 'stderr');
      const buildLogs = allLogs.filter(l => l.source === 'build');

      function renderLogs(logs, elementId) {
        const html = logs.map(log => {
          const ts = new Date(log.timestamp).toLocaleTimeString();
          return `<div class="log-line log-${log.source}">[${ts}] ${escapeHtml(log.data)}</div>`;
        }).join('');
        document.getElementById(elementId).innerHTML = html || '<p>No logs available</p>';
      }

      renderLogs(stdoutLogs, 'log-stdout');
      renderLogs(stderrLogs, 'log-stderr');
      renderLogs(buildLogs, 'log-build');
      renderLogs(allLogs, 'log-all');
    }

    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const tabName = tab.getAttribute('data-tab');
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`log-${tabName}`).classList.add('active');
      });
    });

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""
```

Update `_list_jobs()` to include resource metrics in job list response.

---

### Phase 8: Integration Tests

**File**: `tests/cluster/worker/test_worker_integration_full.py`

Add new test cases:
```python
@pytest.mark.asyncio
@pytest.mark.slow
async def test_resource_monitoring(worker_server, workspace_bundle):
    """Test that resource metrics are collected during job execution."""
    _dashboard, client, _port = worker_server

    def memory_allocator():
        import time
        # Allocate ~10MB
        data = [0] * (10 * 1024 * 1024)
        print(f"Allocated {len(data)} items")
        time.sleep(10)  # Keep running for stats collection
        return len(data)

    entrypoint = Entrypoint(callable=memory_allocator, args=(), kwargs={})
    request = cluster_pb2.RunJobRequest(
        job_id="test-resources",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
    )

    await client.run_job(request)
    await wait_for_job_state(client, "test-resources", {cluster_pb2.JOB_STATE_RUNNING})

    # Wait for stats collection (poll interval is 5s)
    await asyncio.sleep(6)

    status = await client.get_job_status(cluster_pb2.GetStatusRequest(job_id="test-resources"))

    # Verify resource metrics populated
    assert status.resource_usage.memory_mb > 0
    assert status.resource_usage.process_count > 0
    assert status.resource_usage.peak_memory_mb >= status.resource_usage.memory_mb

    # Wait for completion
    await wait_for_job_state(
        client, "test-resources",
        {cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED}
    )


@pytest.mark.asyncio
@pytest.mark.slow
async def test_build_log_capture(worker_server, workspace_bundle):
    """Test that build logs are captured and retrievable."""
    _dashboard, client, _port = worker_server

    entrypoint = create_entrypoint_simple()
    request = cluster_pb2.RunJobRequest(
        job_id="test-build-logs",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
    )

    await client.run_job(request)
    await wait_for_job_state(
        client, "test-build-logs",
        {cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED}
    )

    # Fetch logs
    logs_response = await client.fetch_logs(
        cluster_pb2.FetchLogsRequest(
            job_id="test-build-logs",
            filter=cluster_pb2.FetchLogsFilter(),
        )
    )

    # Verify build logs exist
    build_logs = [log for log in logs_response.logs if log.source == "build"]
    assert len(build_logs) > 0

    # Build logs should mention image/docker/cache
    build_text = " ".join(log.data.lower() for log in build_logs)
    assert any(word in build_text for word in ["image", "docker", "cache", "build"])


@pytest.mark.asyncio
@pytest.mark.slow
async def test_log_timestamps_accurate(worker_server, workspace_bundle):
    """Test that log timestamps reflect actual write times."""
    _dashboard, client, _port = worker_server

    def timestamped_logger():
        import time
        print("First log")
        time.sleep(2)
        print("Second log after 2s")
        return "done"

    entrypoint = Entrypoint(callable=timestamped_logger, args=(), kwargs={})
    request = cluster_pb2.RunJobRequest(
        job_id="test-timestamps",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
    )

    await client.run_job(request)
    await wait_for_job_state(
        client, "test-timestamps",
        {cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED}
    )

    logs_response = await client.fetch_logs(
        cluster_pb2.FetchLogsRequest(
            job_id="test-timestamps",
            filter=cluster_pb2.FetchLogsFilter(),
        )
    )

    stdout_logs = [log for log in logs_response.logs if log.source == "stdout"]
    assert len(stdout_logs) >= 2

    # Timestamps should be ~2 seconds apart
    time_diff_ms = stdout_logs[1].timestamp_ms - stdout_logs[0].timestamp_ms
    assert 1500 < time_diff_ms < 3000  # Allow some variance
```

---

## Critical Files

### To Modify
1. `src/fluster/proto/cluster.proto` - Extend ResourceUsage, add BuildMetrics
2. `src/fluster/cluster/worker/worker_types.py` - Add resource/build fields to Job
3. `src/fluster/cluster/worker/manager.py` - Stats collection, log streaming, build metrics
4. `src/fluster/cluster/worker/builder.py` - Build log capture
5. `src/fluster/cluster/worker/dashboard.py` - New endpoints, UI updates

### To Create
6. `src/fluster/cluster/worker/stats.py` - Stats collection utilities
7. `tests/cluster/worker/test_stats.py` - Unit tests for stats module

### To Extend
8. `tests/cluster/worker/test_worker_integration_full.py` - Integration tests

---

## Dependencies

Add to pyproject.toml:
```toml
[project]
dependencies = [
    # ... existing ...
    "docker>=7.0.0",  # NEW: For container.logs() with timestamps
]
```

Run `uv sync` after adding.

---

## Verification Steps

After implementation:

1. **Proto regeneration**: Run `uv run buf generate`, verify no errors
2. **Unit tests**: Run `uv run pytest tests/cluster/worker/test_stats.py -v`
3. **Integration tests**: Run `uv run pytest tests/cluster/worker/test_worker_integration_full.py -v -k "resource_monitoring or build_log or timestamps"`
4. **Manual verification**:
   - Start worker: `uv run python -m fluster.cluster.worker`
   - Submit test job via WorkerContext
   - Open browser to http://localhost:{port}
   - Verify job list shows memory/CPU metrics
   - Click job ID, verify detail page loads
   - Verify metrics display (memory, CPU, processes, disk)
   - Click log tabs (ALL, STDOUT, STDERR, BUILD)
   - Verify logs display with timestamps
   - Verify build logs show Docker output or cache message
5. **Full test suite**: Run `uv run pytest tests/cluster/worker/ -v`

---

## Notes

- **Docker library**: Using Docker Python library throughout for both `container.stats()` and `container.logs(timestamps=True)` provides accurate timestamps and resource metrics without shell commands
- **Background tasks**: Stats collection and log streaming run as separate asyncio tasks, cancelled on job completion
- **Error handling**: Stats/log collection errors don't fail the job (defensive programming)
- **Memory management**: Current+peak only (no time series) to avoid unbounded growth. Existing TODO about LRU eviction remains.
- **Disk usage**: Computed on-demand via `du` command (efficient, no continuous monitoring needed)
- **Build logs**: Captured during docker build phase, written with timestamps
- **UI refresh**: 5-second polling matches stats collection interval
- **Backward compatibility**: New proto fields have defaults, won't break existing code
- **API structure**: No new REST endpoints - extended existing `/api/jobs` and `/api/jobs/{job_id}` to include resource and build metrics from proto. Added optional `?source=` query param to logs endpoint for filtering.

---

## Future Enhancements (Out of Scope)

- Real-time log streaming (WebSocket)
- Historical resource graphs (time series)
- LRU eviction for completed jobs
- Log rotation/compression
- Worker-level resource monitoring (across all jobs)
- Alert thresholds for resource usage
