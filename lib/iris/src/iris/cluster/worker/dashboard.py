# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP dashboard with Connect RPC and web UI for worker monitoring."""

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.dashboard_common import html_shell, logs_api_response, logs_page_response, static_files_mount
from iris.logging import LogBuffer
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import WorkerServiceWSGIApplication
from iris.rpc.proto_utils import task_state_name


class FakeRequestContext:
    """Minimal stub RequestContext for internal REST-to-RPC bridging.

    RPC methods require a RequestContext parameter but never access it.
    """

    pass


class WorkerDashboard:
    """HTTP dashboard with Connect RPC and web UI."""

    def __init__(
        self,
        service: WorkerServiceImpl,
        host: str = "0.0.0.0",
        port: int = 8080,
        log_buffer: LogBuffer | None = None,
    ):
        self._service = service
        self._host = host
        self._port = port
        self._log_buffer = log_buffer
        self._app = self._create_app()
        self._server: uvicorn.Server | None = None

    @property
    def port(self) -> int:
        return self._port

    def _create_app(self) -> Starlette:
        rpc_wsgi_app = WorkerServiceWSGIApplication(service=self._service)
        rpc_app = WSGIMiddleware(rpc_wsgi_app)

        routes = [
            # Health check (for bootstrap and load balancers)
            Route("/health", self._health),
            # Web dashboard
            Route("/", self._dashboard),
            Route("/task/{task_id:path}", self._task_detail_page),
            Route("/logs", self._logs_page),
            # REST API (for dashboard)
            # Note: logs route must come before get_task to avoid {task_id:path} matching "task-id/logs"
            Route("/api/stats", self._stats),
            Route("/api/tasks", self._list_tasks),
            Route("/api/logs", self._api_logs),
            Route("/api/tasks/{task_id:path}/logs", self._get_logs),
            Route("/api/tasks/{task_id:path}", self._get_task),
            # Static files (JS/CSS for Preact components)
            static_files_mount(),
            # Connect RPC - mount WSGI app wrapped for ASGI
            Mount(rpc_wsgi_app.path, app=rpc_app),
        ]
        return Starlette(routes=routes)

    def _logs_page(self, request: Request) -> HTMLResponse:
        return logs_page_response(request)

    def _api_logs(self, request: Request):
        return logs_api_response(request, self._log_buffer)

    def _health(self, _request: Request) -> JSONResponse:
        """Simple health check endpoint for bootstrap and load balancers."""
        return JSONResponse({"status": "healthy"})

    def _dashboard(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Iris Worker", "/static/worker/app.js"))

    def _task_detail_page(self, request: Request) -> HTMLResponse:
        return HTMLResponse(html_shell("Task Detail", "/static/worker/task-detail.js"))

    def _stats(self, _request: Request) -> JSONResponse:
        ctx = FakeRequestContext()
        response = self._service.list_tasks(cluster_pb2.Worker.ListTasksRequest(), ctx)
        tasks = response.tasks

        return JSONResponse(
            {
                "running": sum(1 for t in tasks if t.state == cluster_pb2.TASK_STATE_RUNNING),
                "pending": sum(1 for t in tasks if t.state == cluster_pb2.TASK_STATE_PENDING),
                "building": sum(1 for t in tasks if t.state == cluster_pb2.TASK_STATE_BUILDING),
                "completed": sum(
                    1
                    for t in tasks
                    if t.state
                    in (
                        cluster_pb2.TASK_STATE_SUCCEEDED,
                        cluster_pb2.TASK_STATE_FAILED,
                        cluster_pb2.TASK_STATE_KILLED,
                    )
                ),
            }
        )

    def _list_tasks(self, _request: Request) -> JSONResponse:
        ctx = FakeRequestContext()
        response = self._service.list_tasks(cluster_pb2.Worker.ListTasksRequest(), ctx)
        tasks = response.tasks

        return JSONResponse(
            [
                {
                    "task_id": t.task_id,
                    "job_id": t.job_id,
                    "task_index": t.task_index,
                    "attempt_id": t.current_attempt_id,
                    "status": task_state_name(t.state),
                    "started_at": t.started_at_ms,
                    "finished_at": t.finished_at_ms,
                    "exit_code": t.exit_code,
                    "error": t.error,
                    # Add resource metrics
                    "memory_mb": t.resource_usage.memory_mb,
                    "memory_peak_mb": t.resource_usage.memory_peak_mb,
                    "cpu_percent": t.resource_usage.cpu_percent,
                    "process_count": t.resource_usage.process_count,
                    "disk_mb": t.resource_usage.disk_mb,
                    # Add build metrics
                    "build_from_cache": t.build_metrics.from_cache,
                    "image_tag": t.build_metrics.image_tag,
                }
                for t in tasks
            ]
        )

    def _get_task(self, request: Request) -> JSONResponse:
        task_id = request.path_params["task_id"]
        ctx = FakeRequestContext()
        try:
            task = self._service.get_task_status(cluster_pb2.Worker.GetTaskStatusRequest(task_id=task_id), ctx)
        except Exception:
            # RPC raises ConnectError with NOT_FOUND for missing tasks
            return JSONResponse({"error": "Not found"}, status_code=404)

        return JSONResponse(
            {
                "task_id": task.task_id,
                "job_id": task.job_id,
                "task_index": task.task_index,
                "attempt_id": task.current_attempt_id,
                "status": task_state_name(task.state),
                "started_at": task.started_at_ms,
                "finished_at": task.finished_at_ms,
                "exit_code": task.exit_code,
                "error": task.error,
                "ports": dict(task.ports),
                "resources": {
                    "memory_mb": task.resource_usage.memory_mb,
                    "memory_peak_mb": task.resource_usage.memory_peak_mb,
                    "cpu_percent": task.resource_usage.cpu_percent,
                    "disk_mb": task.resource_usage.disk_mb,
                    "process_count": task.resource_usage.process_count,
                },
                "build": {
                    "started_ms": task.build_metrics.build_started_ms,
                    "finished_ms": task.build_metrics.build_finished_ms,
                    "duration_ms": (
                        (task.build_metrics.build_finished_ms - task.build_metrics.build_started_ms)
                        if task.build_metrics.build_started_ms
                        else 0
                    ),
                    "from_cache": task.build_metrics.from_cache,
                    "image_tag": task.build_metrics.image_tag,
                },
            }
        )

    def _get_logs(self, request: Request) -> JSONResponse:
        task_id = request.path_params["task_id"]
        tail = request.query_params.get("tail")
        start_line = -int(tail) if tail else 0
        source = request.query_params.get("source")
        ctx = FakeRequestContext()
        log_filter = cluster_pb2.Worker.FetchLogsFilter(start_line=start_line)
        try:
            response = self._service.fetch_task_logs(
                cluster_pb2.Worker.FetchTaskLogsRequest(task_id=task_id, filter=log_filter), ctx
            )
        except Exception:
            # RPC raises ConnectError with NOT_FOUND for missing tasks
            return JSONResponse({"error": "Not found"}, status_code=404)

        logs = [
            {
                "timestamp": entry.timestamp_ms,
                "source": entry.source,
                "data": entry.data,
            }
            for entry in response.logs
        ]

        # Apply source filter if specified
        if source:
            logs = [log for log in logs if log["source"] == source]

        return JSONResponse(logs)

    def run(self) -> None:
        import uvicorn

        uvicorn.run(self._app, host=self._host, port=self._port)

    async def run_async(self) -> None:
        import uvicorn

        config = uvicorn.Config(self._app, host=self._host, port=self._port)
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def shutdown(self) -> None:
        if self._server:
            self._server.should_exit = True
