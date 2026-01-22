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

"""Flask proxy and UI for multiplexing multiple Ray dashboards."""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from flask import Flask, Response, request
import requests
from werkzeug.serving import make_server

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """A single log entry for the dashboard."""

    timestamp: datetime
    cluster: str
    level: str  # "error", "warning", "info"
    message: str
    details: str | None = None  # Full stack trace or additional details


@dataclass
class ResourceUsage:
    """Resource usage information for a single resource type."""

    used: str
    total: str

    def percentage(self) -> float:
        """Calculate usage percentage, handling unit conversions."""
        try:
            used_val = float(self.used.replace("TiB", "").replace("GiB", "").replace("MiB", "").replace("KiB", ""))
            total_val = float(self.total.replace("TiB", "").replace("GiB", "").replace("MiB", "").replace("KiB", ""))
            return (used_val / total_val * 100) if total_val > 0 else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0


@dataclass
class ClusterStatus:
    """Parsed Ray cluster status information."""

    active_nodes: list[str] = field(default_factory=list)
    pending_nodes: list[str] = field(default_factory=list)
    resources: dict[str, ResourceUsage] = field(default_factory=dict)

    def active_count(self) -> int:
        """Number of active nodes."""
        return len(self.active_nodes)

    def pending_count(self) -> int:
        """Number of pending nodes."""
        return len(self.pending_nodes)


@dataclass
class RayPortMapping:
    """Local port mappings for SSH tunnel to a Ray cluster."""

    dashboard_port: int
    gcs_port: int
    api_port: int


@dataclass
class ClusterInfo:
    """Information about a Ray cluster."""

    cluster_name: str
    config_path: str
    head_ip: str
    external_ip: str | None
    zone: str
    project: str


def format_number(value_str: str) -> str:
    """Format numbers compactly (e.g., 61680.0 -> 61.7k, 2049.0 -> 2049)."""
    unit = ""
    clean_str = value_str
    for suffix in ["TiB", "GiB", "MiB", "KiB"]:
        if value_str.endswith(suffix):
            unit = suffix
            clean_str = value_str[: -len(suffix)]
            break

    try:
        num = float(clean_str)

        if not unit:
            if num >= 1_000_000:
                formatted = f"{num / 1_000_000:.1f}"
                formatted = formatted.rstrip("0").rstrip(".")
                return f"{formatted}M"
            if num >= 10_000:
                formatted = f"{num / 1_000:.1f}"
                formatted = formatted.rstrip("0").rstrip(".")
                return f"{formatted}k"

        if num == int(num):
            return f"{int(num)}{unit}"

        formatted = f"{num:.1f}"
        formatted = formatted.rstrip("0").rstrip(".")
        return f"{formatted}{unit}"
    except ValueError:
        return value_str


def parse_ray_status(status_output: str) -> ClusterStatus:
    """Parse ray status output into structured data."""
    result = ClusterStatus()
    lines = status_output.strip().split("\n")
    current_section: str | None = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Node status"):
            current_section = "nodes"
        elif line.startswith("Resources"):
            current_section = "resources"
        elif line.startswith("Active:"):
            current_section = "active_nodes"
        elif line.startswith("Pending:"):
            current_section = "pending_nodes"
        elif line.startswith("Usage:"):
            current_section = "usage"
        elif line.startswith(("Constraints:", "Demands:", "Recent failures:")):
            current_section = None
        elif current_section == "active_nodes" and not line.startswith(("-", "=")):
            if line != "(no active nodes)":
                result.active_nodes.append(line)
        elif current_section == "pending_nodes" and not line.startswith(("-", "=")):
            if line != "(no pending nodes)":
                result.pending_nodes.append(line)
        elif current_section == "usage":
            match = re.match(r"([\d.]+[KMGTB]?i?B?)/([\d.]+[KMGTB]?i?B?)\s+(.+)", line)
            if match:
                used, total, resource_name = match.groups()
                if resource_name == "object_store_memory":
                    continue
                if re.search("ray.*worker", resource_name):
                    continue
                result.resources[resource_name] = ResourceUsage(used=used, total=total)
    return result


@dataclass
class DashboardProxy:
    """Thread-based Flask proxy for multiple Ray dashboards."""

    clusters: dict[str, ClusterInfo]
    port_mappings: dict[str, RayPortMapping]
    proxy_port: int
    server: make_server | None = None
    thread: threading.Thread | None = None
    _logs: deque[LogEntry] = field(default_factory=lambda: deque(maxlen=100))
    _logs_lock: threading.Lock = field(default_factory=threading.Lock)

    def _add_log(self, cluster: str, level: str, message: str, details: str | None = None) -> None:
        """Add a log entry thread-safely."""
        entry = LogEntry(
            timestamp=datetime.now(),
            cluster=cluster,
            level=level,
            message=message,
            details=details,
        )
        with self._logs_lock:
            self._logs.append(entry)

    def _get_logs(self, limit: int = 50) -> list[LogEntry]:
        """Get recent logs thread-safely."""
        with self._logs_lock:
            return list(self._logs)[-limit:]

    def _build_status(self, cluster: str, ports: RayPortMapping) -> ClusterStatus:
        """Fetch status for a cluster using the forwarded port."""
        try:
            gcs_address = f"localhost:{ports.gcs_port}"
            result = subprocess.run(
                ["ray", "status", f"--address={gcs_address}"],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            return parse_ray_status(result.stdout)
        except subprocess.CalledProcessError as e:
            output = e.stderr or e.stdout or ""
            logger.error("ray status failed for %s (exit %s): %s", cluster, e.returncode, output.strip())
            raise
        except Exception:
            logger.exception("Failed to fetch status for %s", cluster)
            raise

    def _render_resources(self, status: ClusterStatus) -> str:
        if not status.resources:
            return ""

        html = '<div class="resources">'
        for name, usage in status.resources.items():
            percent = usage.percentage()
            used_fmt = format_number(usage.used)
            total_fmt = format_number(usage.total)
            html += '<div class="resource-bar">'
            html += f'<div class="resource-label">{name}: {used_fmt}/{total_fmt} ({percent:.1f}%)</div>'
            html += '<div class="bar">'
            html += f'<div class="bar-fill" style="width: {percent}%"></div>'
            html += "</div>"
            html += "</div>"
        html += "</div>"
        return html

    def _build_status_html(self, cluster: str, ports: RayPortMapping) -> str:
        """Render a status card body for HTMX."""
        try:
            status = self._build_status(cluster, ports)

            html = '<div class="status-content">'
            html += f'<div class="stat"><strong>Nodes:</strong> {status.active_count()} active'
            if status.pending_count() > 0:
                html += f", {status.pending_count()} pending"
            html += "</div>"
            html += self._render_resources(status)
            html += "</div>"
            return html
        except subprocess.TimeoutExpired as e:
            self._add_log(
                cluster,
                "error",
                "Timeout fetching status",
                f"Command timed out after {e.timeout}s: {' '.join(e.cmd) if e.cmd else 'unknown command'}",
            )
            return '<div class="error">⏱ Timeout fetching status</div>'
        except subprocess.CalledProcessError as e:
            output = (e.stderr or e.stdout or "").strip()
            self._add_log(
                cluster,
                "error",
                f"ray status failed (exit {e.returncode})",
                output or "no output",
            )
            return '<div class="error">❌ Failed to fetch status</div>'
        except Exception as e:
            self._add_log(
                cluster,
                "error",
                f"Error: {str(e)}",
                traceback.format_exc(),
            )
            return '<div class="error">❌ Failed to fetch status</div>'

    def _create_app(self) -> Flask:
        app = Flask(__name__)

        @app.route("/api/cluster/<cluster>/status-html")
        def cluster_status_html(cluster: str):
            """Get cluster status as HTML for HTMX."""
            if cluster not in self.clusters:
                return '<div class="error">Unknown cluster</div>'

            ports = self.port_mappings[cluster]
            return self._build_status_html(cluster, ports)

        @app.route("/api/logs-html")
        def logs_html():
            """Get recent logs as HTML for HTMX."""
            logs = self._get_logs(50)
            if not logs:
                return '<div class="log-empty">No logs yet</div>'

            html = ""
            for entry in reversed(logs):  # Most recent first
                timestamp = entry.timestamp.strftime("%H:%M:%S")
                level_class = f"log-{entry.level}"
                level_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(entry.level, "")

                html += f'<div class="log-entry {level_class}">'
                html += f'<span class="log-time">{timestamp}</span>'
                html += f'<span class="log-cluster">[{entry.cluster}]</span>'
                html += f'<span class="log-icon">{level_icon}</span>'
                html += f'<span class="log-message">{entry.message}</span>'
                if entry.details:
                    # Escape HTML in details to prevent XSS
                    escaped_details = (
                        entry.details.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace('"', "&quot;")
                    )
                    html += f'<details class="log-details"><summary>Details</summary>'
                    html += f'<pre>{escaped_details}</pre></details>'
                html += "</div>"
            return html

        @app.route("/")
        def index():
            html = """
<!DOCTYPE html>
<html>
<head>
    <title>Ray Clusters Dashboard</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            padding: 20px;
            color: #333;
        }
        h1 {
            margin-bottom: 30px;
            color: #2c3e50;
            font-size: 32px;
        }
        .clusters {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 16px;
        }
        .cluster-card {
            background: white;
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .cluster-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }
        .cluster-name {
            font-size: 18px;
            font-weight: 600;
            color: #2c3e50;
        }
        .cluster-name a {
            color: #3498db;
            text-decoration: none;
        }
        .cluster-name a:hover {
            text-decoration: underline;
        }
        .cluster-meta {
            font-size: 11px;
            color: #95a5a6;
            margin-left: 8px;
            font-weight: 400;
        }
        .direct-link {
            font-size: 12px;
            color: #3498db;
            text-decoration: none;
            padding: 4px 8px;
            border: 1px solid #3498db;
            border-radius: 4px;
        }
        .direct-link:hover {
            background: #3498db;
            color: white;
        }
        .status-loading {
            color: #95a5a6;
            font-style: italic;
        }
        .status-content {
            font-size: 14px;
        }
        .stat {
            margin-bottom: 8px;
            font-size: 13px;
        }
        .resources {
            margin: 10px 0;
        }
        .resource-bar {
            margin-bottom: 8px;
        }
        .resource-label {
            font-size: 12px;
            margin-bottom: 3px;
            color: #555;
        }
        .bar {
            width: 100%;
            height: 6px;
            background: #ecf0f1;
            border-radius: 3px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2980b9);
            transition: width 0.3s ease;
        }
        .error {
            color: #e74c3c;
            padding: 10px;
            background: #fee;
            border-radius: 4px;
            font-size: 13px;
        }
        /* Log panel styles */
        .log-panel {
            margin-top: 30px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .log-header {
            background: #2c3e50;
            color: white;
            padding: 12px 16px;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .log-header-title {
            font-size: 14px;
        }
        .log-refresh-info {
            font-size: 11px;
            opacity: 0.7;
        }
        .log-content {
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 12px;
            background: #1a1a2e;
            color: #eee;
        }
        .log-empty {
            padding: 20px;
            text-align: center;
            color: #666;
        }
        .log-entry {
            padding: 8px 12px;
            border-bottom: 1px solid #2a2a3e;
            display: flex;
            flex-wrap: wrap;
            align-items: flex-start;
            gap: 8px;
        }
        .log-entry:hover {
            background: #2a2a3e;
        }
        .log-time {
            color: #7f8c8d;
            flex-shrink: 0;
        }
        .log-cluster {
            color: #3498db;
            font-weight: 500;
            flex-shrink: 0;
        }
        .log-icon {
            flex-shrink: 0;
        }
        .log-message {
            color: #ecf0f1;
            flex: 1;
            word-break: break-word;
        }
        .log-error .log-message {
            color: #e74c3c;
        }
        .log-warning .log-message {
            color: #f39c12;
        }
        .log-details {
            width: 100%;
            margin-top: 4px;
        }
        .log-details summary {
            cursor: pointer;
            color: #95a5a6;
            font-size: 11px;
        }
        .log-details summary:hover {
            color: #bdc3c7;
        }
        .log-details pre {
            margin-top: 8px;
            padding: 10px;
            background: #0d0d1a;
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
            color: #bdc3c7;
            font-size: 11px;
            max-height: 200px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <h1>Ray Clusters Dashboard</h1>
    <div class="clusters">
"""
            for name, info in self.clusters.items():
                ports = self.port_mappings[name]
                html += f"""
        <div class="cluster-card">
            <div class="cluster-header">
                <div>
                    <div class="cluster-name">
                        <a href="/{name}/">{name}</a>
                        <span class="cluster-meta">{info.head_ip}</span>
                    </div>
                </div>
                <a href="http://localhost:{ports.dashboard_port}" class="direct-link" target="_blank">Direct</a>
            </div>
            <div hx-get="/api/cluster/{name}/status-html"
                 hx-trigger="load"
                 hx-swap="innerHTML"
                 class="status-loading">
                Loading status...
            </div>
        </div>
"""
            html += """
    </div>
    <div class="log-panel">
        <div class="log-header">
            <span class="log-header-title">Proxy Logs</span>
            <span class="log-refresh-info">Auto-refreshes every 5s</span>
        </div>
        <div class="log-content"
             hx-get="/api/logs-html"
             hx-trigger="load, every 5s"
             hx-swap="innerHTML">
            <div class="log-empty">Loading logs...</div>
        </div>
    </div>
</body>
</html>
"""
            return html

        @app.route("/<cluster>/", defaults={"path": ""})
        @app.route("/<cluster>/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
        def proxy(cluster, path):
            if cluster not in self.clusters:
                return f"Unknown cluster: {cluster}", 404

            ports = self.port_mappings[cluster]
            target_url = f"http://localhost:{ports.dashboard_port}/{path}"
            if request.query_string:
                target_url += f"?{request.query_string.decode()}"

            try:
                resp = requests.request(
                    method=request.method,
                    url=target_url,
                    headers={k: v for k, v in request.headers if k.lower() != "host"},
                    data=request.get_data(),
                    cookies=request.cookies,
                    allow_redirects=False,
                    timeout=30,
                )
                excluded_headers = ["content-encoding", "content-length", "transfer-encoding", "connection"]
                headers = [(k, v) for k, v in resp.raw.headers.items() if k.lower() not in excluded_headers]
                return Response(resp.content, resp.status_code, headers)
            except requests.RequestException as e:
                return f"Error connecting to cluster {cluster}: {e}", 502

        return app

    def start(self) -> None:
        """Start proxy server in background thread."""
        app = self._create_app()
        self.server = make_server("localhost", self.proxy_port, app, threaded=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info("Started Ray dashboard proxy on http://localhost:%d", self.proxy_port)

    def stop(self) -> None:
        """Stop proxy server."""
        if self.server:
            self.server.shutdown()
        if self.thread:
            self.thread.join(timeout=5)
