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

"""Flask proxy for multiple Ray dashboards."""

import logging
import re
import subprocess
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import requests
from flask import Flask, Response, request
from werkzeug.serving import make_server

if TYPE_CHECKING:
    from .ray import ClusterInfo, RayPortMapping

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """Resource usage information for a single resource type."""

    used: str
    total: str

    def percentage(self) -> float:
        """Calculate usage percentage, handling unit conversions."""
        try:
            # Strip common units for numeric comparison
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


def format_number(value_str: str) -> str:
    """Format numbers compactly (e.g., 61680.0 -> 61.7k, 2049.0 -> 2049).

    Args:
        value_str: String representation of a number, possibly with units

    Returns:
        Compact formatted string
    """
    # Extract unit suffix if present (GiB, MiB, etc.)
    unit = ""
    clean_str = value_str
    for suffix in ["TiB", "GiB", "MiB", "KiB"]:
        if value_str.endswith(suffix):
            unit = suffix
            clean_str = value_str[: -len(suffix)]
            break

    try:
        num = float(clean_str)

        # If no unit, apply k/M suffix for large numbers
        if not unit:
            if num >= 1_000_000:
                formatted = f"{num / 1_000_000:.1f}"
                formatted = formatted.rstrip("0").rstrip(".")
                return f"{formatted}M"
            elif num >= 10_000:
                formatted = f"{num / 1_000:.1f}"
                formatted = formatted.rstrip("0").rstrip(".")
                return f"{formatted}k"

        # Remove unnecessary decimals
        if num == int(num):
            return f"{int(num)}{unit}"
        formatted = f"{num:.1f}"
        formatted = formatted.rstrip("0").rstrip(".")
        return f"{formatted}{unit}"
    except ValueError:
        return value_str


def parse_ray_status(status_output: str) -> ClusterStatus:
    """Parse ray status output into structured data.

    Args:
        status_output: Raw output from `ray status` command

    Returns:
        ClusterStatus with parsed status information
    """
    result = ClusterStatus()

    lines = status_output.strip().split("\n")
    current_section = None

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
            # Parse resource usage lines like "2049.0/61680.0 CPU"
            match = re.match(r"([\d.]+[KMGTB]?i?B?)\/([\d.]+[KMGTB]?i?B?)\s+(.+)", line)
            if match:
                used, total, resource_name = match.groups()
                # Filter out object_store_memory and ray-worker-manual-* entries
                if resource_name == "object_store_memory":
                    continue
                if re.search("ray.*worker", resource_name):
                    continue
                result.resources[resource_name] = ResourceUsage(used=used, total=total)

    return result


class DashboardProxy:
    """Thread-based Flask proxy for multiple Ray dashboards."""

    def __init__(
        self, clusters: dict[str, "ClusterInfo"], port_mappings: dict[str, "RayPortMapping"], proxy_port: int = 9999
    ):
        self.clusters = clusters
        self.port_mappings = port_mappings
        self.proxy_port = proxy_port
        self.server = None
        self.thread = None

    def create_app(self) -> Flask:
        """Create Flask application."""
        app = Flask(__name__)

        @app.route("/api/cluster/<cluster>/status-html")
        def cluster_status_html(cluster):
            """Get cluster status as HTML for htmx."""
            if cluster not in self.clusters:
                return '<div class="error">Unknown cluster</div>'

            ports = self.port_mappings[cluster]
            gcs_address = f"localhost:{ports.gcs_port}"

            try:
                result = subprocess.run(
                    ["ray", "status", f"--address={gcs_address}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                status = parse_ray_status(result.stdout)

                html = '<div class="status-content">'

                # Nodes section
                html += f'<div class="stat"><strong>Nodes:</strong> {status.active_count()} active'
                if status.pending_count() > 0:
                    html += f", {status.pending_count()} pending"
                html += "</div>"

                # Resources section
                if status.resources:
                    html += '<div class="resources">'
                    for resource_name, resource_usage in status.resources.items():
                        percent = resource_usage.percentage()
                        used_fmt = format_number(resource_usage.used)
                        total_fmt = format_number(resource_usage.total)
                        html += '<div class="resource-bar">'
                        html += '<div class="resource-label">'
                        html += f"{resource_name}: {used_fmt}/{total_fmt} ({percent:.1f}%)"
                        html += "</div>"
                        html += '<div class="bar">'
                        html += f'<div class="bar-fill" style="width: {percent}%"></div>'
                        html += "</div>"
                        html += "</div>"
                    html += "</div>"

                html += "</div>"
                return html
            except subprocess.TimeoutExpired:
                return '<div class="error">⏱ Timeout fetching status</div>'
            except subprocess.CalledProcessError as e:
                return f'<div class="error">❌ Error: {e.stderr[:100]}</div>'
            except Exception as e:
                return f'<div class="error">❌ Error: {str(e)[:100]}</div>'

        @app.route("/")
        def index():
            """Landing page listing all clusters."""
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
</body>
</html>
"""
            return html

        @app.route("/<cluster>/", defaults={"path": ""})
        @app.route("/<cluster>/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
        def proxy(cluster, path):
            """Proxy requests to appropriate cluster dashboard."""
            if cluster not in self.clusters:
                return f"Unknown cluster: {cluster}", 404

            ports = self.port_mappings[cluster]
            target_url = f"http://localhost:{ports.dashboard_port}/{path}"

            # Forward query parameters
            if request.query_string:
                target_url += f"?{request.query_string.decode()}"

            # Proxy the request
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

                # Build response
                excluded_headers = ["content-encoding", "content-length", "transfer-encoding", "connection"]
                headers = [(k, v) for k, v in resp.raw.headers.items() if k.lower() not in excluded_headers]

                return Response(resp.content, resp.status_code, headers)
            except requests.RequestException as e:
                return f"Error connecting to cluster {cluster}: {e}", 502

        return app

    def start(self):
        """Start proxy server in background thread."""
        app = self.create_app()
        self.server = make_server("localhost", self.proxy_port, app, threaded=True)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop proxy server."""
        if self.server:
            self.server.shutdown()
        if self.thread:
            self.thread.join(timeout=5)
