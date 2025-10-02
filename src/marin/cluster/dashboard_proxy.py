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
import threading
from typing import TYPE_CHECKING

import requests
from flask import Flask, Response, request
from werkzeug.serving import make_server

if TYPE_CHECKING:
    from .ray import ClusterInfo

logger = logging.getLogger(__name__)


class DashboardProxy:
    """Thread-based Flask proxy for multiple Ray dashboards."""

    def __init__(
        self, clusters: dict[str, "ClusterInfo"], port_mappings: dict[str, tuple[int, int, int]], proxy_port: int = 9999
    ):
        self.clusters = clusters
        self.port_mappings = port_mappings
        self.proxy_port = proxy_port
        self.server = None
        self.thread = None

    def create_app(self) -> Flask:
        """Create Flask application."""
        app = Flask(__name__)

        @app.route("/")
        def index():
            """Landing page listing all clusters."""
            html = "<h1>Ray Clusters</h1><ul>"
            for name, info in self.clusters.items():
                dashboard_port = self.port_mappings[name][0]
                html += f"""
                <li>
                    <a href="/{name}/">{name}</a> -
                    {info.head_ip} ({info.zone})
                    [<a href="http://localhost:{dashboard_port}">direct</a>]
                </li>
                """
            html += "</ul>"
            return html

        @app.route("/<cluster>/", defaults={"path": ""})
        @app.route("/<cluster>/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
        def proxy(cluster, path):
            """Proxy requests to appropriate cluster dashboard."""
            if cluster not in self.clusters:
                return f"Unknown cluster: {cluster}", 404

            dashboard_port = self.port_mappings[cluster][0]
            target_url = f"http://localhost:{dashboard_port}/{path}"

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
        logger.info(f"Dashboard proxy started at http://localhost:{self.proxy_port}")

        # iterate over clusters and log their dashboard URLs
        for name in self.clusters:
            dashboard_port = self.port_mappings[name][0]
            logger.info(
                f"Cluster '{name}' dashboard available at http://localhost:{dashboard_port} or http://localhost:{self.proxy_port}/{name}/"
            )

    def stop(self):
        """Stop proxy server."""
        if self.server:
            self.server.shutdown()
        if self.thread:
            self.thread.join(timeout=5)
