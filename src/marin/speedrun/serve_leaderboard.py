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

"""
For local development: serves leaderboard files from static directory
"""

import argparse
import http.server
import os
import socketserver
from pathlib import Path


def serve_static(port: int = 8000):
    """Serve the static leaderboard files."""
    static_dir = Path(__file__).parent / "static"
    os.chdir(static_dir)

    Handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", port), Handler) as httpd:
        try:
            print(f"Serving leaderboard at http://localhost:{port}")
            print("Press Ctrl+C to stop")
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        except OSError as e:
            if e.errno == 48:
                print(f"\nError: Port {port} is already in use")
            else:
                raise e
        finally:
            httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Serve the static leaderboard")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = parser.parse_args()

    serve_static(args.port)


if __name__ == "__main__":
    main()
