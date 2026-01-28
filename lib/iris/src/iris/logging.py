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

import logging
import sys
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class BufferedLogRecord:
    timestamp: float
    level: str
    logger_name: str
    message: str


class LogBuffer(Protocol):
    def append(self, record: BufferedLogRecord) -> None: ...
    def query(self, *, prefix: str | None = None, limit: int = 200) -> list[BufferedLogRecord]: ...


class LogRingBuffer:
    """Thread-safe ring buffer that collects Python log records."""

    def __init__(self, maxlen: int = 5000):
        self._buffer: deque[BufferedLogRecord] = deque(maxlen=maxlen)
        self._lock = Lock()

    def append(self, record: BufferedLogRecord) -> None:
        with self._lock:
            self._buffer.append(record)

    def query(self, *, prefix: str | None = None, limit: int = 200) -> list[BufferedLogRecord]:
        with self._lock:
            items = list(self._buffer)
        if prefix:
            items = [r for r in items if r.logger_name.startswith(prefix)]
        return items[-limit:]


class RingBufferHandler(logging.Handler):
    def __init__(self, buffer: LogRingBuffer):
        super().__init__()
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        self._buffer.append(
            BufferedLogRecord(
                timestamp=record.created,
                level=record.levelname,
                logger_name=record.name,
                message=self.format(record),
            )
        )


LOGS_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Iris Logs</title>
  <style>
    body { font-family: monospace; margin: 20px; background: #1e1e1e; color: #d4d4d4; }
    h1 { color: #569cd6; }
    .controls { margin: 10px 0; display: flex; gap: 10px; align-items: center; }
    input, select { padding: 5px; background: #2d2d2d; color: #d4d4d4; border: 1px solid #555; }
    input { width: 300px; }
    #log-container { max-height: 80vh; overflow-y: auto; }
    .log-line { padding: 2px 0; white-space: pre-wrap; word-break: break-all; }
    .log-line.WARNING { color: #dcdcaa; }
    .log-line.ERROR { color: #f44747; }
    .log-line.DEBUG { color: #808080; }
    a { color: #569cd6; }
  </style>
</head>
<body>
  <a href="/">&larr; Dashboard</a>
  <h1>Process Logs</h1>
  <div class="controls">
    <label>Prefix: <input id="prefix" type="text" placeholder="e.g. iris.cluster.controller"></label>
    <label>Limit: <select id="limit">
      <option value="100">100</option>
      <option value="200" selected>200</option>
      <option value="500">500</option>
      <option value="1000">1000</option>
    </select></label>
  </div>
  <div id="log-container"></div>
  <script>
    function escapeHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
    async function refresh() {
      const prefix = document.getElementById('prefix').value;
      const limit = document.getElementById('limit').value;
      const params = new URLSearchParams({limit});
      if (prefix) params.set('prefix', prefix);
      const logs = await fetch('/api/logs?' + params).then(r => r.json());
      document.getElementById('log-container').innerHTML = logs.map(
        l => `<div class="log-line ${escapeHtml(l.level)}">${escapeHtml(l.message)}</div>`
      ).join('');
    }
    refresh();
    setInterval(refresh, 3000);
    document.getElementById('prefix').addEventListener('input', refresh);
    document.getElementById('limit').addEventListener('change', refresh);
  </script>
</body>
</html>
"""


def logs_page_response(request):
    """Starlette route handler returning the logs HTML page."""
    from starlette.responses import HTMLResponse
    return HTMLResponse(LOGS_HTML)


def logs_api_response(request, buffer: LogBuffer | None):
    """Starlette route handler returning log records as JSON."""
    from starlette.responses import JSONResponse
    if not buffer:
        return JSONResponse([])
    prefix = request.query_params.get("prefix") or None
    try:
        limit = int(request.query_params.get("limit", "200"))
    except (ValueError, TypeError):
        limit = 200
    records = buffer.query(prefix=prefix, limit=limit)
    return JSONResponse([
        {"timestamp": r.timestamp, "level": r.level, "logger_name": r.logger_name, "message": r.message}
        for r in records
    ])


_global_buffer = LogRingBuffer()


def get_global_buffer() -> LogRingBuffer:
    return _global_buffer


_configured = False


def configure_logging(level: int = logging.INFO) -> LogRingBuffer:
    """Configure iris logging: stderr handler + ring buffer. Idempotent."""
    global _configured
    if _configured:
        root = logging.getLogger()
        root.setLevel(level)
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, RingBufferHandler):
                h.setLevel(level)
        return _global_buffer

    _configured = True
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    ring_handler = RingBufferHandler(_global_buffer)
    ring_handler.setLevel(logging.DEBUG)
    ring_handler.setFormatter(formatter)
    root.addHandler(ring_handler)

    return _global_buffer
