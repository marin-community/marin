# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Browser dashboard and OpenAI-compatible reverse proxy for a quick-serve vLLM job.

The dashboard is a single self-contained Vue page served at ``/``. It and every
``/v1/*`` request resolve through the Iris controller proxy's
``/proxy/<encoded-name>/`` prefix, so all browser fetches use relative URLs
(``new URL(path, location.href)``) — the proxy does not rewrite HTML bodies, so an
absolute path like ``/v1/chat/completions`` would escape the prefix.

``/v1/*`` requests are reverse-proxied to the local vLLM server with the response
streamed back verbatim, so server-sent-event token streaming works end to end.
"""

from __future__ import annotations

import dataclasses
import logging
import socket
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

import httpx
import uvicorn
from rigging.timing import Duration, ExponentialBackoff
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

# Request headers that must not be forwarded to the upstream vLLM server; httpx
# recomputes Host/Content-Length, and the rest are hop-by-hop per RFC 7230.
_REQUEST_DROP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
# Response headers dropped so the framing matches the re-chunked StreamingResponse.
# Content-Encoding is preserved because aiter_raw() yields the undecoded body.
_RESPONSE_DROP_HEADERS = frozenset({"content-length", "connection", "keep-alive", "transfer-encoding"})


@dataclass(frozen=True)
class ServingInfo:
    """Static serving metadata surfaced at ``/info`` and rendered by the dashboard."""

    model: str
    tensor_parallel_size: int
    max_model_len: int | None
    dtype: str
    has_chat_template: bool
    tpu_type: str
    endpoint: str


def build_dashboard_app(
    *,
    upstream_base_url: str,
    model_id: str,
    info: ServingInfo,
    request_timeout_seconds: float = 600.0,
) -> Starlette:
    """Build the Starlette app fronting a local vLLM server.

    Args:
        upstream_base_url: Root URL of the local vLLM server (without ``/v1``).
        model_id: The model id vLLM reports; surfaced to the dashboard.
        info: Static serving metadata returned from ``/info``.
        request_timeout_seconds: Per-request timeout for upstream proxying.
    """
    state: dict[str, httpx.AsyncClient] = {}

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        state["client"] = httpx.AsyncClient(
            base_url=upstream_base_url,
            timeout=httpx.Timeout(request_timeout_seconds, connect=10.0),
        )
        try:
            yield
        finally:
            await state.pop("client").aclose()

    async def index(_request: Request) -> Response:
        return HTMLResponse(DASHBOARD_HTML)

    async def serving_info(_request: Request) -> Response:
        return JSONResponse(dataclasses.asdict(info))

    async def health(_request: Request) -> Response:
        client = state["client"]
        try:
            response = await client.get("/health")
            ready = response.status_code == 200
        except httpx.HTTPError:
            ready = False
        return JSONResponse(
            {"status": "ok" if ready else "loading", "model": model_id},
            status_code=200 if ready else 503,
        )

    async def proxy(request: Request) -> Response:
        client = state["client"]
        body = await request.body()
        fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in _REQUEST_DROP_HEADERS}
        upstream_request = client.build_request(
            request.method,
            request.url.path,
            params=dict(request.query_params),
            content=body,
            headers=fwd_headers,
        )
        try:
            upstream_response = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            return JSONResponse({"error": f"upstream vLLM request failed: {exc}"}, status_code=502)

        resp_headers = {k: v for k, v in upstream_response.headers.items() if k.lower() not in _RESPONSE_DROP_HEADERS}

        async def body_iter() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream_response.aiter_raw():
                    yield chunk
            finally:
                await upstream_response.aclose()

        return StreamingResponse(
            body_iter(),
            status_code=upstream_response.status_code,
            headers=resp_headers,
            media_type=upstream_response.headers.get("content-type"),
        )

    return Starlette(
        routes=[
            Route("/", index),
            Route("/dashboard", index),
            Route("/info", serving_info),
            Route("/health", health),
            Route("/v1/{path:path}", proxy, methods=["GET", "POST", "OPTIONS"]),
        ],
        lifespan=lifespan,
    )


def bind_serving_socket(host: str, port: int) -> socket.socket:
    """Bind a listening socket up front so the port is claimed before serving.

    Iris allocates the task's named port from a range (30000-40000) that overlaps
    the OS ephemeral range, so any ephemeral socket the task later opens — notably
    vLLM's many internal sockets — can squat the port we need. Binding here, before
    vLLM starts, removes the port from the ephemeral pool and reserves it for us.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    return sock


@contextmanager
def serve_app_background(app: Starlette, sock: socket.socket, *, start_timeout_seconds: float = 30.0) -> Iterator[None]:
    """Run ``app`` under uvicorn on an already-bound ``sock`` in a daemon thread.

    The caller owns the listening socket (see :func:`bind_serving_socket`) so it can
    be claimed before any competing socket in the process can take the port.
    """
    host, port = sock.getsockname()[:2]
    config = uvicorn.Config(app, log_level="info", log_config=None, workers=1)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, name="quick-serve-dashboard", daemon=True)
    logger.info("Starting quick-serve dashboard on %s:%d", host, port)
    thread.start()
    started = ExponentialBackoff(initial=0.02, maximum=1, jitter=0).wait_until(
        lambda: server.started or not thread.is_alive(),
        timeout=Duration.from_seconds(start_timeout_seconds),
    )
    if not started or not server.started:
        server.should_exit = True
        thread.join()
        raise RuntimeError("quick-serve dashboard failed to start")
    try:
        yield
    finally:
        logger.info("Stopping quick-serve dashboard on %s:%d", host, port)
        server.should_exit = True
        thread.join()


# Single-file Vue 3 dashboard. Vue is loaded from a CDN by the browser directly
# (not through the Iris proxy); all same-origin fetches stay relative so they
# resolve under the controller proxy prefix.
DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>marin · quick serve</title>
<script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --fg:#e6edf3;
          --muted:#8b949e; --accent:#2f81f7; --good:#3fb950; --err:#f85149; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:920px; margin:0 auto; padding:26px 16px 64px; }
  header { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
  h1 { font-size:20px; margin:0; font-weight:650; letter-spacing:-.01em; }
  .sub { color:var(--muted); font-size:13px; word-break:break-all; }
  .status { margin-left:auto; font-size:12px; color:var(--muted); white-space:nowrap; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%;
         background:var(--muted); margin-right:6px; vertical-align:middle; }
  .dot.ok { background:var(--good); } .dot.bad { background:var(--err); }
  .meta { display:flex; gap:16px; flex-wrap:wrap; margin-top:10px; color:var(--muted); font-size:12px; }
  .meta b { color:var(--fg); font-weight:600; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px;
          padding:16px; margin-top:18px; }
  label { display:block; font-size:11px; color:var(--muted); margin-bottom:6px;
          text-transform:uppercase; letter-spacing:.05em; }
  textarea,input,select { font-family:inherit; color:var(--fg); background:#0b0f14;
          border:1px solid var(--border); border-radius:8px; }
  textarea { width:100%; min-height:90px; resize:vertical; padding:10px 12px;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; }
  .tabs { display:flex; gap:8px; }
  .tab { padding:7px 16px; border:1px solid var(--border); border-radius:8px;
         background:#0b0f14; color:var(--muted); cursor:pointer; font-size:13px; }
  .tab.active { color:#fff; border-color:var(--accent); background:rgba(47,129,247,.12); }
  .row { display:flex; gap:14px; flex-wrap:wrap; align-items:end; margin-top:14px; }
  .field { display:flex; flex-direction:column; }
  select,input[type=number],input[type=text] { padding:8px 10px; font-size:13px; }
  input[type=number] { width:104px; }
  button.go { margin-left:auto; background:var(--accent); border:none; border-radius:8px;
          padding:10px 22px; font-size:14px; font-weight:600; color:#fff; cursor:pointer; }
  button.go:disabled { opacity:.5; cursor:default; }
  .hint { color:var(--muted); font-size:12px; margin-top:12px; }
  .out { margin-top:16px; white-space:pre-wrap; word-break:break-word;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px;
          background:#0b0f14; border:1px solid var(--border); border-radius:8px;
          padding:13px; min-height:64px; overflow-x:auto; }
  .out .err { color:var(--err); }
  .msg { border:1px solid var(--border); border-radius:8px; padding:10px 12px; margin-bottom:10px; }
  .msg .role { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin-bottom:4px; }
  .msg.user .role { color:var(--accent); }
  .msg.assistant .role { color:var(--good); }
  .msg .body { white-space:pre-wrap; word-break:break-word;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; }
</style>
</head>
<body>
<div id="app" class="wrap">
  <header>
    <h1>marin · quick serve</h1>
    <span class="status"><span class="dot" :class="dotClass"></span>{{ statusText }}</span>
  </header>
  <div class="sub">{{ info.model || model || 'model' }}</div>
  <div class="meta">
    <span>tp <b>{{ info.tensor_parallel_size ?? '?' }}</b></span>
    <span>max_model_len <b>{{ info.max_model_len ?? 'auto' }}</b></span>
    <span>dtype <b>{{ info.dtype || '?' }}</b></span>
    <span>chat template <b>{{ info.has_chat_template ? 'yes' : 'no' }}</b></span>
  </div>

  <div class="card">
    <div class="tabs">
      <div class="tab" :class="{active: mode==='chat'}" @click="mode='chat'">Chat</div>
      <div class="tab" :class="{active: mode==='completion'}" @click="mode='completion'">Completion</div>
    </div>

    <div v-if="mode==='chat'">
      <div class="row" style="margin-top:16px">
        <div class="field" style="flex:1">
          <label>System prompt (optional)</label>
          <input type="text" v-model="system" placeholder="(none)"/>
        </div>
      </div>
      <div v-for="(m,i) in messages" :key="i" class="msg" :class="m.role">
        <div class="role">{{ m.role }}</div>
        <div class="body">{{ m.content }}</div>
      </div>
      <label style="margin-top:14px">Message</label>
      <textarea v-model="userInput" @keydown="onKey" placeholder="Ask something…"></textarea>
    </div>

    <div v-else>
      <label style="margin-top:16px">Prompt</label>
      <textarea v-model="prompt" @keydown="onKey">The capital of France is</textarea>
    </div>

    <div class="row">
      <div class="field"><label>Temperature</label>
        <input type="number" step="0.1" min="0" v-model.number="temperature"/></div>
      <div class="field"><label>Max tokens</label>
        <input type="number" min="1" v-model.number="maxTokens"/></div>
      <div class="field"><label>Top-p</label>
        <input type="number" step="0.05" min="0" max="1" v-model.number="topP"/></div>
      <button class="go" :disabled="busy" @click="run">{{ busy ? 'Generating…' : 'Generate' }}</button>
    </div>
    <div class="hint">Cmd/Ctrl+Enter to run. Streams tokens as they arrive. In chat mode the
      conversation is kept; clear it to start over.</div>
    <div v-if="mode==='chat'" class="hint"><a href="#" @click.prevent="messages=[]">clear conversation</a></div>

    <div class="out" v-if="error"><span class="err">{{ error }}</span></div>
    <div class="out" v-else-if="mode==='completion'">{{ output }}</div>
  </div>
</div>
<script>
const { createApp } = Vue;
const api = path => new URL(path, location.href).toString();

createApp({
  data() {
    return {
      info: {}, model: '', mode: 'chat', status: 'connecting',
      system: '', userInput: '', messages: [], prompt: 'The capital of France is',
      output: '', error: '', busy: false,
      temperature: 0.7, maxTokens: 512, topP: 1.0,
    };
  },
  computed: {
    statusText() {
      const labels = {connecting: 'connecting…', ok: 'ready', loading: 'loading…', bad: 'unreachable'};
      return labels[this.status] || this.status;
    },
    dotClass() { return this.status === 'ok' ? 'ok' : (this.status === 'bad' ? 'bad' : ''); },
  },
  methods: {
    onKey(e) { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') this.run(); },
    async loadInfo() {
      try {
        const r = await fetch(api('info'));
        if (r.ok) {
          this.info = await r.json();
          this.model = this.info.model || '';
          this.mode = this.info.has_chat_template ? 'chat' : 'completion';
        }
      } catch (e) { /* health() reports reachability */ }
    },
    async health() {
      try {
        const r = await fetch(api('health'));
        const j = await r.json().catch(() => ({}));
        this.status = r.ok ? 'ok' : 'loading';
        if (j.model) this.model = j.model;
      } catch (e) { this.status = 'bad'; }
    },
    async run() {
      if (this.busy) return;
      this.error = ''; this.busy = true;
      const model = this.model || this.info.model;
      try {
        if (this.mode === 'chat') await this.runChat(model);
        else await this.runCompletion(model);
      } catch (e) { this.error = String(e); }
      finally { this.busy = false; }
    },
    async runChat(model) {
      const text = this.userInput.trim();
      if (!text) return;
      const history = [];
      if (this.system.trim()) history.push({ role: 'system', content: this.system.trim() });
      this.messages.push({ role: 'user', content: text });
      this.userInput = '';
      for (const m of this.messages) history.push({ role: m.role, content: m.content });
      this.messages.push({ role: 'assistant', content: '' });
      // Mutate through the reactive proxy (not the raw pushed object) so streaming deltas re-render.
      const reply = this.messages[this.messages.length - 1];
      const body = { model, messages: history, stream: true,
                     temperature: this.temperature, max_tokens: this.maxTokens, top_p: this.topP };
      await this.stream('v1/chat/completions', body, delta => {
        const d = delta.choices?.[0]?.delta?.content;
        if (d) reply.content += d;
      });
    },
    async runCompletion(model) {
      this.output = '';
      const body = { model, prompt: this.prompt, stream: true,
                     temperature: this.temperature, max_tokens: this.maxTokens, top_p: this.topP };
      await this.stream('v1/completions', body, delta => {
        const d = delta.choices?.[0]?.text;
        if (d) this.output += d;
      });
    },
    async stream(path, body, onDelta) {
      const r = await fetch(api(path), { method: 'POST',
        headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
      if (!r.ok) { throw new Error(r.status + ' — ' + (await r.text())); }
      const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\\n'); buf = lines.pop();
        for (const line of lines) {
          const t = line.trim();
          if (!t.startsWith('data:')) continue;
          const payload = t.slice(5).trim();
          if (payload === '[DONE]') continue;
          try { onDelta(JSON.parse(payload)); } catch (e) { /* skip keepalives */ }
        }
      }
    },
  },
  mounted() { this.loadInfo(); this.health(); setInterval(this.health, 15000); },
}).mount('#app');
</script>
</body>
</html>"""
