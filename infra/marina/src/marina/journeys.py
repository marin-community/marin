# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Browser tests against an in-process kernel.

``Kernel`` runs the apps on a loopback port; ``Journey`` wraps a Playwright page with
text-based helpers (``sees``, ``click``, ``fill``), screenshots, and a record of page
errors and failed API calls that ``finish`` asserts on. Specs live in
``apps/<name>/journeys/test_*.py`` and take the ``journey`` fixture from
:mod:`marina.journey_plugin`.
"""

import contextlib
import logging
import re
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import uvicorn
from playwright.sync_api import Page, expect

from marina.server import MarinaConfig, create_app

logger = logging.getLogger(__name__)

LOOPBACK = "127.0.0.1"
KERNEL_START_TIMEOUT = 20.0
SETTLE_MS = 250
# Widths a layout has opinions about: a phone, a narrow laptop, a wide desk.
DEFAULT_WIDTHS = (390, 900, 1400)

# Every control a person can press right now, as ``tag:label``. Hidden elements are left
# out. A tick box or radio is named by its label, since its value is ``on`` whatever it does.
_OFFERS_SCRIPT = """
() => {
  const beside = (element) => {
    const named = element.id ? document.querySelector(`label[for="${CSS.escape(element.id)}"]`) : null
    return (named ?? element.closest('label'))?.innerText ?? ''
  }
  const seen = []
  for (const element of document.querySelectorAll('button, a[href], [role=button], input, select, textarea')) {
    if (!(element.offsetWidth || element.offsetHeight)) continue
    const ticked = element.tagName === 'INPUT' && ['checkbox', 'radio'].includes(element.type)
    const label = (
      element.innerText || (ticked ? beside(element) : element.value) || element.placeholder ||
      element.getAttribute('aria-label') || element.getAttribute('title') || ''
    ).trim().replace(/\\s+/g, ' ')
    seen.push(`${element.tagName.toLowerCase()}:${label}`)
  }
  return [...new Set(seen)]
}
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind((LOOPBACK, 0))
        return s.getsockname()[1]


@dataclass
class Kernel:
    """A kernel serving ``config`` on a loopback port from a background thread."""

    config: MarinaConfig
    port: int = field(default_factory=free_port)
    _server: uvicorn.Server | None = None
    _thread: threading.Thread | None = None

    @property
    def origin(self) -> str:
        return f"http://{LOOPBACK}:{self.port}"

    def start(self) -> None:
        app = create_app(self.config)
        self._server = uvicorn.Server(uvicorn.Config(app, host=LOOPBACK, port=self.port, log_level="warning"))
        self._thread = threading.Thread(target=self._server.run, name="marina-kernel", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + KERNEL_START_TIMEOUT
        while time.monotonic() < deadline:
            with contextlib.suppress(httpx.HTTPError):
                if httpx.get(f"{self.origin}/healthz", timeout=1).status_code == 200:
                    return
            time.sleep(0.05)
        raise RuntimeError(f"kernel did not answer /healthz on {self.origin} within {KERNEL_START_TIMEOUT}s")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)


@contextlib.contextmanager
def running_kernel(config: MarinaConfig) -> Iterator[Kernel]:
    kernel = Kernel(config)
    kernel.start()
    try:
        yield kernel
    finally:
        kernel.stop()


@dataclass
class Journey:
    """One person, one app, one browser page.

    Paths are relative to the app: ``visit("/")`` opens ``/<app>/``. Screenshots go to
    ``shots``; ``refusals`` fills with page errors and failed API calls as they happen.
    """

    page: Page
    origin: str
    app: str
    shots: Path
    refusals: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.shots.mkdir(parents=True, exist_ok=True)
        self.page.on("pageerror", lambda error: self.refusals.append(f"uncaught {str(error)[:300]}"))
        self.page.on("console", self._on_console)
        self.page.on("response", self._on_response)

    def _on_console(self, message) -> None:
        if message.type == "error":
            self.refusals.append(f"console {message.text[:300]}")

    def _on_response(self, response) -> None:
        if not response.url.startswith(self.origin) or response.status < 400:
            return
        path = response.url[len(self.origin) :]
        if path.startswith("/api/") or path.startswith(f"/{self.app}/data/") or path.startswith(f"/{self.app}/api/"):
            self.refusals.append(f"{response.status} {path}")

    def url(self, path: str) -> str:
        return f"{self.origin}/{self.app}{path if path.startswith('/') else '/' + path}"

    def visit(self, path: str = "/") -> "Journey":
        """Open an app-relative path and wait for the document and its fonts."""
        self.page.goto(self.url(path), wait_until="domcontentloaded")
        self.page.evaluate("() => document.fonts.ready")
        return self

    def sees(self, text: str | re.Pattern[str], timeout: float = 10.0) -> "Journey":
        """Assert ``text`` is visible somewhere on the page."""
        expect(self.page.get_by_text(text).first).to_be_visible(timeout=timeout * 1000)
        return self

    def absent(self, text: str | re.Pattern[str], timeout: float = 5.0) -> "Journey":
        expect(self.page.get_by_text(text)).to_have_count(0, timeout=timeout * 1000)
        return self

    def click(self, label: str | re.Pattern[str]) -> "Journey":
        """Press the button or link a person reads as ``label``."""
        control = self.page.get_by_role("button", name=label, exact=True).or_(
            self.page.get_by_role("link", name=label, exact=True)
        )
        if control.count() == 0:
            control = self.page.get_by_text(label, exact=True)
        control.first.click()
        return self

    def fill(self, label: str | re.Pattern[str], value: str) -> "Journey":
        """Type into the field a person reads as ``label`` (its label or placeholder)."""
        field_ = self.page.get_by_label(label).or_(self.page.get_by_placeholder(label))
        field_.first.fill(value)
        return self

    def select(self, label: str | re.Pattern[str], option: str) -> "Journey":
        self.page.get_by_label(label).first.select_option(label=option)
        return self

    def offers(self) -> list[str]:
        """Every control a person can press right now, as ``tag:label``."""
        return self.page.evaluate(_OFFERS_SCRIPT)

    def reads(self) -> str:
        """What the page says, with blank lines taken out."""
        text = self.page.evaluate("() => document.body.innerText")
        return re.sub(r"\n{2,}", "\n", text).strip()

    def api(self, path: str) -> object:
        """Fetch a same-origin JSON endpoint as the signed-in person."""
        response = self.page.request.get(f"{self.origin}{path}")
        if not response.ok:
            raise AssertionError(f"{response.status} {path}: {response.text()[:300]}")
        return response.json()

    def shoot(self, name: str, full: bool = False) -> "Journey":
        """Keep an image of the screen as ``<shots>/<name>.png``."""
        self.page.wait_for_timeout(SETTLE_MS)
        self.page.screenshot(path=str(self.shots / f"{name}.png"), full_page=full)
        return self

    def widths(self, name: str, widths: tuple[int, ...] = DEFAULT_WIDTHS) -> "Journey":
        """Photograph the same screen at each width and restore the viewport."""
        before = self.page.viewport_size
        for width in widths:
            self.page.set_viewport_size({"width": width, "height": before["height"] if before else 900})
            self.shoot(f"{name}-{width}")
        if before:
            self.page.set_viewport_size(before)
        return self

    def finish(self) -> None:
        """Fail on anything the page refused or threw while the journey ran."""
        if self.refusals:
            raise AssertionError("the page refused or threw:\n  " + "\n  ".join(self.refusals))
