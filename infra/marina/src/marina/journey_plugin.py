# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for journeys: one kernel per session, one browser page per spec.

The app under test is the ``apps/<name>/journeys/`` directory the spec lives in. The
kernel serves the whole apps directory (so the shell's app switcher is real) from the
data root ``--journey-data-root`` or the ``.data`` directory beside ``apps``. Screenshots
land in ``--journey-shots``/``<app>/``; ``--journey-video`` also records each spec.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Browser, sync_playwright

from marina.db import database_from_env
from marina.journeys import Journey, Kernel, running_kernel
from marina.server import MarinaConfig

JOURNEYS_DIR = "journeys"
DEFAULT_SHOTS_DIR = Path(__file__).resolve().parents[2] / "journeys-out"
APPS_DIR_OPTION = "--journey-apps-dir"
DATA_ROOT_OPTION = "--journey-data-root"
SHOTS_OPTION = "--journey-shots"
VIDEO_OPTION = "--journey-video"
HEADED_OPTION = "--journey-headed"
# Journeys need a browser and a kernel, so a plain `pytest` skips them; `marina journey` enables them.
ENABLE_OPTION = "--journeys"
VIEWPORT = {"width": 1280, "height": 900}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("marina journeys")
    group.addoption(APPS_DIR_OPTION, default=None, help="Apps directory the kernel serves.")
    group.addoption(DATA_ROOT_OPTION, default=None, help="Data root (directory or gs:// URL).")
    group.addoption(SHOTS_OPTION, default=None, help="Where screenshots and videos go.")
    group.addoption(VIDEO_OPTION, action="store_true", default=False, help="Record a video per spec.")
    group.addoption(HEADED_OPTION, action="store_true", default=False, help="Show the browser.")
    group.addoption(ENABLE_OPTION, action="store_true", default=False, help="Run journeys (needs Chromium).")


def journeys_dir_of(path: Path) -> Path:
    """The ``apps/<name>/journeys/`` directory a spec lives in."""
    for parent in path.parents:
        if parent.name == JOURNEYS_DIR:
            return parent
    raise ValueError(f"{path} is not under an apps/<name>/{JOURNEYS_DIR}/ directory")


def app_of(path: Path) -> str:
    return journeys_dir_of(path).parent.name


def apps_dir_of(path: Path) -> Path:
    return journeys_dir_of(path).parent.parent


@pytest.fixture(scope="session")
def marina_kernel(request: pytest.FixtureRequest) -> Iterator[Kernel]:
    first = Path(str(request.session.items[0].path)) if request.session.items else Path.cwd()
    apps_dir = Path(request.config.getoption(APPS_DIR_OPTION) or apps_dir_of(first))
    data_root = request.config.getoption(DATA_ROOT_OPTION) or str(apps_dir.parent / ".data")
    config = MarinaConfig(
        apps_dir=apps_dir, data_root=data_root, iap_audience=None, database=database_from_env(os.environ)
    )
    with running_kernel(config) as kernel:
        yield kernel


@pytest.fixture(scope="session")
def marina_browser(request: pytest.FixtureRequest) -> Iterator[Browser]:
    headed = request.config.getoption(HEADED_OPTION)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not headed)
        yield browser
        browser.close()


@pytest.fixture
def journey(request: pytest.FixtureRequest) -> Iterator[Journey]:
    """A fresh page on the kernel for the app this spec belongs to; refusals fail the test."""
    if not request.config.getoption(ENABLE_OPTION):
        pytest.skip(f"journeys run through `marina journey` or pytest {ENABLE_OPTION}")
    marina_kernel: Kernel = request.getfixturevalue("marina_kernel")
    marina_browser: Browser = request.getfixturevalue("marina_browser")
    app = app_of(Path(str(request.path)))
    shots_root = Path(request.config.getoption(SHOTS_OPTION) or DEFAULT_SHOTS_DIR)
    shots = shots_root / app
    record_video_dir = str(shots / "video") if request.config.getoption(VIDEO_OPTION) else None
    context = marina_browser.new_context(viewport=VIEWPORT, record_video_dir=record_video_dir)
    page = context.new_page()
    walk = Journey(page=page, origin=marina_kernel.origin, app=app, shots=shots)
    try:
        yield walk
        walk.finish()
    finally:
        context.close()
