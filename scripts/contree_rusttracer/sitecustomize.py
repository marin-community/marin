# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import sys
import threading

from _contree_rusttracer import Tracer


def _install():
    active_pid = os.environ.get("TRACER_ACTIVE_PID")
    if active_pid is not None and active_pid != str(os.getpid()):
        return
    os.environ["TRACER_ACTIVE_PID"] = str(os.getpid())
    tracer = Tracer(
        os.environ.get("TRACER_OUTPUT", "/tmp/trace.jsonl"),
        os.environ.get("TRACER_REPO_ROOT", os.getcwd()),
    )

    def _trace(frame, event, arg):
        tracer.trace(frame, event, arg)
        return _trace

    sys.settrace(_trace)
    threading.settrace(_trace)

    def _on_exit():
        sys.settrace(None)
        threading.settrace(None)
        tracer.finish()

    atexit.register(_on_exit)


_install()
