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

"""RPC-based distributed execution context for Fray.

This module provides a Connect RPC-based implementation of the JobContext protocol,
enabling distributed task execution across multiple workers coordinated by a central
controller.

Architecture
------------
The system consists of three main components:

1. **FrayController**: Central coordinator that maintains the task queue and worker registry.
   Implemented as a Connect RPC service (ASGI application) served via uvicorn.

2. **FrayWorker**: Worker process that registers with the controller, fetches tasks,
   executes them using cloudpickle deserialization, and reports results.

3. **FrayContext**: Client-side JobContext implementation that submits tasks to the
   controller and polls for results.

Quick Start
-----------
Starting a controller::

    from fray.job.rpc.controller import FrayControllerServer

    server = FrayControllerServer(port=50051)
    server.start()
    print(f"Controller running at http://localhost:{server._port}")

Starting a worker::

    from fray.job.rpc.worker import FrayWorker

    worker = FrayWorker(controller_address="http://localhost:50051")
    worker.run()

Submitting tasks::

    from fray.job.context import create_job_ctx

    ctx = create_job_ctx("fray", controller_address="http://localhost:50051")

    def add(a, b):
        return a + b

    future = ctx.run(add, 5, 3)
    result = ctx.get(future)  # Returns 8

See Also
--------
lib/fray/examples/rpc/ : Complete example scripts
"""

from fray.job.rpc.context import FrayContext
from fray.job.rpc.controller import FrayControllerServer, FrayControllerServicer

__all__ = ["FrayContext", "FrayControllerServer", "FrayControllerServicer"]
