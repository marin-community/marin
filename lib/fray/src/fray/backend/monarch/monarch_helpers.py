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

"""Helper classes for Monarch backend implementation."""

from __future__ import annotations

import uuid
from typing import Any

try:
    from monarch.actor import Actor, Future, ProcMesh, endpoint

    MONARCH_AVAILABLE = True
except ImportError:
    MONARCH_AVAILABLE = False
    # Create stub types for type checking when Monarch is not available
    Actor = object
    Future = object
    ProcMesh = object

    def endpoint(fn):
        """Stub endpoint decorator."""
        return fn


class MonarchObjectRef:
    """Wraps Monarch Future to provide Fray-compatible object reference."""

    def __init__(self, future: Future | None = None, take_first: bool = False, obj_id: str | None = None):
        """
        Initialize MonarchObjectRef.

        Args:
            future: Monarch Future object (for task/actor results)
            take_first: If True, extract first result from mesh results (single-actor semantics)
            obj_id: Object ID for object store references
        """
        self._future = future
        self._take_first = take_first
        self._obj_id = obj_id  # For object store references
        self._is_object_store_ref = obj_id is not None

    def __repr__(self):
        if self._is_object_store_ref:
            return f"MonarchObjectRef(obj_id={self._obj_id})"
        return f"MonarchObjectRef({self._future}, take_first={self._take_first})"


class MonarchActorHandle:
    """Wraps Monarch actor to provide Fray-compatible handle."""

    def __init__(self, actor: Any):
        """
        Initialize MonarchActorHandle.

        Args:
            actor: Monarch actor instance returned from ProcMesh.spawn()
        """
        self._actor = actor

    def __getattr__(self, name: str):
        """Intercept method calls to wrap results in MonarchObjectRef."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def method_wrapper(*args, **kwargs):
            # Get the endpoint from the actor
            endpoint = getattr(self._actor, name)
            # Call the endpoint and get future
            result = endpoint(*args, **kwargs)
            # Wrap in MonarchObjectRef
            return MonarchObjectRef(result, take_first=False)

        return method_wrapper


class ObjectStoreActor(Actor):
    """Actor-based object store for put/get functionality.

    Monarch doesn't have a built-in object store like Ray, so we implement
    one using a dedicated actor that maintains a dictionary of stored objects.
    """

    def __init__(self):
        """Initialize the object store with an empty dictionary."""
        super().__init__()
        self._store: dict[str, Any] = {}

    @endpoint
    def put(self, obj_id: str, obj: Any) -> None:
        """
        Store an object in the object store.

        Args:
            obj_id: Unique identifier for the object
            obj: Object to store
        """
        self._store[obj_id] = obj

    @endpoint
    def get(self, obj_id: str) -> Any:
        """
        Retrieve an object from the object store.

        Args:
            obj_id: Unique identifier for the object

        Returns:
            The stored object

        Raises:
            KeyError: If obj_id not found
        """
        return self._store[obj_id]


def generate_object_id() -> str:
    """Generate a unique object ID for object store."""
    return f"obj-{uuid.uuid4().hex[:16]}"
