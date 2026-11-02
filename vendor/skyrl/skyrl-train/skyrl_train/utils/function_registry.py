"""Ray-synchronized registry machinery for user-defined training algorithms."""

from __future__ import annotations

from typing import Callable, List

import cloudpickle
import ray
from loguru import logger


@ray.remote
class RegistryActor:
    """Shared Ray actor for managing function registries across processes."""

    def __init__(self):
        self.registry = {}

    def register(self, name: str, func_serialized: bytes):
        """Register a serialized function."""
        self.registry[name] = func_serialized

    def get(self, name: str):
        """Get a serialized function by name."""
        return self.registry.get(name)

    def list_available(self):
        """List all available function names."""
        return list(self.registry.keys())

    def unregister(self, name: str):
        """Unregister a function by name."""
        return self.registry.pop(name, None)


class BaseFunctionRegistry:
    """Base class for function registries with Ray actor synchronization."""

    # Subclasses should override these class attributes
    _actor_name = None
    _function_type = "Function"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._functions = {}
        cls._ray_actor = None
        cls._synced_to_actor = False

    @classmethod
    def _get_or_create_actor(cls):
        """Get or create the Ray actor for managing the registry using get_if_exists."""
        if not ray.is_initialized():
            raise Exception("Ray is not initialized, cannot create registry actor")

        if cls._ray_actor is None:
            # Use get_if_exists to create actor only if it doesn't exist
            cls._ray_actor = RegistryActor.options(name=cls._actor_name, get_if_exists=True).remote()
        return cls._ray_actor

    @classmethod
    def _sync_local_to_actor(cls):
        """Sync all local functions to Ray actor."""
        if cls._synced_to_actor:
            return
        if not ray.is_initialized():
            raise Exception("Ray is not initialized, cannot sync with actor")

        try:
            actor = cls._get_or_create_actor()
            if actor is not None:
                for name, func in cls._functions.items():
                    func_serialized = cloudpickle.dumps(func)
                    ray.get(actor.register.remote(name, func_serialized))
                cls._synced_to_actor = True
        except Exception as e:
            logger.error(f"Error syncing {cls._function_type} to actor: {e}")
            raise e

    @classmethod
    def sync_with_actor(cls):
        """Sync local registry with Ray actor if Ray is available."""
        # Only try if Ray is initialized
        if not ray.is_initialized():
            raise Exception("Ray is not initialized, cannot sync with actor")

        # Repeated Ray init/shutdown cycles can leave a stale actor handle on the class.
        try:
            _ = ray.get_actor(cls._actor_name)  # this raises exception if the actor is stale
        except ValueError:
            cls._ray_actor = None
            cls._synced_to_actor = False

        # First, sync our local functions to the actor
        cls._sync_local_to_actor()

        actor = cls._get_or_create_actor()
        if actor is None:
            return

        available = ray.get(actor.list_available.remote())

        # Sync any new functions from actor to local registry
        for name in available:
            if name not in cls._functions:
                func_serialized = ray.get(actor.get.remote(name))
                if func_serialized is not None:
                    try:
                        func = cloudpickle.loads(func_serialized)
                        cls._functions[name] = func
                    except Exception as e:
                        # If deserialization fails, skip this function
                        logger.error(f"Error deserializing {name} from actor: {e}")
                        raise e

    @classmethod
    def register(cls, name: str, func: Callable):
        """Register a function.

        If ray is initialized, this function will get or create a named ray actor (RegistryActor)
        for the registry, and sync the registry to the actor.

        If ray is not initalized, the function will be stored in the local registry only.

        To make sure all locally registered functions are available to all ray processes,
        call sync_with_actor() after ray.init().

        Args:
            name: Name of the function to register.
            func: Function to register.

        Raises:
            ValueError: If the function is already registered.
        """
        if name in cls._functions:
            raise ValueError(f"{cls._function_type} '{name}' already registered")

        # Always store in local registry first
        cls._functions[name] = func

        # Try to sync with Ray actor if Ray is initialized
        if ray.is_initialized():
            actor = cls._get_or_create_actor()
            if actor is not None:
                func_serialized = cloudpickle.dumps(func)
                ray.get(actor.register.remote(name, func_serialized))

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get a function by name.

        If ray is initialized, this function will first sync the local registry with the RegistryActor.
        Then it will return the function if it is found in the registry.

        Args:
            name: Name of the function to get. Can be a string or a StrEnum.

        Returns:
            The function if it is found in the registry.
        """
        # Try to sync with actor first if Ray is available
        if ray.is_initialized():
            cls.sync_with_actor()

        if name not in cls._functions:
            available = list(cls._functions.keys())
            raise ValueError(f"Unknown {cls._function_type.lower()} '{name}'. Available: {available}")
        return cls._functions[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered functions."""
        # Try to sync with actor first if Ray is available
        if ray.is_initialized():
            cls.sync_with_actor()
        return list(cls._functions.keys())

    @classmethod
    def unregister(cls, name: str):
        """Unregister a function. Useful for testing."""
        # Try to sync with actor first to get any functions that might be in the actor but not local
        if ray.is_initialized():
            cls.sync_with_actor()

        # Track if we found the function anywhere
        found_locally = name in cls._functions
        found_in_actor = False

        # Remove from local registry if it exists
        if found_locally:
            del cls._functions[name]

        # Try to remove from Ray actor if Ray is available
        if ray.is_initialized():
            actor = cls._get_or_create_actor()
            if actor is not None:
                # Check if it exists in actor first
                available_in_actor = ray.get(actor.list_available.remote())
                if name in available_in_actor:
                    found_in_actor = True
                    ray.get(actor.unregister.remote(name))

        # Only raise error if the function wasn't found anywhere
        if not found_locally and not found_in_actor:
            raise ValueError(f"{cls._function_type} '{name}' not registered")

    @classmethod
    def shutdown_actor(cls):
        """Force-stop the registry actor without clearing local functions."""
        if ray.is_initialized():
            try:
                actor = ray.get_actor(cls._actor_name)  # this raises exception if the actor is stale
                ray.kill(actor)
            except ValueError:
                pass  # Actor may already be gone
        cls._ray_actor = None
        cls._synced_to_actor = False
