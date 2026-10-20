"""
Supabase client configuration for SkyRL database registration.

Copied from LLaMA-Factory's unified_db module.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded supabase module
_supabase_module = None
_Client = None


def _get_supabase():
    """Lazy-load supabase module to avoid import errors when not installed."""
    global _supabase_module, _Client
    if _supabase_module is None:
        try:
            import supabase
            from supabase import Client

            _supabase_module = supabase
            _Client = Client
        except ImportError:
            raise ImportError("supabase-py is required for database registration. Install with: pip install supabase")
    return _supabase_module, _Client


class SupabaseConfig:
    """Configuration class for Supabase connection."""

    @staticmethod
    def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
        """Get setting from environment variables."""
        env_val = os.environ.get(key.upper())
        if env_val is not None:
            return env_val
        return default

    @property
    def supabase_url(self) -> str:
        """Supabase project URL."""
        return self.get_setting("SUPABASE_URL", "")

    @property
    def supabase_anon_key(self) -> str:
        """Supabase anonymous/public key."""
        return self.get_setting("SUPABASE_ANON_KEY", "")

    @property
    def supabase_service_role_key(self) -> str:
        """Supabase service role key for admin operations."""
        return self.get_setting("SUPABASE_SERVICE_ROLE_KEY", "")

    @property
    def is_configured(self) -> bool:
        """Check if Supabase is properly configured."""
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def has_admin_access(self) -> bool:
        """Check if admin access is available."""
        return bool(self.supabase_service_role_key)


# Create config instance
supabase_config = SupabaseConfig()


def create_supabase_client(use_admin: bool = False):
    """
    Create and configure Supabase client.

    Args:
        use_admin: If True, use service role key for admin operations

    Returns:
        Configured Supabase client

    Raises:
        ValueError: If Supabase is not properly configured
    """
    supabase, Client = _get_supabase()

    if not supabase_config.is_configured:
        raise ValueError(
            "Supabase not configured. Please set SUPABASE_URL and SUPABASE_ANON_KEY in your environment variables."
        )

    # Choose the appropriate key
    if use_admin and supabase_config.has_admin_access:
        key = supabase_config.supabase_service_role_key
    else:
        key = supabase_config.supabase_anon_key

    # Try to create client with optional timeout settings
    try:
        from supabase.lib.client_options import ClientOptions

        options = ClientOptions(
            postgrest_client_timeout=30,
            storage_client_timeout=30,
        )
    except (ImportError, TypeError):
        options = None

    if options is not None:
        try:
            return supabase.create_client(supabase_config.supabase_url, key, options)
        except (AttributeError, TypeError) as exc:
            logger.debug(f"Supabase client options failed: {exc}; retrying without options.")

    return supabase.create_client(supabase_config.supabase_url, key)


def get_default_client():
    """Get default Supabase client for regular operations."""
    return create_supabase_client(use_admin=False)


def get_admin_client():
    """Get admin Supabase client for operations that bypass RLS."""
    return create_supabase_client(use_admin=True)
