from functools import lru_cache
import os
from supabase import Client, create_client


def _get_env_or_raise(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Environment variable {key} is required for Supabase integration")
    return value


@lru_cache
def get_supabase_client() -> Client:
    """Return a cached Supabase client using service-role key for server-side ops."""
    url = _get_env_or_raise("SUPABASE_URL")
    service_key = _get_env_or_raise("SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, service_key)
