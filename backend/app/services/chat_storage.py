from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Optional
from uuid import uuid4
import os

from .supabase_client import get_supabase_client

CHAT_BUCKET_ENV = "SUPABASE_CHAT_BUCKET"


def _bucket_name() -> str:
    bucket = os.getenv(CHAT_BUCKET_ENV)
    if not bucket:
        raise RuntimeError("SUPABASE_CHAT_BUCKET environment variable is required")
    return bucket


def store_attachment(file_obj: BinaryIO, filename: str, content_type: Optional[str] = None) -> str:
    """Upload attachment to Supabase storage and return public URL."""
    client = get_supabase_client()
    bucket = _bucket_name()
    suffix = Path(filename).suffix or ".bin"
    key = f"{datetime.utcnow().strftime('%Y/%m/%d')}/{uuid4().hex}{suffix}"

    file_obj.seek(0)
    client.storage.from_(bucket).upload(key, file_obj.read(), {
        "content-type": content_type or "application/octet-stream"
    })
    # Assuming bucket is public; for private buckets generate signed URL instead
    return client.storage.from_(bucket).get_public_url(key)
