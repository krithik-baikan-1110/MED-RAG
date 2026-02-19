from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .supabase_client import get_supabase_client

CHAT_SESSIONS = "chat_sessions"
CHAT_MESSAGES = "chat_messages"


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def create_session(user_id: str, *, session_id: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
    """Create a new chat session and return the persisted record."""
    client = get_supabase_client()
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "updated_at": _utc_now_iso(),
    }
    if session_id:
        payload["id"] = session_id
    if title:
        payload["title"] = title

    response = client.table(CHAT_SESSIONS).upsert(payload, on_conflict="id").execute()
    data = response.data or []
    if not data:
        raise RuntimeError("Failed to create chat session")
    return data[0]


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    client = get_supabase_client()
    response = (
        client.table(CHAT_SESSIONS)
        .select("id, user_id, title, created_at, updated_at")
        .eq("id", session_id)
        .limit(1)
        .execute()
    )
    data = response.data or []
    return data[0] if data else None


def update_session(session_id: str, *, title: Optional[str] = None) -> None:
    update_fields: Dict[str, Any] = {"updated_at": _utc_now_iso()}
    if title is not None:
        update_fields["title"] = title

    client = get_supabase_client()
    client.table(CHAT_SESSIONS).update(update_fields).eq("id", session_id).execute()


def delete_session(session_id: str) -> None:
    client = get_supabase_client()
    client.table(CHAT_SESSIONS).delete().eq("id", session_id).execute()


def list_sessions(user_id: str) -> List[Dict[str, Any]]:
    client = get_supabase_client()
    response = (
        client.table(CHAT_SESSIONS)
        .select("id, title, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return response.data or []


def insert_message(
    session_id: str,
    role: str,
    content: str,
    attachment_url: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    client = get_supabase_client()
    record = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "attachment_url": attachment_url,
        "meta": meta or {},
    }
    client.table(CHAT_MESSAGES).insert(record).execute()
    touch_session(session_id)


def load_messages(session_id: str) -> List[Dict[str, Any]]:
    client = get_supabase_client()
    response = (
        client.table(CHAT_MESSAGES)
        .select("id, role, content, attachment_url, meta, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )
    return response.data or []


def touch_session(session_id: str) -> None:
    client = get_supabase_client()
    client.table(CHAT_SESSIONS).update({"updated_at": _utc_now_iso()}).eq("id", session_id).execute()
