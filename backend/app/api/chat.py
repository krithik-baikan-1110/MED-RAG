from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.core.langchain_orchestrator import orchestrator
from backend.app.services import chat_repository

router = APIRouter()


class ChatMessageRequest(BaseModel):
    session_id: str
    question: Optional[str] = None
    image_path: Optional[str] = None
    domain: Optional[str] = None
    user_id: Optional[str] = None


class CreateSessionRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    title: Optional[str] = None


class DeleteSessionRequest(BaseModel):
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


class PrimeRequest(BaseModel):
    session_id: str
    image_path: Optional[str] = None
    question: Optional[str] = None
    domain: Optional[str] = None


@router.post("/message")
def create_message(req: ChatMessageRequest):
    try:
        result = orchestrator.generate_response(
            session_id=req.session_id,
            question=req.question or "",
            image_path=req.image_path,
            domain_hint=req.domain,
            user_id=req.user_id,
        )
        return {"success": True, **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - unexpected runtime issues
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/sessions")
def create_session(req: CreateSessionRequest):
    try:
        record = chat_repository.create_session(
            user_id=req.user_id,
            session_id=req.session_id,
            title=req.title,
        )
        return {"success": True, "session": record}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/sessions/{user_id}")
def list_sessions(user_id: str):
    try:
        sessions = chat_repository.list_sessions(user_id)
        return {"success": True, "sessions": sessions}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    try:
        chat_repository.delete_session(session_id)
        orchestrator.reset_session(session_id)
        return {"success": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/history/{session_id}")
def get_history(session_id: str):
    history = orchestrator.get_history(session_id)
    return {"success": True, "session_id": session_id, "history": history}


@router.post("/reset")
def reset_session(req: ResetRequest):
    orchestrator.reset_session(req.session_id)
    return {"success": True}


@router.post("/prime")
def prime_session(req: PrimeRequest):
    try:
        result = orchestrator.generate_response(
            session_id=req.session_id,
            question=req.question or "",
            image_path=req.image_path,
            domain_hint=req.domain,
            silent=True,
        )
        return {"success": True, **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - unexpected runtime issues
        raise HTTPException(status_code=500, detail=str(exc)) from exc
