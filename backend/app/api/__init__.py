from fastapi import APIRouter

from backend.app.api import chat, inference

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(inference.router, prefix="/api", tags=["rag"])
