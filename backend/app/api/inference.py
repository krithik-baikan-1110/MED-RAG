# backend/app/api/inference.py

from fastapi import APIRouter, Form
from pydantic import BaseModel
from typing import Optional
from backend.app.core.rag_pipeline import run_rag_pipeline

router = APIRouter()

class InferenceRequest(BaseModel):
    question: str
    image_path: Optional[str] = None
    domain: Optional[str] = None

@router.post("/infer")
def infer(req: InferenceRequest):
    out = run_rag_pipeline(req.question, image_path=req.image_path, domain_hint=req.domain)
    return out
