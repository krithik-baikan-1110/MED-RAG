# backend/app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import base64
from pathlib import Path

from dotenv import load_dotenv

from backend.app.api import api_router

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="MED-RAG Backend")

FAVICON_BYTES = base64.b64decode(
    "AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAGAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
)

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.environ.setdefault("UPLOAD_DIR", UPLOAD_DIR)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",       # local dev
        "https://medrag.online",         # production
        "https://www.medrag.online",     # production www
        "http://medrag.online",          # fallback
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "MED-RAG backend running"}

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename or "upload")[-1] or ".png"
        unique_name = f"{uuid.uuid4().hex}{suffix}"
        dest = os.path.join(UPLOAD_DIR, unique_name)
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Use relative path — nginx or frontend will resolve the full URL
        file_url = f"/uploads/{unique_name}"
        return JSONResponse(
            {
                "status": "success",
                "file_url": file_url,
                "file_path": dest,
            }
        )
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(content=FAVICON_BYTES, media_type="image/x-icon")
