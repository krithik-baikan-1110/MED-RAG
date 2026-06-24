# backend/app/core/rag_utils.py
"""
BiomedCLIP embedding utilities via HuggingFace Inference API.
No local model loading — all inference is done remotely,
keeping RAM usage minimal.
"""
import os
import io
import base64
import logging
import time
import numpy as np
import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("medrag.rag_utils")

HF_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
HF_MODEL_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Embedding dimension for BiomedCLIP
EMBED_DIM = 512

# Retry config for cold-start handling
MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds — HF free tier cold starts can take 10-30s


def _hf_headers() -> dict:
    """Build authorization headers for HuggingFace Inference API."""
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    return headers


def _post_with_retry(payload: dict = None, data: bytes = None, content_type: str = None) -> dict:
    """
    POST to HuggingFace Inference API with retry logic for cold starts.
    The free tier may return 503 while the model is loading.
    """
    headers = _hf_headers()
    if content_type:
        headers["Content-Type"] = content_type

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if data is not None:
                resp = requests.post(HF_API_URL, headers=headers, data=data, timeout=120)
            else:
                resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)

            if resp.status_code == 503:
                # Model is loading (cold start on free tier)
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                wait_time = body.get("estimated_time", RETRY_DELAY)
                log.info(
                    "HF model loading (attempt %d/%d). Waiting %.0fs...",
                    attempt, MAX_RETRIES, wait_time,
                )
                time.sleep(wait_time)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES:
                log.error("HuggingFace API request failed after %d attempts: %s", MAX_RETRIES, e)
                raise
            log.warning("HF API attempt %d failed: %s. Retrying in %ds...", attempt, e, RETRY_DELAY)
            time.sleep(RETRY_DELAY)

    raise RuntimeError(f"HuggingFace API failed after {MAX_RETRIES} retries")


def embed_image(image: "Image.Image | str") -> np.ndarray:
    """
    Embed an image using BiomedCLIP via HuggingFace Inference API.
    Accepts a PIL Image or a file path string.
    Returns L2-normalized numpy vector of shape (512,).
    """
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    # Convert to bytes (JPEG for smaller payload)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    result = _post_with_retry(data=image_bytes, content_type="image/png")

    # HF feature-extraction returns list of floats or nested list
    vec = np.array(result, dtype=np.float32).flatten()

    # BiomedCLIP should return 512-dim; take first EMBED_DIM if longer
    if vec.shape[0] > EMBED_DIM:
        vec = vec[:EMBED_DIM]

    # L2 normalize
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec


def embed_text(text: str) -> np.ndarray:
    """
    Embed text using BiomedCLIP via HuggingFace Inference API.
    Returns L2-normalized numpy vector of shape (512,).
    """
    if not text:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    payload = {"inputs": text}
    result = _post_with_retry(payload=payload)

    # HF feature-extraction returns list of floats or nested list
    vec = np.array(result, dtype=np.float32).flatten()

    if vec.shape[0] > EMBED_DIM:
        vec = vec[:EMBED_DIM]

    # L2 normalize
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec
