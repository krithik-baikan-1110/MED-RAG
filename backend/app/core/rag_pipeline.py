# """
# rag_pipeline.py — MED-RAG (Adaptive-K + Qwen3VL via OpenRouter)

# Features:
# - Loads BiomedCLIP via open_clip for image/text embeddings (same model used in ingestion)
# - Retrieves candidates from Weaviate
# - Performs local hybrid reranking combining Weaviate certainty + text/image cosine similarity
# - Adaptive-K selection (automatic number of documents to pass to LLM)
# - Uses Qwen3-VL via OpenRouter (OpenAI-compatible client) for generation
# - Defensive checks and logging
# """

# import os
# import sys
# import math
# import logging
# from typing import List, Dict, Any, Optional
# from pathlib import Path

# from dotenv import load_dotenv
# load_dotenv()

# import torch
# import numpy as np
# from PIL import Image

# import open_clip
# import weaviate
# from openai import OpenAI
# from sklearn.metrics.pairwise import cosine_similarity

# # ---- Logging ----
# logger = logging.getLogger("medrag.rag_pipeline")
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# # ---- Config (from env, or defaults) ----
# PROJECT_ROOT = Path(__file__).resolve().parents[3]
# WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # BiomedCLIP model identifier used both for ingestion & query embeddings
# BIOMEDCLIP_MODEL = os.getenv("BIOMEDCLIP_MODEL", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
# VECTOR_CLASS = os.getenv("WEAVIATE_CLASS", "MedicalReport")

# # Adaptive-retrieval / rerank params
# MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "40"))
# MIN_K = int(os.getenv("MIN_K", "2"))
# SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.75"))
# WEIGHT_WEAV = float(os.getenv("WEIGHT_WEAV", "0.5"))
# WEIGHT_TEXT = float(os.getenv("WEIGHT_TEXT", "0.3"))
# WEIGHT_IMG = float(os.getenv("WEIGHT_IMG", "0.2"))

# # Qwen3VL / OpenRouter
# OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_BASE = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen/qwen3-vl-235b-a22b-thinking")

# # ---- Validate LLM API key presence (we will allow retrieval-only flow without it) ----
# if not OPENROUTER_API_KEY:
#     logger.warning("OPENAI_API_KEY / OPENROUTER_API_KEY not set. LLM generation will fail if invoked. "
#                    "You can still run retrieval and reranking locally.")

# # ---- Weaviate client ----
# try:
#     db = weaviate.Client(WEAVIATE_URL)
#     if db.is_ready():
#         logger.info(f"✅ Connected to Weaviate at {WEAVIATE_URL}")
#     else:
#         logger.warning(f"⚠️ Weaviate at {WEAVIATE_URL} not ready (is_ready() false).")
# except Exception as e:
#     logger.error("Failed to connect to Weaviate: %s", e)
#     raise

# # ---- Load BiomedCLIP via OpenCLIP ----
# logger.info("Loading BiomedCLIP model via OpenCLIP on %s ...", DEVICE)
# try:
#     model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
#         BIOMEDCLIP_MODEL,
#         device=DEVICE
#     )
#     model.eval()
#     preprocess = preprocess_val
#     tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL)
#     logger.info("✅ BiomedCLIP loaded via OpenCLIP on %s", DEVICE)
# except Exception as e:
#     logger.exception("Failed to load BiomedCLIP via OpenCLIP: %s", e)
#     raise

# # ---- Embedding helpers (ensure L2-normalized vectors) ----
# def normalize_np(arr: np.ndarray) -> np.ndarray:
#     if arr is None:
#         return None
#     norm = np.linalg.norm(arr)
#     if norm == 0 or math.isnan(norm):
#         return arr
#     return arr / norm

# def embed_image(image_path: str) -> np.ndarray:
#     img = Image.open(image_path).convert("RGB")
#     x = preprocess(img).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         vec = model.encode_image(x)
#     vec = vec.squeeze().cpu().numpy()
#     vec = normalize_np(vec)
#     return vec

# def embed_text(text: str) -> np.ndarray:
#     # tokenizer returns tokens on device
#     tokens = tokenizer(text).to(DEVICE)
#     with torch.no_grad():
#         vec = model.encode_text(tokens)
#     vec = vec.squeeze().cpu().numpy()
#     vec = normalize_np(vec)
#     return vec

# # ---- Hybrid reranker + adaptive-K ----
# def cos_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
#     if a is None or b is None:
#         return 0.0
#     # both should be normalized already; fallback to sklearn if not
#     try:
#         # dot product is equivalent to cosine if normalized
#         if np.allclose(np.linalg.norm(a), 1.0, atol=1e-3) and np.allclose(np.linalg.norm(b), 1.0, atol=1e-3):
#             return float(np.dot(a, b))
#         return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])
#     except Exception:
#         return 0.0

# def hybrid_rerank_and_adaptive_k(query_vec: np.ndarray, candidates: List[Dict[str, Any]]):
#     """
#     candidates: list of weaviate hit dicts with fields:
#       - _additional: {certainty}
#       - text_embedding (list) optional
#       - image_embedding (list) optional
#     returns: reranked list and chosen k
#     """
#     scored = []
#     certs = []
#     for c in candidates:
#         cert = c.get("_additional", {}).get("certainty", 0.0)
#         certs.append(cert)
#     if not candidates:
#         return [], 0

#     for c in candidates:
#         weav_cert = c.get("_additional", {}).get("certainty", 0.0)
#         text_emb = np.array(c.get("text_embedding")) if c.get("text_embedding") else None
#         img_emb = np.array(c.get("image_embedding")) if c.get("image_embedding") else None

#         score_text = cos_sim(query_vec, text_emb)
#         score_img = cos_sim(query_vec, img_emb)

#         combined = WEIGHT_WEAV * weav_cert + WEIGHT_TEXT * score_text + WEIGHT_IMG * score_img

#         scored.append({
#             "combined_score": combined,
#             "weav_certainty": weav_cert,
#             "score_text": score_text,
#             "score_image": score_img,
#             "obj": c
#         })

#     scored.sort(key=lambda x: x["combined_score"], reverse=True)

#     # adaptive K: look for largest drop in weaviate certainties in the scored order
#     ordered_certs = [s["weav_certainty"] for s in scored]
#     k = MIN_K
#     if len(ordered_certs) <= MIN_K:
#         k = len(ordered_certs)
#     else:
#         diffs = np.diff(ordered_certs)
#         drop_idx = int(np.argmax(np.abs(diffs))) + 1  # +1 because diff length is n-1
#         k = max(MIN_K, min(drop_idx, len(ordered_certs)))
#         if ordered_certs and ordered_certs[0] > SIM_THRESHOLD:
#             k = max(k, 3)
#         if ordered_certs and ordered_certs[-1] > SIM_THRESHOLD:
#             k = len(ordered_certs)

#     return scored, k

# # ---- Qwen3VL generation (OpenRouter via OpenAI-compatible client) ----
# def get_openrouter_client() -> Optional[OpenAI]:
#     if not OPENROUTER_API_KEY:
#         return None
#     try:
#         client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE)
#         return client
#     except Exception as e:
#         logger.exception("Failed to create OpenRouter/OpenAI client: %s", e)
#         return None

# _openrouter_client = None

# def qwen3vl_generate(question: str, retrieved_objs: List[Dict[str, Any]], image_path: Optional[str] = None) -> str:
#     """
#     Build a structured prompt with retrieved context and call Qwen3-VL via OpenRouter.
#     Returns the generated text (string). Requires OPENROUTER_API_KEY set.
#     """
#     global _openrouter_client
#     if _openrouter_client is None:
#         _openrouter_client = get_openrouter_client()

#     if _openrouter_client is None:
#         raise RuntimeError("OpenRouter API key not configured. Set OPENAI_API_KEY / OPENROUTER_API_KEY in env.")

#     # Prepare retrieved context as numbered cases
#     context_text = "\n\n".join(
#         [f"Case {i+1}: {r.get('report_text','')}" for i, r in enumerate(retrieved_objs)]
#     )

#     system_prompt = (
#         "You are Qwen-3VL, a multimodal medical AI assistant specialized in radiology/pathology/ophthalmology.\n"
#         "You are provided with a user question and retrieved clinical cases.\n"
#         "Be factual, concise, use radiology/pathology/ophthalmology terminology appropriate to the domain.\n"
#         "State uncertainty clearly and avoid hallucination.\n"
#         "Output MUST follow this structure:\n"
#         "• Findings: <bullet or short paragraph>\n"
#         "• Impression: <concise impression / differential if applicable>\n"
#     )

#     user_text = f"Retrieved context:\n{context_text}\n\nUser question: {question}\n\nAnswer using the retrieved cases and, if provided, the image."

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": [{"type": "text", "text": user_text}]}
#     ]

#     # attach image as base64 if provided (OpenRouter supports data URI images for Qwen3-VL)
#     if image_path and os.path.exists(image_path):
#         try:
#             import base64
#             with open(image_path, "rb") as f:
#                 b64 = base64.b64encode(f.read()).decode("utf-8")
#             messages[1]["content"].append({"type": "image_url", "image_url": f"data:image/png;base64,{b64}"})
#         except Exception:
#             # fallback: try to upload to local FastAPI endpoint (if available) — not required here
#             logger.warning("Failed to attach image as base64; continuing without image.")

#     # call model
#     try:
#         resp = _openrouter_client.chat.completions.create(
#             model=QWEN_MODEL,
#             messages=messages,
#             temperature=0.2,
#             max_tokens=600,
#             top_p=0.9,
#             stream=False,
#         )
#         # API returns choices with message content
#         content = resp.choices[0].message.content.strip()
#         return content
#     except Exception as e:
#         logger.exception("OpenRouter generation failed: %s", e)
#         raise

# # ---- Main RAG pipeline function ----
# def run_rag_pipeline(question: str, image_path: Optional[str] = None) -> str:
#     """
#     End-to-end RAG:
#     1. Embed query (image or text)
#     2. Retrieve MAX_CANDIDATES from Weaviate (near vector)
#     3. Hybrid rerank + adaptive-K
#     4. Generate via Qwen3VL (OpenRouter)
#     Returns generated answer string.
#     """
#     logger.info("⚙️ Running MED-RAG pipeline for question: %s", question)

#     # 1) Embed query
#     if image_path:
#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"Query image not found: {image_path}")
#         logger.info("Embedding query image...")
#         qvec = embed_image(image_path)
#     else:
#         logger.info("Embedding query text...")
#         qvec = embed_text(question)

#     # 2) Retrieve from Weaviate
#     try:
#         logger.info("Querying Weaviate for top %d candidates...", MAX_CANDIDATES)
#         resp = (
#             db.query
#             .get(VECTOR_CLASS, ["image_path", "report_text", "text_embedding", "image_embedding", "_additional{certainty}"])
#             .with_near_vector({"vector": qvec.tolist()})
#             .with_limit(MAX_CANDIDATES)
#             .do()
#         )
#         candidates = resp.get("data", {}).get("Get", {}).get(VECTOR_CLASS, [])
#         if not candidates:
#             logger.warning("⚠️ No results retrieved from Weaviate for this query.")
#             # still proceed with empty context if LLM key present (LLM will rely on image only)
#             candidates = []
#     except Exception as e:
#         logger.exception("Weaviate query failed: %s", e)
#         raise

#     # 3) Hybrid rerank + adaptive K
#     scored, k = hybrid_rerank_and_adaptive_k(qvec, candidates)
#     logger.info("🔍 Hybrid rerank produced %d scored candidates; adaptive-K = %d", len(scored), k)

#     # choose top-k objects
#     top_k_objs = [s["obj"] for s in scored[:k]]

#     # 4) Generate via Qwen3VL
#     try:
#         answer = qwen3vl_generate(question, top_k_objs, image_path=image_path)
#         logger.info("✅ Generation completed.")
#         return answer
#     except Exception as e:
#         # If LLM fails and we still have candidate context, return a fallback summary built from retrieved docs
#         logger.warning("LLM generation failed: %s", e)
#         if top_k_objs:
#             fallback = "Retrieved context (no LLM result):\n" + "\n\n".join(
#                 [f"- {o.get('image_path','')} : {o.get('report_text','')[:300]}" for o in top_k_objs]
#             )
#             return fallback
#         raise

# # ---- If run as main: small debug run (do not call heavy LLM unless key present) ----
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image", help="Path to query image (optional)")
#     parser.add_argument("--question", default="Does this image show pneumonia?", help="Question text")
#     args = parser.parse_args()

#     print("Project root:", PROJECT_ROOT)
#     try:
#         out = run_rag_pipeline(args.question, args.image)
#         print("\n==== Generated Answer ====\n")
#         print(out)
#     except Exception as e:
#         logger.exception("RAG run failed: %s", e)
#         raise
"""
backend/app/core/rag_pipeline.py
MED-RAG RAG pipeline: embeddings (BiomedCLIP via open_clip), Weaviate retrieval,
hybrid rerank (image/text embeddings + Weaviate certainty), adaptive-K and generation
via Qwen3-VL (OpenRouter).
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("medrag.rag_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# External libs
import weaviate
from openai import OpenAI  # openrouter-compatible client (openai package)

# Shared embedding utilities (HuggingFace Inference API)
from backend.app.core.rag_utils import embed_text, embed_image, EMBED_DIM


OPHTHALMOLOGY_CODE_MAP: Dict[str, str] = {
    "N": "Normal (no eye disease)",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration (AMD)",
    "D": "Diabetic Retinopathy (DR)",
    "M": "Macular Edema (ME) or Diabetic Macular Edema (DME)",
    "O": "Other eye diseases or abnormalities",
}


def _parse_ophthalmology_labels(raw: Optional[str]) -> List[Dict[str, str]]:
    if not raw:
        return []
    parts = re.split(r"[\s,;/]+", raw.strip())
    seen = set()
    parsed: List[Dict[str, str]] = []
    for part in parts:
        code = part.strip().upper()
        if not code or code in seen:
            continue
        description = OPHTHALMOLOGY_CODE_MAP.get(code)
        if description:
            parsed.append({"code": code, "description": description})
            seen.add(code)
    return parsed

# optional domain classifier local module (exists in your repo)
try:
    from backend.app.core import domain_classifier
except Exception:
    domain_classifier = None

# CONFIG
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
VECTOR_CLASS = os.getenv("WEAVIATE_CLASS", "MedicalReport")
# DEVICE and BIOMEDCLIP_ID are no longer needed — embeddings are computed
# remotely via HuggingFace Inference API (see rag_utils.py)
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "40"))
MIN_K = int(os.getenv("MIN_K", "3"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.75"))
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen/qwen3-vl-235b-a22b-thinking")

def get_openrouter_api_key() -> Optional[str]:
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

def get_openrouter_base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

def openrouter_default_headers() -> Dict[str, str]:
    app_title = os.getenv("OPENROUTER_APP_TITLE", "MED-RAG (Local Dev)")
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
    headers: Dict[str, str] = {
        "X-Title": app_title,
    }
    if referer:
        headers["HTTP-Referer"] = referer
    return headers

# CONNECT WEAVIATE
db = weaviate.Client(url=WEAVIATE_URL)
if db.is_ready():
    log.info("✅ Connected to Weaviate at %s", WEAVIATE_URL)
else:
    log.warning("⚠️ Weaviate not ready at %s", WEAVIATE_URL)

try:
    agg = (
        db.query
        .aggregate(VECTOR_CLASS)
        .with_meta_count()
        .do()
    )
    aggregate_data = agg.get("data", {}).get("Aggregate", {}).get(VECTOR_CLASS, [])
    meta_count = None
    if isinstance(aggregate_data, list) and aggregate_data:
        meta_count = aggregate_data[0].get("meta", {}).get("count")
    if meta_count:
        log.info("📦 Weaviate class '%s' contains %d objects.", VECTOR_CLASS, meta_count)
    else:
        log.warning(
            "⚠️ Weaviate class '%s' currently has no objects. Ingest data to enable retrieval.",
            VECTOR_CLASS,
        )
except Exception as exc:
    log.warning("Unable to inspect Weaviate class '%s' population: %s", VECTOR_CLASS, exc)

# BiomedCLIP embeddings are now computed via HuggingFace Inference API
# (imported from rag_utils.py — no local model loading needed)
log.info("✅ Using BiomedCLIP via HuggingFace Inference API (no local model)")

# DOMAIN DETECTION
def detect_domain(question: Optional[str], image_path: Optional[str]) -> Dict[str, Any]:
    if domain_classifier:
        try:
            if image_path:
                dom, score = domain_classifier.predict_domain_from_image(image_path, embed_image)
                return {"domain": dom, "score": score, "method": "image-prototype"}
            else:
                dom_embs = domain_classifier.get_domain_embeddings(embed_text)
                if question:
                    q_emb = embed_text(question)
                    scores = {d: float(np.dot(q_emb, v)) for d, v in dom_embs.items()}
                    best = max(scores, key=scores.get)
                    return {"domain": best, "score": scores[best], "method": "text-prototype"}
        except Exception as e:
            log.warning("domain detect fail: %s", e)
    # fallback heuristics:
    q = (question or "").lower()
    if any(k in q for k in ["retina", "fundus", "ophthalm", "macula"]):
        return {"domain": "ophthalmology", "score": 0.7, "method": "keyword"}
    if any(k in q for k in ["biopsy", "h&e", "histology", "slide", "pathology"]):
        return {"domain": "pathology", "score": 0.7, "method": "keyword"}
    return {"domain": "radiology", "score": 0.5, "method": "fallback"}

# HYBRID RERANK
def hybrid_rerank(candidates: List[Dict], query_vec: np.ndarray, modality: str = "image") -> List[Dict]:
    out = []
    for r in candidates:
        props = r.get("properties", r)
        emb_score = 0.0
        try:
            if modality == "image" and props.get("image_embedding"):
                stored = np.array(props["image_embedding"], dtype=float)
                stored = stored / (np.linalg.norm(stored) + 1e-12)
                emb_score = float(np.dot(query_vec, stored))
            elif modality == "text" and props.get("text_embedding"):
                stored = np.array(props["text_embedding"], dtype=float)
                stored = stored / (np.linalg.norm(stored) + 1e-12)
                emb_score = float(np.dot(query_vec, stored))
        except Exception:
            emb_score = 0.0
        cert = float(r.get("_additional", {}).get("certainty", 0.0))
        hybrid = 0.75 * emb_score + 0.25 * cert
        r["_hybrid_score"] = hybrid
        out.append(r)
    out = sorted(out, key=lambda x: x.get("_hybrid_score", 0.0), reverse=True)
    return out

# ADAPTIVE-K retrieval
def retrieve_adaptive_k(
    query_vector: np.ndarray,
    modality: str = "image",
    domain_filter: Optional[str] = None,
) -> List[Dict]:
    if query_vector is None:
        return []
    query_builder = (
        db.query
        .get(
            VECTOR_CLASS,
            [
                "image_path",
                "left_image_path",
                "right_image_path",
                "report_text",
                "projection",
                "image_embedding",
                "text_embedding",
                "domain",
                "label",
            ],
        )
        .with_near_vector({"vector": query_vector.tolist()})
        .with_limit(MAX_CANDIDATES)
        .with_additional(["certainty", "distance", "id"])
    )

    if domain_filter:
        where_filter = {
            "path": ["domain"],
            "operator": "Equal",
            "valueText": domain_filter,
        }
        query_builder = query_builder.with_where(where_filter)

    resp = query_builder.do()
    if resp.get("errors"):
        log.error("Weaviate query returned errors for class '%s': %s", VECTOR_CLASS, resp["errors"])
        return []
    results = resp.get("data", {}).get("Get", {}).get(VECTOR_CLASS, [])
    if not results:
        log.warning(
            "⚠️ No results retrieved from Weaviate. Confirm ingest pipeline populated class '%s' and query vector is valid.",
            VECTOR_CLASS,
        )
        return []
    for item in results:
        props = item.get("properties", item)
        domain_value = str(props.get("domain", ""))[0:].lower()
        if domain_value == "ophthalmology":
            labels_info = _parse_ophthalmology_labels(props.get("label"))
            if labels_info:
                props["ophthalmology_labels"] = labels_info
    reranked = hybrid_rerank(results, query_vector, modality=modality)
    certs = np.array([float(r.get("_additional", {}).get("certainty", 0.0)) for r in reranked])
    if len(certs) <= 1:
        adaptive_k = max(MIN_K, min(len(reranked), 3))
    else:
        drop_idx = int(np.argmax(np.abs(np.diff(certs))))
        adaptive_k = max(MIN_K, min(drop_idx + 1, len(reranked)))
    if certs.size and certs[0] > SIM_THRESHOLD:
        adaptive_k = max(adaptive_k, 3)
    log.info("🔍 Hybrid rerank produced %d scored candidates; adaptive-K = %d", len(reranked), adaptive_k)
    return reranked[:adaptive_k]

# QWEN3-VL generation via OpenRouter (OpenAI client)
def _format_retrieved_context(retrieved_context: List[Dict]) -> str:
    lines = []
    for idx, ctx in enumerate(retrieved_context, 1):
        meta_parts = []
        image_ref = ctx.get("image_path")
        if not image_ref:
            if ctx.get("left_image_path"):
                image_ref = f"left:{ctx['left_image_path']}"
            if ctx.get("right_image_path"):
                suffix = f"right:{ctx['right_image_path']}"
                image_ref = f"{image_ref}, {suffix}" if image_ref else suffix
        if image_ref:
            meta_parts.append(f"image={image_ref}")
        if ctx.get("projection"):
            meta_parts.append(f"projection={ctx['projection']}")
        if ctx.get("certainty") is not None:
            meta_parts.append(f"certainty={ctx['certainty']:.2f}")
        if ctx.get("hybrid_score") is not None:
            meta_parts.append(f"hybrid={ctx['hybrid_score']:.2f}")
        header = f"Case {idx}"
        if meta_parts:
            header += " (" + ", ".join(meta_parts) + ")"
        report = (ctx.get("report_text") or "").strip() or "[no report text]"
        details = [report]
        labels = ctx.get("ophthalmology_labels") or []
        if labels:
            label_text = "; ".join(f"{item['code']}: {item['description']}" for item in labels)
            details.append(f"Ophthalmology labels: {label_text}")
        lines.append(f"{header}\n" + "\n".join(details))
    return "\n\n".join(lines) if lines else "[no retrieved cases]"


def qwen3vl_generate(
    question: str,
    retrieved_context: List[Dict],
    image_path: Optional[str] = None,
    temperature: float = 0.2,
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    api_key = get_openrouter_api_key()
    if not api_key:
        return {"success": False, "text": "", "raw": "OPENROUTER_API_KEY missing"}
    client = OpenAI(
        api_key=api_key,
        base_url=get_openrouter_base_url(),
        default_headers=openrouter_default_headers(),
    )
    context_text = _format_retrieved_context(retrieved_context)
    domain_descriptor = domain or "medical imaging"
    system_prompt = (
        "You are a cautious multimodal assistant specialising in {domain}.\n"
        "Base every statement strictly on the retrieved cases or the supplied image.\n"
        "If evidence is missing or conflicting, state the uncertainty clearly and advise follow-up.\n"
        "Output must include two sections:\n"
        "Findings: bullet list grounded in the retrieved cases (reference case numbers).\n"
        "Impression: concise synthesis with confidence level."
    ).format(domain=domain_descriptor)
    user_text = (
        "Retrieved cases (numbered):\n"
        f"{context_text}\n\n"
        f"Question: {question}\n"
        "Instructions:\n"
        "1. Refer to cases as Case X when citing evidence.\n"
        "2. Do NOT invent new findings; if context is insufficient, say so explicitly.\n"
        "3. Note any mismatch between the question image and retrieved cases."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_text}]}
    ]
    if image_path and os.path.exists(image_path):
        try:
            import base64
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            messages[1]["content"].append({"type": "image_url", "image_url": f"data:image/png;base64,{b64}"})
        except Exception as e:
            log.warning("Could not attach image as base64: %s", e)
    try:
        resp = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=600,
            top_p=0.9,
            stream=False,
        )
        try:
            text = resp.choices[0].message.content.strip()
        except Exception:
            text = str(resp)
        return {"success": True, "text": text, "raw": resp}
    except Exception as e:
        log.error("OpenRouter generation failed: %s", e)
        return {"success": False, "text": "", "raw": str(e)}

# MAIN pipeline
def run_rag_pipeline(question: str, image_path: Optional[str] = None, domain_hint: Optional[str] = None) -> Dict[str, Any]:
    log.info("⚙️ Running MED-RAG for question: %s", question)
    domain_info = detect_domain(question, image_path)
    domain = domain_hint or domain_info.get("domain", "radiology")
    if image_path and os.path.exists(image_path):
        query_vec = embed_image(image_path)
        modality = "image"
    else:
        query_vec = embed_text(question)
        modality = "text"
    retrieved_objs = retrieve_adaptive_k(query_vec, modality=modality)
    retrieved_context = []
    for o in retrieved_objs:
        props = o.get("properties", o)
        retrieved_context.append({
            "report_text": props.get("report_text", ""),
            "image_path": props.get("image_path"),
            "projection": props.get("projection"),
            "certainty": float(o.get("_additional", {}).get("certainty", 0.0)),
            "hybrid_score": float(o.get("_hybrid_score", 0.0))
        })
    if not retrieved_context:
        log.info("No Weaviate context available; skipping LLM call.")
        return {
            "question": question,
            "image_path": image_path,
            "domain": domain,
            "domain_info": domain_info,
            "retrieved": [],
            "adaptive_k": 0,
            "generated": {
                "success": False,
                "text": "",
                "raw": "No Weaviate results available for this query.",
            },
            "error": "No Weaviate results found. Ensure data has been ingested.",
        }

    gen = qwen3vl_generate(question, retrieved_context, image_path=image_path, domain=domain)
    out = {
        "question": question,
        "image_path": image_path,
        "domain": domain,
        "domain_info": domain_info,
        "retrieved": retrieved_context,
        "adaptive_k": len(retrieved_context),
        "generated": gen
    }
    log.info("✅ Pipeline finished (generation success=%s)", gen.get("success", False))
    return out

# Exports for other scripts
__all__ = ["run_rag_pipeline", "retrieve_adaptive_k", "embed_text", "embed_image"]
