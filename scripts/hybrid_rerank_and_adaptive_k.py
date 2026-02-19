# scripts/hybrid_rerank_and_adaptive_k.py
import weaviate
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from PIL import Image
import open_clip, torch
import os

WEAVIATE_URL = "http://localhost:8080"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

# config
MAX_CANDIDATES = 40
MIN_K = 2
SIM_THRESHOLD = 0.75   # you can tune
WEIGHT_WEAV = 0.5
WEIGHT_TEXT = 0.3
WEIGHT_IMG  = 0.2

# load model (same as ingest)
model, _, preprocess = open_clip.create_model_and_transforms(MODEL, device=DEVICE)
tokenizer = open_clip.get_tokenizer(MODEL)
model.eval()

def embed_image_np(path):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().flatten()

def embed_text_np(text):
    tokens = tokenizer(text).to(DEVICE)
    with torch.no_grad():
        v = model.encode_text(tokens)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().flatten()

def cos(a,b):
    if a is None or b is None:
        return 0.0
    return float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0,0])

def hybrid_rerank(query_vector, candidates):
    # candidates: list of weaviate hits (dicts with text_embedding, image_embedding, _additional.certainty)
    scored = []
    certs = []
    for r in candidates:
        cert = r["_additional"]["certainty"]
        certs.append(cert)
    certs_arr = np.array(certs)
    # compute pairwise
    for r in candidates:
        weav_cert = r["_additional"]["certainty"]
        text_emb = np.array(r.get("text_embedding")) if r.get("text_embedding") else None
        img_emb  = np.array(r.get("image_embedding")) if r.get("image_embedding") else None
        s_text = cos(query_vector, text_emb)
        s_img  = cos(query_vector, img_emb)
        score = WEIGHT_WEAV * weav_cert + WEIGHT_TEXT * s_text + WEIGHT_IMG * s_img
        scored.append((score, weav_cert, s_text, s_img, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    # adaptive-K by largest drop in certs among scored order
    ordered_certs = [s[1] for s in scored]
    if len(ordered_certs) <= MIN_K:
        k = len(ordered_certs)
    else:
        diffs = np.diff(ordered_certs)
        drop_idx = int(np.argmax(np.abs(diffs))) + 1
        k = max(MIN_K, min(drop_idx, len(ordered_certs)))
        # safeguards
        if ordered_certs[0] > SIM_THRESHOLD: k = max(k, 3)
        if ordered_certs[-1] > SIM_THRESHOLD: k = len(ordered_certs)
    return scored, k

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--limit", type=int, default=MAX_CANDIDATES)
    args = parser.parse_args()

    client = weaviate.Client(WEAVIATE_URL)
    qvec = embed_image_np(args.image)
    resp = client.query.get("MedicalReport", ["image_path","report_text","text_embedding","image_embedding","_additional{certainty}"])\
        .with_near_vector({"vector": qvec.tolist()}).with_limit(args.limit).do()
    hits = resp["data"]["Get"]["MedicalReport"]
    scored, k = hybrid_rerank(qvec, hits)
    print(f"Retrieved {len(hits)} -> hybrid top {len(scored)}, adaptive-K = {k}\n")
    out = []
    for i,(score, weav_c, s_text, s_img, r) in enumerate(scored[:max(10,k)]):
        out.append({
            "rank": i+1,
            "combined_score": score,
            "weav_cert": weav_c,
            "cos_text": s_text,
            "cos_img": s_img,
            "image_path": r["image_path"],
            "report_text": (r["report_text"][:300] + "...") if r.get("report_text") else ""
        })
    print(json.dumps(out, indent=2))
    print("\nAdaptive K:", k)
