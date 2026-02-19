# backend/app/core/domain_classifier.py
import os
import numpy as np
from dotenv import load_dotenv
from typing import Tuple
from backend.app.core.rag_utils import embed_text, embed_image  # we'll add rag_utils next

load_dotenv()

# small textual prototypes per domain
DOMAIN_PROTOTYPES = {
    "radiology": "chest x-ray radiograph lungs mediastinum CT MRI radiology report",
    "ophthalmology": "fundus retina optic disc macula diabetic retinopathy ophthalmology",
    "pathology": "histology slide pathology H&E microscopic tissue pathology report"
}

def build_domain_embeddings(embed_fn):
    out = {}
    for d, txt in DOMAIN_PROTOTYPES.items():
        emb = embed_fn(txt)
        # normalize
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        out[d] = emb
    return out

# lazy generate
_domain_embs = None
def get_domain_embeddings(embed_fn):
    global _domain_embs
    if _domain_embs is None:
        _domain_embs = build_domain_embeddings(embed_fn)
    return _domain_embs

def predict_domain_from_image(image_np_or_path, embed_image_fn, threshold=0.25) -> Tuple[str,float]:
    """
    Returns (domain, score). Score is cosine similarity to best prototype.
    """
    if isinstance(image_np_or_path, str):
        q = embed_image_fn(image_np_or_path)  # embed_image returns numpy vector
    else:
        q = image_np_or_path
    q = q / (np.linalg.norm(q) + 1e-12)
    dom_embs = get_domain_embeddings(lambda t: embed_text(t))  # uses embed_text from rag_utils
    scores = {d: float(np.dot(q, v)) for d,v in dom_embs.items()}
    best = max(scores, key=scores.get)
    return best, scores[best]
