# backend/app/core/rag_utils.py
import os
import open_clip
import torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.getenv("BIOMEDCLIP_MODEL", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

# lazy init
_model = None
_preprocess = None
_tokenizer = None

def init_model():
    global _model, _preprocess, _tokenizer
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(MODEL_NAME, device=DEVICE)
        _model.eval()
        _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return _model, _preprocess, _tokenizer

def embed_image(image: Image.Image or str):
    model, preprocess, _tokenizer = init_model()
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_image(x)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
    return feat.squeeze().cpu().numpy()

def embed_text(text: str):
    model, preprocess, tokenizer = init_model()
    tokens = tokenizer(text).to(DEVICE)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
    return feat.squeeze().cpu().numpy()
