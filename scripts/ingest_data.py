#!/usr/bin/env python3
"""
scripts/ingest_data.py

Ingest script for MED-RAG supporting:
 - radiology (single image + report_text)
 - ophthalmology (paired left/right images possible)
 - pathology (single image + metadata)

Requirements:
 - weaviate-client v3 (>=3.26.7,<4)
 - open-clip (pip install open_clip_torch)
 - transformers, torch, pillow, pandas, tqdm
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import open_clip
import weaviate

# ---------------------------
# Utility helpers
# ---------------------------
def detect_filename_col(df):
    """Detect which column contains filenames."""
    cand = ["filename", "file", "image", "image_path", "img", "left_image", "right_image"]
    for c in cand:
        if c in df.columns:
            return c
    # fallback: any column that looks like filenames
    for c in df.columns:
        if df[c].dtype == object and df[c].str.contains(r"\.(jpg|jpeg|png|bmp|tif|tiff)$", case=False, na=False).any():
            return c
    raise RuntimeError("Could not detect filename column. Rename image column to 'filename' or similar.")

def ensure_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            parsed = json.loads(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [x]
    return [x]

def l2_normalize(vec: np.ndarray):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# ---------------------------
# Ingest implementation
# ---------------------------
REQUIRED_PROPERTIES = [
    {"name": "domain", "dataType": ["string"]},
    {"name": "image_path", "dataType": ["string"]},
    {"name": "left_image_path", "dataType": ["string"]},
    {"name": "right_image_path", "dataType": ["string"]},
    {"name": "report_text", "dataType": ["text"]},
    {"name": "projection", "dataType": ["string"]},
    {"name": "image_embedding", "dataType": ["number[]"]},
    {"name": "left_image_embedding", "dataType": ["number[]"]},
    {"name": "right_image_embedding", "dataType": ["number[]"]},
    {"name": "text_embedding", "dataType": ["number[]"]},
]


def _get_class_schema(client, class_name):
    schema = client.schema.get()
    for clazz in schema.get("classes", []):
        if clazz.get("class") == class_name:
            return clazz
    return None


def _ensure_properties(client, class_name):
    existing = _get_class_schema(client, class_name) or {}
    existing_props = {prop["name"] for prop in existing.get("properties", [])}
    for prop in REQUIRED_PROPERTIES:
        if prop["name"] in existing_props:
            continue
        try:
            client.schema.property.create(class_name, prop)
            print(f"Added missing property '{prop['name']}' to class '{class_name}'.")
        except Exception as exc:
            print(f"Failed to add property '{prop['name']}' to class '{class_name}': {exc}")


def create_schema_if_missing(client, class_name="MedicalReport"):
    if client.schema.exists(class_name):
        print(f"Class '{class_name}' already exists in Weaviate schema.")
        _ensure_properties(client, class_name)
        return

    schema = {
        "class": class_name,
        "vectorizer": "none",
        "properties": REQUIRED_PROPERTIES,
    }
    client.schema.create_class(schema)
    print(f"Created class '{class_name}' in Weaviate schema.")


def load_models(device, biomedclip_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
    print(f"Loading BiomedCLIP (OpenCLIP) model: {biomedclip_name} on {device}")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        biomedclip_name,
        device=device
    )
    model.eval()
    preprocess = preprocess_val
    tokenizer = open_clip.get_tokenizer(biomedclip_name)
    print("✅ BiomedCLIP (OpenCLIP) ready.")
    return model, preprocess, tokenizer


def embed_image_pil(model, preprocess, image_pil, device):
    # image_pil is PIL.Image
    x = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(x)
    emb = emb.cpu().numpy().squeeze()
    emb = emb.astype(np.float32)
    emb = l2_normalize(emb)
    return emb


def embed_text_openclip(model, tokenizer, text, device):
    # tokenizer returns tokens ready for model.encode_text
    if not isinstance(text, str):
        text = str(text)
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
    emb = emb.cpu().numpy().squeeze()
    emb = emb.astype(np.float32)
    emb = l2_normalize(emb)
    return emb

# --- add after client = weaviate.Client(...) in your ingest_data.py ---

def exists_in_weaviate(client, class_name, projection_value):
    """
    Returns True if an object with projection == projection_value exists.
    Uses a simple WHERE filter.
    """
    if not projection_value:
        return False
    try:
        where_filter = {
            "path": ["projection"],
            "operator": "Equal",
            "valueString": str(projection_value)
        }
        resp = (
            client.query
            .get(class_name, ["projection"])
            .with_where(where_filter)
            .with_limit(1)
            .do()
        )
        objs = resp.get("data", {}).get("Get", {}).get(class_name, [])
        return len(objs) > 0
    except Exception as e:
        print("Warning: exists_in_weaviate() failed:", e)
        return False


# ---------------------------
# Main ingestion function
# ---------------------------
def ingest(csv_path,
           image_folder,
           weaviate_url,
           domain,
           class_name="MedicalReport",
           batch_size=32,
           cache_dir="cache",
           device_str=None,
           biomedclip_model="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
           max_items=None):
    device = torch.device("cuda" if torch.cuda.is_available() and (device_str is None or "cpu" not in device_str.lower()) else "cpu")
    print("Device:", device)

    # load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV ({len(df)} rows): {csv_path}")

    # detect filename column
    filename_col = None
    for c in ["filename", "file", "image", "image_path", "img", "Left-Fundus", "Right-Fundus"]:
        if c in df.columns:
            filename_col = c
            break
    left_col = None
    right_col = None
    left_candidates = [
        "Left-Fundus",
        "left_fundus",
        "left_image",
        "left_images",
        "Left_image",
        "leftImg",
    ]
    right_candidates = [
        "Right-Fundus",
        "right_fundus",
        "right_image",
        "right_images",
        "Right_image",
        "rightImg",
    ]
    for c in left_candidates:
        if c in df.columns:
            left_col = c
            break
    for c in right_candidates:
        if c in df.columns:
            right_col = c
            break

    if filename_col is None:
        try:
            filename_col = detect_filename_col(df)
        except RuntimeError:
            filename_col = None
    if filename_col is None and not left_col and not right_col:
        raise RuntimeError("Could not detect filename column. Rename image column to 'filename' or similar.")
    print("Using filename column:", filename_col or "(none)")

    # detect left/right for ophthalmology
    has_left = left_col is not None
    has_right = right_col is not None
    if filename_col:
        name_lower = filename_col.lower()
        has_left = has_left or ("left" in name_lower)
        has_right = has_right or ("right" in name_lower)

    # choose report text column (try common names)
    text_col = None
    for c in ["report_text", "report", "impression", "findings", "report_text_cleaned", "Left-Diagnostic Keywords", "Right-Diagnostic Keywords", "labels", "diagnosis", "label"]:
        if c in df.columns:
            text_col = c
            break
    print("Using text column:", text_col)

    # prepare weaviate client (v3 style)
    client = weaviate.Client(url=weaviate_url)
    # create schema if missing
    create_schema_if_missing(client, class_name)

    # load models
    model, preprocess, tokenizer = load_models(device, biomedclip_name=biomedclip_model)

    # cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # iterate rows
    total = len(df) if max_items is None else min(max_items, len(df))
    it = range(total)

    # support for paired ODIR: if CSV has Left-Fundus and Right-Fundus as separate columns,
    # we will ingest each CSV row as a single object containing whichever side(s) are present.
    # If your CSV is single rows per image, ingestion will ingest each image separately.
    pbar = tqdm(it, ncols=120, desc=f"Ingesting {domain} -> {class_name}")

    for i in pbar:
        row = df.iloc[i]
        # gather image paths
        # Cases:
        # - Ophthalmology: CSV may contain both Left-Fundus and Right-Fundus columns (file names)
        # - Others: CSV has one filename column

        left_path = None
        right_path = None
        single_path = None

        # Ophthalmology pairing detection
        if left_col or right_col:
            left_candidates_vals = ensure_list(row.get(left_col)) if left_col else []
            right_candidates_vals = ensure_list(row.get(right_col)) if right_col else []

            if left_candidates_vals:
                first_left = os.path.basename(str(left_candidates_vals[0]))
                left_path = os.path.join(image_folder, first_left)
            if right_candidates_vals:
                first_right = os.path.basename(str(right_candidates_vals[0]))
                right_path = os.path.join(image_folder, first_right)
        else:
            # fallback to filename_col (could be single-image row or left/right named file)
            fname = row.get(filename_col)
            if pd.isna(fname):
                continue
            fname = os.path.basename(str(fname))
            # if filename contains left/right token, try to split into left/right by pattern (e.g., '26_left.jpg')
            lower = fname.lower()
            if "_left" in lower or "left" in lower:
                left_path = os.path.join(image_folder, fname)
            elif "_right" in lower or "right" in lower:
                right_path = os.path.join(image_folder, fname)
            else:
                single_path = os.path.join(image_folder, fname)

        # build properties
        report_text = ""
        if text_col and pd.notna(row.get(text_col)):
            report_text = str(row.get(text_col))

        projection_value = ""
        if filename_col and filename_col in df.columns and pd.notna(row.get(filename_col)):
            val = row.get(filename_col)
            if isinstance(val, list):
                first_val = val[0] if val else ""
            else:
                first_val = ensure_list(val)[0] if isinstance(val, str) else val
            if first_val:
                projection_value = os.path.basename(str(first_val))
        if not projection_value and left_col and pd.notna(row.get(left_col)):
            left_vals = ensure_list(row.get(left_col))
            if left_vals:
                projection_value = os.path.basename(str(left_vals[0]))
        if not projection_value and "patient_id" in df.columns and pd.notna(row.get("patient_id")):
            projection_value = f"{domain}_{row['patient_id']}"

        properties = {
            "domain": domain,
            "report_text": report_text,
            "projection": projection_value,
        }

        # Embeddings - initialize to None
        image_emb = None
        left_emb = None
        right_emb = None
        text_emb = None

        # compute text embedding (if text exists)
        if report_text:
            try:
                text_emb = embed_text_openclip(model, tokenizer, report_text, device)
                properties["text_embedding"] = text_emb.tolist()
            except Exception as e:
                print("Text embed failed:", e)
                text_emb = None

        # compute image embeddings
        def safe_embed_image(path):
            if not path or not os.path.exists(path):
                return None
            # cache key
            key = os.path.join(cache_dir, f"{Path(path).stem}.npy")
            if os.path.exists(key):
                try:
                    arr = np.load(key)
                    return arr.astype(np.float32)
                except Exception:
                    pass
            try:
                img = Image.open(path).convert("RGB")
                emb = embed_image_pil(model, preprocess, img, device)
                np.save(key, emb)
                return emb
            except Exception as e:
                print(f"Image embedding failed for {path}: {e}")
                return None

        if single_path:
            image_emb = safe_embed_image(single_path)
            if image_emb is not None:
                properties["image_path"] = os.path.basename(single_path)
                properties["image_embedding"] = image_emb.tolist()

        else:
            # handle left/right
            if left_path:
                left_emb = safe_embed_image(left_path)
                if left_emb is not None:
                    properties["left_image_path"] = os.path.basename(left_path)
                    properties["left_image_embedding"] = left_emb.tolist()
                    properties.setdefault("image_path", os.path.basename(left_path))
            if right_path:
                right_emb = safe_embed_image(right_path)
                if right_emb is not None:
                    properties["right_image_path"] = os.path.basename(right_path)
                    properties["right_image_embedding"] = right_emb.tolist()
                    properties.setdefault("image_path", os.path.basename(right_path))

            # if CSV row also had a 'filename' and not left/right separately, store that as image_path
            if not properties.get("image_path") and filename_col and filename_col in row and pd.notna(row.get(filename_col)):
                properties["image_path"] = os.path.basename(str(row.get(filename_col)))

        # Decide which vector to store as object's vector in Weaviate:
        # Priority: average(left_emb,right_emb) if both exist, else image_emb, else left_emb or right_emb, else text_emb
        vector_to_store = None
        if left_emb is not None and right_emb is not None:
            avg = (left_emb + right_emb) / 2.0
            avg = l2_normalize(avg)
            vector_to_store = avg.astype(np.float32)
        elif image_emb is not None:
            vector_to_store = image_emb.astype(np.float32)
        elif left_emb is not None:
            vector_to_store = left_emb.astype(np.float32)
        elif right_emb is not None:
            vector_to_store = right_emb.astype(np.float32)
        elif text_emb is not None:
            vector_to_store = text_emb.astype(np.float32)
        else:
            # nothing to index -- skip
            print(f"Skipping row {i}: no valid image/text found.")
            continue

        # create object in Weaviate
                # ------------------------------
        # Skip if already in Weaviate
        # ------------------------------
        try:
            proj_val = (
                properties.get("projection")
                or properties.get("image_path")
                or None
            )

            if exists_in_weaviate(client, class_name, proj_val):
                # You can uncomment if you want logs:
                # print(f"Skipping row {i}: already exists (projection={proj_val})")
                continue

            # ------------------------------
            # Create new object
            # ------------------------------
            client.data_object.create(
                properties,
                class_name,
                vector=vector_to_store.tolist()
            )

        except Exception as e:
            print(f"Failed to create object for row {i}: {e}")
            time.sleep(0.1)
            continue

    print("Ingestion finished.")


# ---------------------------
# CLI
# ---------------------------
def main_cli():
    parser = argparse.ArgumentParser(description="Ingest datasets into Weaviate for MED-RAG")
    parser.add_argument("--csv", required=True, help="Path to input CSV for the domain (train split).")
    parser.add_argument("--image-folder", required=True, help="Path to folder containing images.")
    parser.add_argument("--weaviate-url", default="http://localhost:8080", help="Weaviate URL (v3 client).")
    parser.add_argument("--domain", required=True, choices=["radiology", "ophthalmology", "pathology"],
                        help="Domain label to store in objects.")
    parser.add_argument("--class-name", default="MedicalReport", help="Weaviate class name.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (not heavily used here).")
    parser.add_argument("--cache-dir", default="cache", help="Cache dir to store numpy embeddings.")
    parser.add_argument("--max-items", type=int, default=None, help="Limit processing to first N rows (for testing).")
    parser.add_argument("--device", default=None, help="cpu or cuda; auto-detected if omitted.")
    parser.add_argument("--biomedclip-model", default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                        help="OpenCLIP model identifier for BiomedCLIP (hf-hub:... recommended).")
    args = parser.parse_args()

    ingest(
        csv_path=args.csv,
        image_folder=args.image_folder,
        weaviate_url=args.weaviate_url,
        domain=args.domain,
        class_name=args.class_name,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        device_str=args.device,
        biomedclip_model=args.biomedclip_model,
        max_items=args.max_items
    )


if __name__ == "__main__":
    main_cli()
