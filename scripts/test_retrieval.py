import weaviate
import numpy as np
from PIL import Image
import open_clip
import torch
import os

WEAVIATE_URL = "http://localhost:8080"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load BiomedCLIP (same model as ingestion!)
model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    device=DEVICE
)
model.eval()
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')


def embed_image(path):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        v = model.encode_image(x)
        v /= v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().flatten()


def test_retrieval(image_path):
    client = weaviate.Client(WEAVIATE_URL)

    print("Embedding query image...")
    q = embed_image(image_path)

    print("Querying Weaviate...")
    result = (
        client.query
        .get("MedicalReport", ["image_path", "report_text", "_additional {certainty}"])
        .with_near_vector({"vector": q.tolist()})
        .with_limit(10)
        .do()
    )

    hits = result["data"]["Get"]["MedicalReport"]
    print(f"\nTop {len(hits)} retrieved results:")

    for i, h in enumerate(hits):
        print(f"\nRank {i+1}")
        print("certainty:", h["_additional"]["certainty"])
        print("image:", h["image_path"])
        print("text:", h["report_text"][:150], "...")


if __name__ == "__main__":
    test_image = "C:/Users/krith/OneDrive/Desktop/MED-RAG/data/IUXRAY/images/images_normalized/2000_IM-0654-1001.dcm.png"
    test_retrieval(test_image)
