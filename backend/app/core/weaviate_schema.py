# backend/app/core/weaviate_schema.py
import os
import weaviate
from dotenv import load_dotenv

load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
CLASS_NAME = "MedicalReport"

client = weaviate.Client(url=WEAVIATE_URL)

def create_schema():
    if client.schema.exists(CLASS_NAME):
        print(f"Class '{CLASS_NAME}' already exists.")
        return

    schema = {
        "class": CLASS_NAME,
        "vectorizer": "none",
        "properties": [
            {"name": "patient_id", "dataType": ["string"]},
            {"name": "domain", "dataType": ["string"]},
            {"name": "eye", "dataType": ["string"]},
            {"name": "image_path", "dataType": ["string"]},
            {"name": "report_text", "dataType": ["text"]},
            {"name": "label", "dataType": ["string"]},
            {"name": "image_embedding", "dataType": ["number[]"]},
            {"name": "text_embedding", "dataType": ["number[]"]}
        ]
    }
    client.schema.create_class(schema)
    print(f"Created class '{CLASS_NAME}' in Weaviate schema.")

if __name__ == "__main__":
    create_schema()
