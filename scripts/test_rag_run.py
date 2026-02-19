# scripts/test_rag_run.py
import argparse
from backend.app.core.rag_pipeline import run_rag_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="path to image")
parser.add_argument("--question", default="What is the finding?")
parser.add_argument("--domain", default=None)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

res = run_rag_pipeline(args.question, image_path=args.image, domain=args.domain)
print(res.get("answer") or res)
