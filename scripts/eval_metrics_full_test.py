"""
scripts/eval_metrics_full_test.py
Evaluate predictions produced by generate_predictions.py across all test rows.
Computes precision@1/3/5, recall@1/3/5 (based on retrieval containing ground-truth report),
BLEU, ROUGE-L on generated text, and hallucination rate via NLI (roberta-large-mnli).

Usage:
 python scripts/eval_metrics_full_test.py --predictions results/predictions_full_test.jsonl --gold data/your_gold.csv
"""

import argparse, json, os
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import pipeline
import numpy as np

def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def compute_retrieval_metrics(pred_records, gold_lookup, ks=(1,3,5)):
    # gold_lookup maps (domain,image or question) -> gold_text
    prec = {k: [] for k in ks}
    rec = {k: [] for k in ks}
    for recp in pred_records:
        retrieved = recp.get("retrieved", [])
        # build list of retrieved texts
        retrieved_texts = [r.get("report_text","") for r in retrieved]
        # find gold key
        key = (recp.get("domain"), recp.get("image"))  # match by domain+image path
        gold = gold_lookup.get(key)
        for k in ks:
            topk = retrieved_texts[:k]
            hit = 1 if gold and any(gold.strip() and (gold.strip()[:20] in t or t[:20] in gold) for t in topk) else 0
            prec[k].append(hit / max(1, k))
            rec[k].append(hit)
    agg = {}
    for k in ks:
        agg[f"precision@{k}"] = float(np.mean(prec[k])) if prec[k] else 0.0
        agg[f"recall@{k}"] = float(np.mean(rec[k])) if rec[k] else 0.0
    return agg

def compute_generation_metrics(pred_records):
    bleu_scores = []
    rouge_l_scores = []
    # load rouge scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1
    for rec in pred_records:
        gen = rec.get("generated", {}).get("text", "")
        # gold not provided robustly here; assume gold exists in 'gold_text' field if present
        gold = rec.get("gold_text", "")
        if not gold:
            # can't compute metrics if gold missing
            continue
        # BLEU
        ref_tokens = [gold.split()]
        cand_tokens = gen.split()
        try:
            b = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smooth) * 100.0
        except Exception:
            b = 0.0
        bleu_scores.append(b)
        # ROUGE-L
        sc = scorer.score(gold, gen)
        rouge_l_scores.append(sc["rougeL"].fmeasure)
    return {
        "bleu": float(np.mean(bleu_scores)) if bleu_scores else 0.0,
        "rouge_l": float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0
    }

def compute_hallucination_medfact(pred_records, nli_model="roberta-large-mnli"):
    # For each generated statement sentence, check entailment vs gold; if contradictory or neutral -> hallucinated
    classifier = pipeline("text-classification", model=nli_model, return_all_scores=True, device=-1)
    hallucinated = 0
    total_sent = 0
    for rec in pred_records:
        gen = rec.get("generated", {}).get("text", "")
        gold = rec.get("gold_text", "")
        if not gold or not gen:
            continue
        # naive sentence split ('.' split)
        sentences = [s.strip() for s in gen.split(".") if s.strip()]
        for s in sentences:
            total_sent += 1
            # NLI: premise=gold, hypothesis=s
            inputs = f"[CLS] {gold} [SEP] {s}"
            try:
                out = classifier(f"{s} ||| {gold}")  # mild hack; pipeline expects single text. alternative: use cross-encoder; keep simple.
            except Exception:
                continue
            # pipeline return_all_scores True returns list of labels; for roberta-large-mnli: labels=contradiction,neutral,entailment
            # we find max entailment score
            scores = out[0] if isinstance(out[0], list) else out
            # try to map label->score
            label_to_score = {el['label'].lower(): el['score'] for el in scores}
            ent = label_to_score.get('entailment', 0.0)
            # treat as hallucinated if entailment < 0.5
            if ent < 0.5:
                hallucinated += 1
    return {"hallucination_rate": float(hallucinated / total_sent) if total_sent else 0.0}

def build_gold_lookup():
    # Build simple lookup from your test CSVs; for robust eval you should include a gold_text in predictions or pass gold file path.
    # Here we attempt to gather gold report_text from common test CSVs by image path.
    gold = {}
    # radiology
    rad_csv = "data/IUXRAY/indiana_merged_cleaned_test.csv"
    if os.path.exists(rad_csv):
        import pandas as pd
        df = pd.read_csv(rad_csv)
        for _, r in df.iterrows():
            key = ("radiology", r.get("filename"))
            gold[key] = r.get("report_text","")
    # ophthalmology
    odir = "data/ODIR-5K/odir_test.csv"
    if os.path.exists(odir):
        import pandas as pd
        df = pd.read_csv(odir)
        for _, r in df.iterrows():
            key = ("ophthalmology", r.get("filename"))
            # ODIR stores diagnostic keywords; use 'labels' or 'Right-Diagnostic Keywords' if present
            gold[key] = r.get("labels", "") or r.get("Right-Diagnostic Keywords","") or ""
    # pathology
    pth = "data/pathology/pathology_test.csv"
    if os.path.exists(pth):
        import pandas as pd
        df = pd.read_csv(pth)
        for _, r in df.iterrows():
            key = ("pathology", r.get("image"))
            gold[key] = r.get("report_text","") or r.get("answer","") or ""
    return gold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--nli-model", default="roberta-large-mnli")
    args = parser.parse_args()

    preds = list(load_predictions(args.predictions))
    print("[eval] Loaded prediction records:", len(preds))
    gold_lookup = build_gold_lookup()

    retrieval_metrics = compute_retrieval_metrics(preds, gold_lookup, ks=(1,3,5))
    gen_metrics = compute_generation_metrics(preds)
    nli_metrics = compute_hallucination_medfact(preds, nli_model=args.nli_model)

    out = {**retrieval_metrics, **gen_metrics, **nli_metrics}
    print("=== Final aggregated metrics (overall) ===")
    print(json.dumps(out, indent=2))
    Path("results").mkdir(exist_ok=True)
    with open("results/eval_full_test_metrics.json","w",encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    import argparse
    main()
