"""
Evaluation utilities for MED-RAG:
 - precision@k, recall@k
 - BLEU (sacrebleu)
 - ROUGE-L (rouge_score)
 - Hallucination rate (via NLI / entailment)
 - MedFactScore (aggregate entailment score over extracted sentences)

Usage:
    # default (uses roberta-large-mnli on CPU)
    python scripts/eval_metrics.py

    # choose a different NLI model and use GPU 0
    setx NLI_MODEL "your-org/your-medical-nli-model"   # Windows persistent
    set NLI_MODEL=your-org/your-medical-nli-model     # Windows session
    $env:NLI_DEVICE="0"; python .\scripts\eval_metrics.py   # PowerShell example
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from transformers import pipeline, AutoConfig
from tqdm import tqdm
import re
import os
import sys
import math
import warnings
import json

# ----------------------
# Config / environment
# ----------------------
NLI_MODEL = os.getenv("NLI_MODEL", "roberta-large-mnli")
NLI_DEVICE = int(os.getenv("NLI_DEVICE", "-1"))  # -1 -> CPU, 0 -> first GPU
ENTAILMENT_THRESHOLD = float(os.getenv("ENTAILMENT_THRESHOLD", "0.7"))

# If you have a HF token in env (HUGGINGFACE_TOKEN), transformers will pick it up automatically.
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# ----------------------
# Utilities
# ----------------------
def precision_at_k(retrieved: List[Any], relevant: set, k: int) -> float:
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    if len(topk) == 0:
        return 0.0
    return sum(1 for r in topk if r in relevant) / float(len(topk))

def recall_at_k(retrieved: List[Any], relevant: set, k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    topk = retrieved[:k]
    return sum(1 for r in topk if r in relevant) / float(len(relevant))

def compute_bleu(candidate: str, references: List[str]) -> float:
    if not candidate or candidate.strip() == "" or not references:
        return 0.0
    bleu = sacrebleu.corpus_bleu([candidate], [references])
    return float(bleu.score)

def compute_rouge_l(candidate: str, references: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    if not candidate or candidate.strip() == "" or not references:
        return 0.0
    scores = [scorer.score(ref, candidate)['rougeL'].fmeasure for ref in references]
    return float(max(scores) if scores else 0.0)

# ----------------------
# NLI / entailment pipeline
# ----------------------
def prepare_nli_pipeline(model_name: str = None, device: int = -1):
    """
    Prepare an NLI pipeline. Uses text-classification with top_k=None to return all label scores.
    Falls back to roberta-large-mnli if loading fails.
    """
    if model_name is None:
        model_name = "roberta-large-mnli"

    print(f"[eval_metrics] Preparing NLI pipeline: {model_name} on device {device}")
    try:
        # top_k=None returns all scores (similar to return_all_scores=True)
        nli_pipe = pipeline("text-classification", model=model_name, device=device if device >= 0 else -1, top_k=None)
        print(f"[eval_metrics] Loaded NLI model: {model_name}")
        return nli_pipe, model_name
    except Exception as e:
        warnings.warn(f"[eval_metrics] Failed to load NLI model '{model_name}': {e}. Falling back to roberta-large-mnli.")
        try:
            fallback = "roberta-large-mnli"
            nli_pipe = pipeline("text-classification", model=fallback, device=device if device >= 0 else -1, top_k=None)
            print(f"[eval_metrics] Loaded fallback NLI model: {fallback}")
            return nli_pipe, fallback
        except Exception as e2:
            raise RuntimeError(f"[eval_metrics] Failed to load fallback NLI model: {e2}")

_ENTAIL_RE = re.compile(r'(?<=[\.\?\!])\s+')
def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sents = [s.strip() for s in _ENTAIL_RE.split(text) if s.strip()]
    return sents

def entailment_probability(nli_pipeline, premise: str, hypothesis: str) -> float:
    """
    Compute probability that premise entails hypothesis.
    The pipeline returns a list of dicts [{'label':..,'score':..}, ...] under many settings.
    We look for a label indicating entailment (case-insensitive match).
    Returns a float in [0,1].
    """
    if not premise or not hypothesis:
        return 0.0
    try:
        # many NLI models accept two-text inputs via a tuple or dict; transformers pipelines handle pair input as a tuple
        out = nli_pipeline((premise, hypothesis))
    except Exception:
        # fallback: pass as single combined string if model expects that
        out = nli_pipeline(f"{premise} </s> {hypothesis}")

    # normalize output to list of dicts
    # out could be list(list(dict)) for batch or just list(dict)
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        scores = out[0]
    elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        scores = out
    else:
        # unknown format
        try:
            scores = list(out)
        except Exception:
            return 0.0

    # find label that most likely corresponds to entailment
    best_ent_score = 0.0
    for entry in scores:
        label = entry.get("label", "").lower()
        score = float(entry.get("score", 0.0))
        if "enta" in label or "entail" in label:
            best_ent_score = max(best_ent_score, score)
        # some models have 'ENTAILMENT' exactly or localized variants
    # if no explicit entailment label found, pick the label with max score if it's plausibly entailment
    if best_ent_score == 0.0:
        # choose the highest scoring label as a weak proxy (not ideal)
        best_ent_score = max((float(e.get("score", 0.0)) for e in scores), default=0.0)

    return float(best_ent_score)

def hallucination_and_medfact(nli_pipeline, generated: str, retrieved_texts: List[str], entailment_threshold: float = 0.7) -> Tuple[float, float, List[Dict[str,Any]]]:
    """
    - Split generated text into sentences.
    - For each sentence compute max(entailment) across retrieved_texts.
    - If max < entailment_threshold => hallucinated.
    Returns (hallucination_rate, medfact_score, per_sentence_details)
    """
    sents = split_sentences(generated)
    if not sents:
        return 0.0, 0.0, []

    per_sent = []
    max_ent_list = []
    for sent in sents:
        best = 0.0
        best_ctx = None
        for ctx in retrieved_texts:
            p = entailment_probability(nli_pipeline, ctx, sent)
            if p > best:
                best = p
                best_ctx = ctx
            if best >= 0.999:
                break
        is_hall = 1 if best < entailment_threshold else 0
        per_sent.append({"sentence": sent, "max_entailment": best, "hallucinated": bool(is_hall), "best_ctx_sample": None if not best_ctx else (best_ctx[:300] + '...')})
        max_ent_list.append(best)

    hallucination_rate = sum(1 for x in per_sent if x["hallucinated"]) / float(len(per_sent))
    medfact_score = float(sum(max_ent_list) / len(max_ent_list))
    return hallucination_rate, medfact_score, per_sent

# ----------------------
# Wrapper: evaluate_examples
# ----------------------
def evaluate_examples(
    examples: List[Dict[str, Any]],
    ks: List[int] = [1,3,5],
    nli_model: str = None,
    device: int = -1,
    entailment_threshold: float = 0.7
) -> Dict[str, Any]:
    device_id = device if device is not None else -1
    nli_pipeline, used_model = prepare_nli_pipeline(model_name=nli_model or NLI_MODEL, device=device_id)

    per_case = []
    p_at_k = defaultdict(list)
    r_at_k = defaultdict(list)
    bleu_scores = []
    rouge_scores = []
    halluc_rates = []
    medfact_scores = []

    for ex in tqdm(examples, desc="Evaluating"):
        rid = ex.get("id")
        gold = set(ex.get("gold_ids", []))
        retrieved = ex.get("retrieved_ids", [])
        generated = ex.get("generated", "") or ""
        refs = ex.get("reference", [])
        if isinstance(refs, str):
            refs = [refs] if refs.strip() else []
        retrieved_texts = ex.get("retrieved_texts", []) or []

        # retrieval metrics
        for k in ks:
            p = precision_at_k(retrieved, gold, k)
            r = recall_at_k(retrieved, gold, k)
            p_at_k[k].append(p)
            r_at_k[k].append(r)

        # BLEU/ROUGE
        if refs:
            bleu_scores.append(compute_bleu(generated, refs))
            rouge_scores.append(compute_rouge_l(generated, refs))
        else:
            bleu_scores.append(0.0)
            rouge_scores.append(0.0)

        # hallucination & medfact via entailment
        hall, medfact, details = hallucination_and_medfact(nli_pipeline, generated, retrieved_texts, entailment_threshold=entailment_threshold)
        halluc_rates.append(hall)
        medfact_scores.append(medfact)

        per_case.append({
            "id": rid,
            "precision_at_k": {k: precision_at_k(retrieved, gold, k) for k in ks},
            "recall_at_k": {k: recall_at_k(retrieved, gold, k) for k in ks},
            "bleu": bleu_scores[-1],
            "rouge_l": rouge_scores[-1],
            "hallucination_rate": hall,
            "medfact_score": medfact,
            "nli_details_sample": details[:3]
        })

    # aggregate
    agg = {}
    for k in ks:
        agg[f"precision@{k}"] = float(np.mean(p_at_k[k])) if p_at_k[k] else 0.0
        agg[f"recall@{k}"] = float(np.mean(r_at_k[k])) if r_at_k[k] else 0.0

    agg["bleu"] = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    agg["rouge_l"] = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    agg["hallucination_rate"] = float(np.mean(halluc_rates)) if halluc_rates else 0.0
    agg["medfact_score"] = float(np.mean(medfact_scores)) if medfact_scores else 0.0
    agg["nli_model_used"] = used_model
    agg["entailment_threshold"] = entailment_threshold

    return {"aggregate": agg, "per_case": per_case}

# ----------------------
# Example usage when run directly
# ----------------------
if __name__ == "__main__":
    # minimal synthetic examples for demonstration; replace with your eval set
    examples = [
        {
            "id": "case1",
            "gold_ids": {"A","B"},
            "retrieved_ids": ["A","C","D","E"],
            "generated": "Findings: Lungs clear. Impression: No pneumonia.",
            "reference": "No radiographic evidence of pneumonia.",
            "retrieved_texts": ["Lungs clear. Heart normal.", "Cardiomegaly present."]
        },
        {
            "id": "case2",
            "gold_ids": {"X"},
            "retrieved_ids": ["Y","Z"],
            "generated": "Findings: Right lower lobe consolidation consistent with pneumonia.",
            "reference": "Right lower lobe consolidation consistent with pneumonia.",
            "retrieved_texts": ["No consolidation seen.", "Left lower lobe atelectasis."]
        }
    ]

    print("[eval_metrics] Device set to use", "cpu" if NLI_DEVICE < 0 else f"cuda:{NLI_DEVICE}")
    metrics = evaluate_examples(
        examples,
        ks=[1,3],
        nli_model=os.getenv("NLI_MODEL", None),
        device=NLI_DEVICE,
        entailment_threshold=float(os.getenv("ENTAILMENT_THRESHOLD", ENTAILMENT_THRESHOLD))
    )

    print(json.dumps(metrics["aggregate"], indent=2))
    # print first per-case
    print(json.dumps(metrics["per_case"][0], indent=2))
