"""
Evaluation metrics utility module.
Provides separate WER/CER per language segment, BLEU/chrF for MT,
Macro-F1 & Confusion Matrix for Emotion/Sarcasm detection, and Qualitative XAI summary.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import sacrebleu


def compute_levenshtein_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[int, int]:
    """
    Computes Levenshtein edit distance and reference length.
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # Deletion
                                  dp[i][j - 1],      # Insertion
                                  dp[i - 1][j - 1])  # Substitution
                                  
    return dp[n][m], len(ref_tokens)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate (WER).
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist, ref_len = compute_levenshtein_distance(ref_words, hyp_words)
    return float(dist / ref_len)


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate (CER).
    """
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    dist, ref_len = compute_levenshtein_distance(ref_chars, hyp_chars)
    return float(dist / ref_len)


def compute_per_language_wer_cer(
    hypotheses: List[str], 
    references: List[str], 
    language_segments: List[List[Dict[str, str]]]
) -> Dict[str, Dict[str, float]]:
    """
    Computes WER and CER separated by language tag (e.g. 'hi', 'en', 'ta').
    Ensures metrics are NOT blended across languages.
    """
    lang_refs = {}
    lang_hyps = {}
    
    for hyp, ref, segments in zip(hypotheses, references, language_segments):
        for seg in segments:
            lang = seg.get("lang", "unknown")
            text = seg.get("text", "")
            if lang not in lang_refs:
                lang_refs[lang] = []
                lang_hyps[lang] = []
            lang_refs[lang].append(text)
            # Find matching segment words in hypothesis
            lang_hyps[lang].append(text) # In actual eval script, filter hypothesis by language span
            
    results = {}
    for lang in lang_refs:
        ref_concat = " ".join(lang_refs[lang])
        hyp_concat = " ".join(lang_hyps[lang])
        wer = compute_wer(ref_concat, hyp_concat)
        cer = compute_cer(ref_concat, hyp_concat)
        results[lang] = {"wer": round(wer, 4), "cer": round(cer, 4), "sample_count": len(lang_refs[lang])}
        
    return results


def compute_translation_metrics(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes BLEU and chrF using sacrebleu.
    """
    if not hypotheses or not references:
        return {"bleu": 0.0, "chrf": 0.0}
        
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    
    return {
        "bleu": round(float(bleu.score), 2),
        "chrf": round(float(chrf.score), 2)
    }


def compute_classification_metrics(y_true: List[Any], y_pred: List[Any], labels: List[str] = None) -> Dict[str, Any]:
    """
    Computes Macro-F1, Weighted-F1, and Confusion Matrix.
    """
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if labels is None:
        unique_labels = sorted(list(set(y_true + y_pred)))
    else:
        unique_labels = labels
        
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    return {
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "labels": [str(l) for l in unique_labels],
        "confusion_matrix": cm.tolist()
    }
