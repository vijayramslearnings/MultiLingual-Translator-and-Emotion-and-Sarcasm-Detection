"""
Module 5 — Emotion Detection (Text, Acoustic & Late Fusion Layer)
5a Text Emotion Classifier (HF XLM-R / DistilRoBERTa) -> Softmax probability distribution
5b Acoustic Emotion Classifier (Wav2Vec2 / Acoustic Spectral Features) -> Softmax probability distribution
5c Late Fusion Layer -> Combines 5a + 5b, computes agreement flag, returns final label & confidence.
"""

import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List
from pipeline.utils.audio_utils import extract_acoustic_features


class TextEmotionClassifier:
    """Module 5a: Text Emotion Detection (Hugging Face Transformers / MuRIL MTL)"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("emotion_text", {})
        self.backend = self.config.get("backend", "transformers")
        self.model_name = self.config.get("model_name", "j-hartmann/emotion-english-distilroberta-base")
        self.labels = self.config.get("labels", ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"])
        self.classifier = None
        self.muril_engine = None
        self._initialize_backend()

    def _initialize_backend(self):
        if self.backend == "muril_mtl" or self.config.get("use_muril_mtl"):
            try:
                from pipeline.codemix_mtl import CodeMixedInferenceEngine
                self.muril_engine = CodeMixedInferenceEngine()
                return
            except Exception as e:
                print(f"[Text Emotion Warning] Could not initialize MuRIL MTL engine: {e}. Falling back to standard pipeline.")

        try:
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self.classifier = pipeline("text-classification", model=self.model_name, top_k=None, device=device)
        except Exception as e:
            print(f"[Text Emotion Warning] Could not load text emotion model '{self.model_name}': {e}. Using deterministic text emotion classifier.")
            self.classifier = None

    # Affective Lexicon for Code-Mixed Indian Conversations (Hindi, Tamil, English)
    CODE_MIX_EMOTIONS = {
        "joy": [
            "happy", "joy", "great", "excellent", "good", "awesome", "super", "wonderful", "love", "lovely",
            "blessed", "glad", "delighted", "proud", "congrats", "congratulations", "thanks", "thank you",
            "shukriya", "dhanyavad", "nandri", "badhiya", "bahut badhiya", "bohot achha", "kamaal", "khush",
            "khushi", "pyaar", "mazza", "maza", "nalla", "romba nalla", "semma", "mass", "gethu", "azhaga",
            "bagundi", "chala bagundi", "chennagide", "sundaram", "adbhut"
        ],
        "anger": [
            "angry", "anger", "furious", "mad", "annoyed", "frustrated", "irritated", "hate", "worst",
            "terrible", "horrible", "useless", "stupid", "idiot", "nonsense", "rubbish", "cheat",
            "gussa", "bekar", "bilkul bekar", "ganda", "ghatiya", "bakwas", "kopam", "kevalam",
            "kevalama", "sanda", "sandai", "erichal", "thappu", "waste", "cheat", "scam"
        ],
        "sadness": [
            "sad", "sadness", "unhappy", "depressed", "heartbroken", "crying", "cry", "tears", "pain",
            "lonely", "sorry", "miss you", "grief", "loss", "mourning", "regret",
            "dukh", "dard", "rona", "udas", "udasi", "kavala", "valikidhu", "azhugai", "thunbam",
            "badha", "kashtam", "marandhuten"
        ],
        "fear": [
            "fear", "afraid", "scared", "frightened", "terrified", "panic", "worried", "worry",
            "nervous", "anxious", "anxiety", "danger", "threat",
            "dar", "darr", "khauf", "tension", "bayama", "bayam", "payam", "bayandhuten", "bhayam"
        ],
        "surprise": [
            "surprise", "surprised", "shock", "shocked", "shocking", "amazed", "astonished", "unbelievable",
            "unexpected", "omg", "wow", "arrey", "arre", "wah", "waah", "aaha", "aacharyam", "enna da"
        ],
        "disgust": [
            "disgust", "disgusted", "disgusting", "gross", "nasty", "dirty", "yuck", "sickening",
            "chhi", "chee", "thoo", "karumam"
        ]
    }

    def predict(self, text: str) -> Dict[str, Any]:
        """Returns softmax distribution across emotion classes, top label, and neural incongruity signals."""
        clean_text = (text or "").strip()
        if not clean_text:
            return {
                "label": "neutral",
                "confidence": 0.85,
                "probabilities": {"neutral": 0.85, "joy": 0.05, "sadness": 0.03, "anger": 0.03, "surprise": 0.02, "disgust": 0.01, "fear": 0.01},
                "raw_text": ""
            }

        # Check for MuRIL MTL engine if loaded
        if self.muril_engine is not None:
            try:
                res = self.muril_engine.evaluate(clean_text)
                emo = res["emotion_detection"]
                sarc = res["sarcasm_detection"]
                return {
                    "label": emo["predicted_emotion"],
                    "confidence": emo["confidence"],
                    "probabilities": emo["emotion_distribution"],
                    "neural_incongruity": sarc["context_incongruity_distance"],
                    "sarcasm_score": sarc["sarcasm_score"],
                    "is_sarcastic": sarc["is_sarcastic"],
                    "raw_text": clean_text
                }
            except Exception as ex:
                pass

        # 1. DistilRoBERTa Transformer inference
        prob_dict = {}
        if self.classifier is not None:
            try:
                words = clean_text.split()[:200]
                safe_text = " ".join(words)
                results = self.classifier(safe_text, truncation=True, max_length=512)[0]
                prob_dict = {res["label"].lower(): float(res["score"]) for res in results}
            except Exception as ex:
                pass

        # 2. Code-Mixed Affective Lexicon matching
        low = clean_text.lower()
        lex_scores = {lbl: 0.0 for lbl in self.labels}
        for emo_lbl, words in self.CODE_MIX_EMOTIONS.items():
            for w in words:
                if re.search(r'\b' + re.escape(w) + r'\b', low):
                    lex_scores[emo_lbl] += 0.35

        # Merge transformer probabilities with code-mix lexicon signals
        if prob_dict:
            combined = {}
            for lbl in self.labels:
                trans_score = prob_dict.get(lbl, 0.01)
                lex_boost = lex_scores.get(lbl, 0.0)
                combined[lbl] = trans_score + lex_boost
        else:
            combined = lex_scores.copy()
            if sum(combined.values()) == 0:
                combined["neutral"] = 0.70

        # Normalize to valid probability distribution
        total = sum(combined.values())
        if total > 0:
            prob_dict = {k: round(v / total, 4) for k, v in combined.items()}
        else:
            prob_dict = {"neutral": 0.70, "joy": 0.15, "sadness": 0.05, "anger": 0.05, "surprise": 0.03, "disgust": 0.01, "fear": 0.01}

        # Check for sarcasm contrast in text (e.g. positive praise + failure)
        # Sarcasm detection handles final sarcasm flag, but compute lexical sarcasm score here
        sarcasm_score = 0.0
        from pipeline.modules.module6_sarcasm import SarcasmDetector
        has_contrast, c_score, _, _ = SarcasmDetector._detect_lexical_contrast(clean_text)
        if has_contrast:
            sarcasm_score = c_score
            # If surface expression was mock-joy ("great job breaking server"), true emotion is anger/frustration or surprise
            if prob_dict.get("joy", 0) > 0.35:
                prob_dict["anger"] = round(max(0.70, prob_dict.get("joy", 0.7)), 4)
                prob_dict["surprise"] = 0.15
                prob_dict["joy"] = 0.05

        top_item = max(prob_dict.items(), key=lambda x: x[1])
        return {
            "label": top_item[0],
            "confidence": round(top_item[1], 4),
            "probabilities": prob_dict,
            "sarcasm_score": sarcasm_score,
            "raw_text": clean_text
        }


class AcousticEmotionClassifier:
    """Module 5b: Acoustic Emotion Detection from raw speech signal & timestamps"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("emotion_acoustic", {})
        self.model_name = self.config.get("model_name", "superb/wav2vec2-base-superb-er")
        self.labels = ["joy", "anger", "sadness", "neutral", "surprise"]
        self.model = None
        self._initialize_backend()

    def _initialize_backend(self):
        try:
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self.model = pipeline("audio-classification", model=self.model_name, top_k=None, device=device)
        except Exception as e:
            print(f"[Acoustic Emotion Warning] Could not load wav2vec2 SER model '{self.model_name}': {e}. Using acoustic spectral emotion engine.")
            self.model = None

    def predict(self, audio_path: str, waveform: torch.Tensor, sample_rate: int = 16000) -> Dict[str, Any]:
        """Extracts acoustic features and predicts emotion probability distribution."""
        np_wave = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        if self.model is not None:
            try:
                input_data = {"raw": np_wave, "sampling_rate": sample_rate}
                res = self.model(input_data)
                label_map = {
                    "happy": "joy",
                    "hap": "joy",
                    "angry": "anger",
                    "ang": "anger",
                    "neu": "neutral",
                    "sad": "sadness"
                }
                prob_dict = {}
                for item in res:
                    raw_lbl = item["label"].lower().strip()
                    lbl = label_map.get(raw_lbl, raw_lbl)
                    prob_dict[lbl] = float(item["score"])
                
                for c in self.labels:
                    if c not in prob_dict:
                        prob_dict[c] = 0.01

                top_item = max(prob_dict.items(), key=lambda x: x[1])
                return {
                    "label": top_item[0],
                    "confidence": round(top_item[1], 4),
                    "probabilities": {k: round(v, 4) for k, v in prob_dict.items()}
                }
            except Exception as ex:
                pass

        features = extract_acoustic_features(waveform, sample_rate)
        energy_std = features["std_energy"]
        pitch_var = features["pitch_variance"]

        if features["max_energy_spike"] > 0.6 and pitch_var > 0.05:
            prob_dict = {"anger": 0.78, "joy": 0.10, "neutral": 0.07, "sadness": 0.03, "surprise": 0.02}
        elif pitch_var > 0.02:
            prob_dict = {"joy": 0.72, "surprise": 0.15, "neutral": 0.08, "anger": 0.03, "sadness": 0.02}
        else:
            prob_dict = {"neutral": 0.65, "sadness": 0.20, "joy": 0.10, "anger": 0.03, "surprise": 0.02}

        top_item = max(prob_dict.items(), key=lambda x: x[1])
        return {
            "label": top_item[0],
            "confidence": round(top_item[1], 4),
            "probabilities": prob_dict,
            "acoustic_features": features
        }


class EmotionFusionLayer:
    """Module 5c: Late Fusion Layer combining Text and Acoustic Emotion Predictions"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("fusion", {})
        self.text_weight = self.config.get("text_weight", 0.55)
        self.acoustic_weight = self.config.get("acoustic_weight", 0.45)

    def fuse(self, text_res: Dict[str, Any], acoustic_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combines 5a + 5b via weighted average softmax distribution.
        Outputs final emotion label, confidence, per-modality confidence, and modality_agreement flag.
        """
        text_probs = text_res.get("probabilities", {})
        acoustic_probs = acoustic_res.get("probabilities", {})

        all_labels = set(text_probs.keys()).union(set(acoustic_probs.keys()))
        fused_probs = {}

        for lbl in all_labels:
            p_text = text_probs.get(lbl, 0.0)
            p_acous = acoustic_probs.get(lbl, 0.0)
            fused_probs[lbl] = (self.text_weight * p_text) + (self.acoustic_weight * p_acous)

        total_p = sum(fused_probs.values())
        if total_p > 0:
            fused_probs = {k: v / total_p for k, v in fused_probs.items()}

        top_label, fused_confidence = max(fused_probs.items(), key=lambda x: x[1])
        modality_agreement = (text_res["label"] == acoustic_res["label"])

        return {
            "label": top_label,
            "confidence": round(float(fused_confidence), 4),
            "text_modality": {
                "label": text_res["label"],
                "confidence": round(float(text_res["confidence"]), 4)
            },
            "acoustic_modality": {
                "label": acoustic_res["label"],
                "confidence": round(float(acoustic_res["confidence"]), 4)
            },
            "modality_agreement": modality_agreement,
            "fused_probabilities": {k: round(v, 4) for k, v in fused_probs.items()}
        }
