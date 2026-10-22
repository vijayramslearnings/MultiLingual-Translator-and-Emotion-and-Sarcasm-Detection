"""
Forward Inference Pipeline for Indian Code-Mixed Languages.
Evaluates raw, noisy code-mixed strings through:
1. Phonetic Normalization & Elongation Squashing
2. Emoji Paralinguistic Anchor & Token LID Feature Extraction
3. MuRIL Backbone + Feature Fusion Layer
4. BiLSTM Emotion Classifier Head (6 classes)
5. Sarcasm Detection Head with Contrastive Incongruity Distance Metric
6. IndicTrans2 Local Translation & IndicXlit Transliteration
"""

import json
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

from .preprocessor import CodeMixedPreprocessor, PreprocessedOutput
from .model import MuRILMultiTaskIncongruityModel, MTLModelConfig
from .indic_translation import IndicTrans2LocalEngine, IndicXlitLocalEngine


class CodeMixedInferenceEngine:
    """
    Production-ready Inference Engine for Code-Mixed Indian Language Understanding.
    Resolves Sentiment Context Incongruity and manages noisy phonetic orthography.
    """
    def __init__(
        self,
        config: Optional[MTLModelConfig] = None,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = config or MTLModelConfig()

        print(f"[Engine Init] Initializing CodeMixedInferenceEngine on device '{self.device}'...")
        
        # 1. Preprocessor & Feature Extractor
        self.preprocessor = CodeMixedPreprocessor()

        # 2. Tokenizer (MuRIL Tokenizer or fallback)
        self.tokenizer = self._load_muril_tokenizer()

        # 3. MuRIL Multi-Task Contrastive Model
        self.model = MuRILMultiTaskIncongruityModel(self.cfg)
        self.model = self.model.to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[Engine Init] Loaded trained weights from: {checkpoint_path}")

        self.model.eval()

        # 4. Translation & Transliteration Engines
        self.translator = IndicTrans2LocalEngine(device=self.device)
        self.xlit = IndicXlitLocalEngine()

        print(f"[Engine Init] All neural encoders, transliterators, and MTL heads active.")

    def _load_muril_tokenizer(self):
        import os
        if self.cfg.offline_fast_mode or os.environ.get("INDIC_FAST_MODE") == "1":
            return None
        try:
            from transformers import AutoTokenizer
            try:
                return AutoTokenizer.from_pretrained(self.cfg.pretrained_model_name, local_files_only=True)
            except Exception:
                return AutoTokenizer.from_pretrained(self.cfg.pretrained_model_name, local_files_only=False)
        except Exception as e:
            print(f"[Tokenizer Notice] MuRIL tokenizer loading offline fallback: {e}")
            return None

    def _tokenize(self, text: str, max_length: int = 128):
        if self.tokenizer is not None:
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        else:
            tokens = text.split()[:max_length]
            ids = [101] + [min(190000, abs(hash(t)) % 190000 + 1) for t in tokens] + [102]
            mask = [1] * len(ids)
            if len(ids) < max_length:
                pad_len = max_length - len(ids)
                ids += [0] * pad_len
                mask += [0] * pad_len
            else:
                ids = ids[:max_length]
                mask = mask[:max_length]
            return (
                torch.tensor([ids], dtype=torch.long, device=self.device),
                torch.tensor([mask], dtype=torch.long, device=self.device)
            )

    def evaluate(
        self,
        raw_text: str,
        source_lang: Optional[str] = None,
        target_lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end forward pipeline on a noisy code-mixed input string.
        """
        # ----------------------------------------------------------------------
        # Stage 1: Advanced Preprocessing & Signal Extraction
        # ----------------------------------------------------------------------
        prep: PreprocessedOutput = self.preprocessor.preprocess(raw_text)

        # ----------------------------------------------------------------------
        # Stage 2: Tokenization & Feature Tensor Construction
        # ----------------------------------------------------------------------
        input_ids, attention_mask = self._tokenize(prep.normalized_text)
        aux_tensor = torch.tensor(
            prep.auxiliary_feature_vector,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # (1, 32)

        # ----------------------------------------------------------------------
        # Stage 3: MuRIL Multi-Task Neural Forward Pass
        # ----------------------------------------------------------------------
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                aux_features=aux_tensor
            )

        emotion_probs_tensor = outputs["emotion_probs"].squeeze(0).cpu().numpy()
        sarcasm_probs_tensor = outputs["sarcasm_probs"].squeeze(0).cpu().numpy()
        incongruity_dist = float(outputs["incongruity_distance"].squeeze().cpu().item())

        # Emotion distribution
        emotion_labels = MuRILMultiTaskIncongruityModel.EMOTION_LABELS
        emotion_dist = {
            emotion_labels[i]: round(float(emotion_probs_tensor[i]), 4)
            for i in range(len(emotion_labels))
        }
        top_emotion = max(emotion_dist.items(), key=lambda x: x[1])

        # Sarcasm resolution
        # Heuristic calibration: if surface emotion is positive/joy but incongruity is high or
        # sarcastic paralinguistic anchors (🙃, 😂, 🤡, !!) exist, incorporate contextual tension
        sarcasm_score = float(sarcasm_probs_tensor[1])
        # If emoji sarcasm flag is high or incongruity distance > 0.8, calibrate sarcasm score
        if prep.emoji_dense_vector[3] > 0.70 or prep.emoji_dense_vector[10] > 0.5:
            sarcasm_score = max(sarcasm_score, 0.75 + (prep.emoji_dense_vector[3] * 0.20))
        elif prep.punctuation_intensity["punct_intensity_score"] > 0.6 and top_emotion[0] in ["positive", "joy"] and incongruity_dist > 0.7:
            sarcasm_score = max(sarcasm_score, 0.78)

        is_sarcastic = bool(sarcasm_score >= 0.50)
        sarcasm_confidence = round(float(sarcasm_score if is_sarcastic else (1.0 - sarcasm_score)), 4)

        # ----------------------------------------------------------------------
        # Stage 4: Local IndicTrans2 Translation & IndicXlit Transliteration
        # ----------------------------------------------------------------------
        src_indic_code = "hi"
        if source_lang:
            if "ta" in source_lang.lower():
                src_indic_code = "ta"
            elif "te" in source_lang.lower():
                src_indic_code = "te"
            elif "bn" in source_lang.lower():
                src_indic_code = "bn"
        elif prep.indic_ratio > 0:
            # Check script or sample
            src_indic_code = "hi"

        translated_text = self.translator.translate(
            text=prep.normalized_text,
            src_lang=src_indic_code,
            tgt_lang=target_lang
        )

        transliterated_script = self.xlit.transliterate(
            text=prep.normalized_text,
            target_lang=src_indic_code
        )

        # ----------------------------------------------------------------------
        # Stage 5: Compile Explainable Diagnostic Result
        # ----------------------------------------------------------------------
        reasoning = (
            f"Preprocessed: '{raw_text}' -> '{prep.normalized_text}'. "
            f"Token LID detected {prep.indic_ratio*100:.1f}% Indic, {prep.english_ratio*100:.1f}% English "
            f"(switch density: {prep.switch_density:.2f}). "
            f"Head A literal emotion: {top_emotion[0]} ({top_emotion[1]:.2f}). "
            f"Head B Contrastive Incongruity Distance: {incongruity_dist:.3f}. "
            f"{'High semantic incongruity confirms sarcasm/irony.' if is_sarcastic else 'Congruent semantic alignment indicates literal expression.'}"
        )

        return {
            "input": {
                "raw_text": raw_text,
                "normalized_text": prep.normalized_text,
                "transliterated_brahmic_script": transliterated_script,
            },
            "linguistic_analysis": {
                "tokens": prep.tokens,
                "token_languages": list(zip(prep.tokens, prep.token_lids)),
                "switch_count": prep.switch_count,
                "switch_density": round(prep.switch_density, 4),
                "indic_ratio": round(prep.indic_ratio, 4),
                "english_ratio": round(prep.english_ratio, 4),
                "extracted_emojis": prep.emojis,
                "punctuation_intensity": prep.punctuation_intensity
            },
            "translation": {
                "source_language_inferred": src_indic_code,
                "target_language": target_lang,
                "translated_text": translated_text,
                "engine": "AI4Bharat IndicTrans2 Local"
            },
            "emotion_detection": {
                "predicted_emotion": top_emotion[0],
                "confidence": round(top_emotion[1], 4),
                "emotion_distribution": emotion_dist,
                "head_type": "BiLSTM + GeLU sequence on fused MuRIL representations"
            },
            "sarcasm_detection": {
                "is_sarcastic": is_sarcastic,
                "confidence": sarcasm_confidence,
                "sarcasm_score": round(sarcasm_score, 4),
                "context_incongruity_distance": round(incongruity_dist, 4),
                "loss_mechanism": "Contrastive Incongruity Metric Learning",
                "reasoning": reasoning
            }
        }
