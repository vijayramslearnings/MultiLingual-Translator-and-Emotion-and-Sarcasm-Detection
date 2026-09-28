"""
Orchestrator Module — Exposes MultilingualVoicePipeline.
A non-agentic, fully deterministic pipeline wiring together Modules 1 through 7 in fixed sequence.
Every stage executes sequentially with zero dynamic planning or tool calling.
"""

import os
import yaml
import json
import torch
from typing import Dict, Any, Optional, List

from pipeline.utils.audio_utils import load_audio, extract_acoustic_features
from pipeline.modules.module1_asr import SpeechRecognizer
from pipeline.modules.module2_lid import LanguageIdentifier
from pipeline.modules.module3_transliteration import TextNormalizer
from pipeline.modules.module4_mt import MachineTranslator
from pipeline.modules.module5_emotion import TextEmotionClassifier, AcousticEmotionClassifier, EmotionFusionLayer
from pipeline.modules.module6_sarcasm import SarcasmDetector
from pipeline.modules.module7_xai import ExplainableAI


class MultilingualVoicePipeline:
    """
    Master pipeline serving class.
    Exposes .process(audio_path, target_lang) -> dict
    """
    def __init__(self, config_path: str = "config/default_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        print(f"[Pipeline Init] Initializing Multilingual Voice Translation System...")
        
        # Instantiate each module in sequence
        self.asr = SpeechRecognizer(self.config)
        self.lid = LanguageIdentifier(self.config)
        self.normalizer = TextNormalizer(self.config)
        self.mt = MachineTranslator(self.config)
        self.emotion_text = TextEmotionClassifier(self.config)
        self.emotion_acoustic = AcousticEmotionClassifier(self.config)
        self.emotion_fusion = EmotionFusionLayer(self.config)
        self.sarcasm = SarcasmDetector(self.config)
        self.xai = ExplainableAI(self.config)
        
        print(f"[Pipeline Init] All 7 deterministic modules successfully loaded and ready.")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def process(
        self, 
        audio_path: str, 
        target_lang: str = "en", 
        context: Optional[List[Dict[str, str]]] = None,
        override_transcript: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes audio input through deterministic pipeline in fixed sequence:
        Audio -> ASR -> LID -> Normalization -> MT -> Emotion (Text/Acoustic/Fusion) -> Sarcasm -> XAI -> JSON Output
        """
        # Load audio waveform for acoustic modules
        waveform, sr = load_audio(audio_path, target_sr=self.config.get("asr", {}).get("sample_rate", 16000))
        acoustic_features = extract_acoustic_features(waveform, sr)

        # Stage 1: ASR (Speech to Text)
        if override_transcript is not None:
            raw_transcript = override_transcript
            asr_res = {"transcript": raw_transcript, "confidence": 1.0, "word_timestamps": []}
        else:
            asr_res = self.asr.transcribe(audio_path)
            raw_transcript = asr_res["transcript"]

        # Stage 2: Language Identification (token/segment level)
        lid_res = self.lid.identify(raw_transcript)
        language_segments = lid_res["language_segments"]

        # Stage 3: Transliteration & Normalization (pre-MT cleanup)
        normalized_transcript = self.normalizer.normalize(raw_transcript, language_segments)

        # Stage 4: Machine Translation
        translated_text = self.mt.translate(normalized_transcript, target_lang=target_lang)

        # Stage 5: Emotion Detection (Text 5a, Acoustic 5b, Fusion 5c)
        text_emo = self.emotion_text.predict(raw_transcript)
        acous_emo = self.emotion_acoustic.predict(audio_path, waveform, sr)
        fused_emo = self.emotion_fusion.fuse(text_emo, acous_emo)

        # Stage 6: Sarcasm Detection
        sarcasm_res = self.sarcasm.detect(
            text_emotion=text_emo,
            acoustic_emotion=acous_emo,
            fusion_emotion=fused_emo,
            acoustic_features=acoustic_features,
            context=context
        )

        # Stage 7: XAI / Explanation Layer
        explanation_res = self.xai.generate_explanations(
            transcript=raw_transcript,
            emotion_label=fused_emo["label"],
            acoustic_emotion_label=acous_emo["label"],
            acoustic_features=acoustic_features
        )

        # Construct final single structured JSON result matching exact specification
        output_json = {
            "transcript": raw_transcript,
            "language_segments": language_segments,
            "translated_text": translated_text,
            "target_language": target_lang,
            "emotion": {
                "label": fused_emo["label"],
                "confidence": fused_emo["confidence"],
                "text_modality": fused_emo["text_modality"],
                "acoustic_modality": fused_emo["acoustic_modality"],
                "modality_agreement": fused_emo["modality_agreement"]
            },
            "sarcasm": {
                "present": sarcasm_res["present"],
                "confidence": sarcasm_res["confidence"],
                "incongruity_score": sarcasm_res["incongruity_score"],
                "reason": sarcasm_res["reason"]
            },
            "explanation": {
                "top_attributed_tokens": explanation_res["top_attributed_tokens"],
                "acoustic_cues": explanation_res["acoustic_cues"]
            }
        }

        return output_json
