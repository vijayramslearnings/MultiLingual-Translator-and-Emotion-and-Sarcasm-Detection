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
        source_lang: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None,
        override_transcript: Optional[str] = None,
        client_live_transcript: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes audio input through deterministic pipeline in fixed sequence:
        Audio -> ASR -> LID -> Normalization -> MT -> Emotion (Text/Acoustic/Fusion) -> Sarcasm -> XAI -> JSON Output
        Accepts source_lang (e.g. 'ta-en' Tanglish, 'hi-en' Hinglish, 'te-en' Telglish) to guide tokenization and translation.
        """
        # Load audio waveform for acoustic modules
        waveform, sr = load_audio(audio_path, target_sr=self.config.get("asr", {}).get("sample_rate", 16000))
        acoustic_features = extract_acoustic_features(waveform, sr)

        # Stage 1: ASR (Speech to Text)
        if override_transcript is not None:
            raw_transcript = override_transcript
            asr_res = {"transcript": raw_transcript, "confidence": 1.0, "word_timestamps": []}
        else:
            # Primary: Transcribe using local neural Whisper ASR model on GPU/MPS
            asr_res = self.asr.transcribe(audio_path, source_lang=source_lang)
            raw_transcript = asr_res.get("transcript", "").strip()

            # Fallback: If Whisper produced empty text on soft speech, use client live transcript
            if not raw_transcript and client_live_transcript and client_live_transcript.strip():
                raw_transcript = client_live_transcript.strip()
                words = raw_transcript.split()
                step = 3.0 / max(1, len(words))
                word_timestamps = [{"word": w, "start": round(i * step, 2), "end": round((i + 1) * step, 2)} for i, w in enumerate(words)]
                asr_res = {"transcript": raw_transcript, "confidence": 0.90, "word_timestamps": word_timestamps}

        # Stage 2: Language Identification (token/segment level with user source_lang prior)
        lid_res = self.lid.identify(raw_transcript, source_lang=source_lang)
        language_segments = lid_res["language_segments"]

        # Stage 3: Transliteration & Normalization (pre-MT cleanup)
        normalized_transcript = self.normalizer.normalize(raw_transcript, language_segments, source_lang=source_lang)

        # Stage 4: Machine Translation with Phonetic Pronunciation Guide
        mt_detailed = self.mt.translate_detailed(
            normalized_transcript, 
            target_lang=target_lang,
            original_text=raw_transcript,
            source_lang=source_lang
        )
        translated_text = mt_detailed["translated_text"]
        pronunciation_guide = mt_detailed["pronunciation_guide"]
        detected_dialect = mt_detailed.get("detected_language", "Code-Mixed")

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
            context=context,
            raw_text=raw_transcript
        )

        # Stage 7: XAI / Explanation Layer
        explanation_res = self.xai.generate_explanations(
            transcript=raw_transcript,
            emotion_label=fused_emo["label"],
            acoustic_emotion_label=acous_emo["label"],
            acoustic_features=acoustic_features,
            fused_emotion=fused_emo,
            text_emotion=text_emo,
            sarcasm_res=sarcasm_res
        )

        # Construct final single structured JSON result matching exact specification
        output_json = {
            "transcript": raw_transcript,
            "source_language": source_lang or "auto",
            "language_segments": language_segments,
            "detected_dialect": detected_dialect,
            "translated_text": translated_text,
            "pronunciation_guide": pronunciation_guide,
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
                "sarcasm_justification": explanation_res.get("sarcasm_justification", ""),
                "emotion_justification": explanation_res.get("emotion_justification", ""),
                "top_attributed_tokens": explanation_res["top_attributed_tokens"],
                "acoustic_cues": explanation_res["acoustic_cues"]
            }
        }

        return output_json

    def process_text(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: str = "en",
        context: Optional[List[Dict[str, str]]] = None,
        tone_type: str = "neutral"
    ) -> Dict[str, Any]:
        """
        Fast in-memory text processing:
        Runs LID -> Normalization -> MT -> Text Emotion -> Sarcasm -> XAI
        without needing audio file I/O.
        """
        raw_transcript = (text or "").strip()
        if not raw_transcript:
            return {
                "transcript": "",
                "source_language": source_lang or "auto",
                "language_segments": [],
                "detected_dialect": "Empty",
                "translated_text": "",
                "pronunciation_guide": "",
                "target_language": target_lang,
                "emotion": {"label": "neutral", "confidence": 1.0, "text_modality": {"label": "neutral", "confidence": 1.0}, "acoustic_modality": {"label": "neutral", "confidence": 1.0}, "modality_agreement": True},
                "sarcasm": {"present": False, "confidence": 1.0, "incongruity_score": 0.0, "reason": "empty input"},
                "explanation": {"top_attributed_tokens": [], "acoustic_cues": []}
            }

        # Stage 2: LID
        lid_res = self.lid.identify(raw_transcript, source_lang=source_lang)
        language_segments = lid_res["language_segments"]

        # Stage 3: Normalization
        normalized_transcript = self.normalizer.normalize(raw_transcript, language_segments, source_lang=source_lang)

        # Stage 4: MT
        mt_detailed = self.mt.translate_detailed(
            normalized_transcript,
            target_lang=target_lang,
            original_text=raw_transcript,
            source_lang=source_lang
        )
        translated_text = mt_detailed["translated_text"]
        pronunciation_guide = mt_detailed["pronunciation_guide"]
        detected_dialect = mt_detailed.get("detected_language", "Code-Mixed")

        # Stage 5: Emotion (Text)
        text_emo = self.emotion_text.predict(raw_transcript)
        # Synthetic neutral acoustic baseline for text-only processing
        acous_emo = {"label": tone_type or "neutral", "confidence": 0.50, "scores": {tone_type or "neutral": 0.50}}
        fused_emo = self.emotion_fusion.fuse(text_emo, acous_emo)

        mock_acoustic_features = {
            "pitch_variance": 5.0,
            "spectral_centroid_variance": 120.0,
            "harmonics_ratio": 0.85
        }

        # Stage 6: Sarcasm
        sarcasm_res = self.sarcasm.detect(
            text_emotion=text_emo,
            acoustic_emotion=acous_emo,
            fusion_emotion=fused_emo,
            acoustic_features=mock_acoustic_features,
            context=context,
            raw_text=raw_transcript
        )

        # Stage 7: XAI
        explanation_res = self.xai.generate_explanations(
            transcript=raw_transcript,
            emotion_label=fused_emo["label"],
            acoustic_emotion_label=acous_emo["label"],
            acoustic_features=mock_acoustic_features,
            fused_emotion=fused_emo,
            text_emotion=text_emo,
            sarcasm_res=sarcasm_res
        )

        return {
            "transcript": raw_transcript,
            "source_language": source_lang or "auto",
            "language_segments": language_segments,
            "detected_dialect": detected_dialect,
            "translated_text": translated_text,
            "pronunciation_guide": pronunciation_guide,
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
                "sarcasm_justification": explanation_res.get("sarcasm_justification", ""),
                "emotion_justification": explanation_res.get("emotion_justification", ""),
                "top_attributed_tokens": explanation_res["top_attributed_tokens"],
                "acoustic_cues": explanation_res["acoustic_cues"]
            }
        }
