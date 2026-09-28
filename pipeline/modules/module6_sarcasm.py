"""
Module 6 — Sarcasm Detection
Core signal: Incongruity between text-predicted sentiment/emotion and acoustic-predicted sentiment/tone.
(e.g., text sentiment: positive 0.82 vs acoustic tone: negative/angry 0.77 -> high incongruity -> sarcasm likely).
Outputs:
  - present: bool
  - confidence: float
  - incongruity_score: float
  - reason: str
Supports optional conversational context (prior dialogue turns).
"""

from typing import Dict, Any, List, Optional


class SarcasmDetector:
    # Valence score mapping for emotion classes
    EMOTION_VALENCE = {
        "joy": 0.90,
        "happy": 0.90,
        "hap": 0.90,
        "surprise": 0.60,
        "neutral": 0.00,
        "neu": 0.00,
        "sadness": -0.70,
        "sad": -0.70,
        "fear": -0.75,
        "disgust": -0.80,
        "anger": -0.85,
        "ang": -0.85
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("sarcasm", {})
        self.incongruity_threshold = self.config.get("incongruity_threshold", 0.40)
        self.sentiment_weight = self.config.get("weights", {}).get("sentiment_mismatch", 0.70)
        self.pitch_weight = self.config.get("weights", {}).get("acoustic_pitch_spike", 0.30)

    def detect(
        self, 
        text_emotion: Dict[str, Any], 
        acoustic_emotion: Dict[str, Any], 
        fusion_emotion: Dict[str, Any],
        acoustic_features: Optional[Dict[str, Any]] = None,
        context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Detects sarcasm via text-acoustic incongruity.
        """
        text_label = text_emotion.get("label", "neutral")
        text_conf = text_emotion.get("confidence", 0.5)
        
        acoustic_label = acoustic_emotion.get("label", "neutral")
        acoustic_conf = acoustic_emotion.get("confidence", 0.5)

        v_text = self.EMOTION_VALENCE.get(text_label, 0.0) * text_conf
        v_acous = self.EMOTION_VALENCE.get(acoustic_label, 0.0) * acoustic_conf

        # 1. Primary Sentiment/Valence Incongruity
        # Range [0.0, 2.0] normalized to [0.0, 1.0]
        valence_diff = abs(v_text - v_acous) / 2.0

        # 2. Acoustic Pitch Variance / Spike Bonus
        pitch_spike = 0.0
        if acoustic_features:
            pitch_var = acoustic_features.get("pitch_variance", 0.0)
            if pitch_var > 0.04:
                pitch_spike = min(1.0, pitch_var * 10.0)

        # 3. Contextual Adjustment (if prior turns indicate serious or contrary setup)
        context_modifier = 0.0
        if context and len(context) > 0:
            last_turn = context[-1].get("text", "").lower()
            if any(w in last_turn for w in ["really?", "seriously?", "sure", "why"]):
                context_modifier = 0.10

        # Total Incongruity Score
        incongruity_score = (self.sentiment_weight * valence_diff) + \
                            (self.pitch_weight * pitch_spike) + \
                            context_modifier
        incongruity_score = round(float(min(1.0, incongruity_score)), 4)

        is_sarcastic = incongruity_score >= self.incongruity_threshold
        sarcasm_confidence = round(float(incongruity_score if is_sarcastic else (1.0 - incongruity_score)), 4)

        # Generate XAI plain-language reasoning string
        reason = (
            f"text sentiment: {text_label} ({text_conf:.2f}) vs "
            f"acoustic tone: {acoustic_label} ({acoustic_conf:.2f}) → "
            f"incongruity score ({incongruity_score:.2f}) {'≥' if is_sarcastic else '<'} "
            f"threshold ({self.incongruity_threshold:.2f}) → "
            f"{'sarcasm likely' if is_sarcastic else 'literal utterance'}"
        )

        return {
            "present": is_sarcastic,
            "confidence": sarcasm_confidence,
            "incongruity_score": incongruity_score,
            "reason": reason
        }
