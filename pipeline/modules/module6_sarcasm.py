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
TEXT-FIRST DESIGN: Works reliably for text-only input (no audio needed).
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

    # Positive praise / hyperbole phrases in English, Hinglish, and Tanglish
    POSITIVE_PRAISE_TOKENS = [
        "great job", "good job", "wonderful", "lovely", "brilliant", "genius", "awesome", "perfect",
        "super", "kamaal", "badhiya", "bahut badhiya", "bohot achha", "bohot accha", "bohot badhiya",
        "shabash", "wah", "waah", "kya baat hai", "aaha", "nalla velai", "romba nalla", "semma",
        "mass", "thanks a lot", "thank you so much", "love waiting", "congratulations", "proud of you",
        "lovely surprise", "acting da", "dimag lagaya", "shandar", "arreeee", "bhai kya dimag",
        "super presentation", "super doctor", "brilliant idea", "adbhut", "chala bagundi",
        "what a lovely", "what a brilliant", "what a great", "fantastic", "amazing", "so helpful",
        "love it", "loving it", "best ever", "outstanding", "incredible", "mind blowing",
        "great work", "excellent work", "nicely done", "well done", "such a great",
        "oru number", "vera level", "semma da", "super pa", "nalla pannitinga", "wow great",
        "wow brilliant", "wow amazing", "wow fantastic"
    ]

    # Negative failure, crash, disaster, or delay phrases
    NEGATIVE_SITUATION_TOKENS = [
        "break", "broke", "broken", "breaking", "crash", "crashed", "crashing", "delete", "deleted",
        "deleting", "fail", "failed", "failing", "delayed", "delay", "fire", "fired", "ruin", "ruined",
        "destroy", "destroyed", "waste", "wasted", "down", "bug", "error", "worst", "terrible", "late",
        "traffic", "slow", "freeze", "froze", "boring", "kevalam", "kevalama", "tholanjudhu", "poyiduchu",
        "thoongitanga", "bekar", "ganda", "taalab", "yamraj", "loss", "useless", "stupid", "idiot",
        "3 saal baad", "delayed by", "operation pannanum", "could have been an email", "waste of time",
        "listening to elevator music", "wrong again", "another problem", "server down", "production broke",
        "pochu", "mudinjuchu", "disaster", "chaos", "messed up", "rubbish", "nonsense", "total mess",
        "blew up", "exploded", "collapsed", "shutdown", "not working", "penne pochu", "varala",
        "poyiduchu", "kedaikkala", "kodukala", "thappu", "velai illama", "uda diya", "uda di",
        "barbaad", "barbad", "khatam", "sothapita", "sothappal", "sothapittan", "tholachuta",
        "naashti", "nashti", "kharab", "bigad diya", "tod diya", "thookitaanga", "fail pannitaanga",
        "two hours aacha", "hours aacha", "neram aacha", "neramaacha", "avlo neram", "ivlo neram",
        "ivlo nerama", "avlo nerama", "yosika dhaan", "yosikka dhaan", "yosika ivlo", "yosika avlo",
        "itna time laga", "itna time", "poora din", "sochne me itna", "itna socha", "took you so long",
        "took two hours", "took hours", "did it take you", "bas itna hi", "idhu thaan un", "idhuve perusu",
        "adhuku dhaan", "adhukku dhaan"
    ]

    # Ironic & sarcastic paralinguistic emojis and markers
    SARCASTIC_MARKERS = [
        "\U0001f644", "\U0001f921", "\U0001f643", "\U0001f60f", "\U0001f44f", "\U0001f480", "\U0001f612", "\U0001f926", "\U0001f925",
        "as if", "yeah right", "sure thing", "oh really", "obviously", "of course",
        "totally fine", "no problem at all", "absolutely brilliant", "just great",
        "oh sure", "right sure", "yes sure", "oh wow", "nice try", "good luck with that",
        "congratulations genius", "wonderful timing", "perfect timing", "how convenient"
    ]

    # Pure sarcastic phrases (standalone sarcasm without needing contrast)
    STANDALONE_SARCASTIC_PHRASES = [
        "yeah right", "oh sure", "as if", "nice try", "brilliant as always",
        "wow thanks for nothing", "oh great another", "just what i needed",
        "great timing", "perfect timing", "how convenient", "wonderful thanks",
        "thanks for ruining", "really helpful", "so helpful oh", "oh how helpful",
        "congratulations you broke", "you really outdid yourself",
        "clap clap", "slow clap", "what a surprise", "color me shocked",
        "never saw that coming", "what could go wrong", "totally unexpected",
        "great job breaking", "great job deleting", "great job crashing",
        "nice job breaking", "nice job deleting",
        "bahut badhiya kaam kiya", "wah kya kaam kiya", "kamaal kar diya yaar",
        "shabash delete kar diya", "kya dimag lagaya yaar", "wah bhai kya", "kya kamaal ka kaam",
        "nalla velai pannitinga pochu", "semma velai pannitinga tholanjudhu"
    ]

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("sarcasm", {})
        self.incongruity_threshold = self.config.get("incongruity_threshold", 0.40)
        self.sentiment_weight = self.config.get("weights", {}).get("sentiment_mismatch", 0.70)
        self.pitch_weight = self.config.get("weights", {}).get("acoustic_pitch_spike", 0.30)

    @classmethod
    def _detect_lexical_contrast(cls, text: str) -> tuple:
        """
        Detects intra-textual semantic paradox: positive praise contrasted with negative failure/situation.
        Also detects standalone sarcastic phrases and ironic markers.
        Returns (has_contrast: bool, contrast_score: float, detected_pos: str, detected_neg: str)
        """
        if not text:
            return False, 0.0, "", ""

        import re
        low = text.lower()
        matched_pos = ""
        matched_neg = ""

        # Check standalone sarcastic phrases first (highest confidence)
        for phrase in cls.STANDALONE_SARCASTIC_PHRASES:
            if phrase in low:
                return True, 0.92, phrase, "standalone sarcastic phrase"

        for p in cls.POSITIVE_PRAISE_TOKENS:
            if p in low:
                matched_pos = p
                break

        for n in cls.NEGATIVE_SITUATION_TOKENS:
            if n in low:
                matched_neg = n
                break

        # Check rhetorical mocking question: positive praise followed by mock effort/delay question
        if matched_pos and not matched_neg:
            mock_time_match = re.search(r"\b(?:two|three|four|2|3|4|ivlo|avlo|itna|poora|whole|full)\s*(?:hours|hrs|mins|minutes|neram|time|din|day|days)\b.*(?:\?|aacha|laga|eduthucha|took)", low)
            if mock_time_match:
                matched_neg = mock_time_match.group(0).strip()
            else:
                mock_think_match = re.search(r"\b(?:yosika|yosikka|sochne|socha|think)\b.*(?:\?|aacha|neram|time|hours)", low)
                if mock_think_match:
                    matched_neg = mock_think_match.group(0).strip()

        has_marker = any(m in text for m in cls.SARCASTIC_MARKERS) or any(m in low for m in cls.SARCASTIC_MARKERS)

        if matched_pos and matched_neg:
            score = 0.92 if has_marker else 0.88
            return True, score, matched_pos, matched_neg
        elif matched_pos and has_marker:
            return True, 0.80, matched_pos, "ironic emoji/marker"
        elif matched_neg and ("love" in low or "great" in low or "thanks" in low or "wonderful" in low):
            return True, 0.78, "ironic appraisal", matched_neg
        elif has_marker and (matched_pos or matched_neg):
            token = matched_pos or matched_neg
            return True, 0.72, token, "sarcastic marker"

        return False, 0.0, "", ""

    def detect(
        self,
        text_emotion: Dict[str, Any],
        acoustic_emotion: Dict[str, Any],
        fusion_emotion: Dict[str, Any],
        acoustic_features: Optional[Dict[str, Any]] = None,
        context: Optional[List[Dict[str, str]]] = None,
        raw_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detects sarcasm via:
        1. Pure text lexical paradox engine (PRIMARY - works without audio)
        2. Standalone sarcastic phrase detection
        3. Text-acoustic modality incongruity (SECONDARY - only when acoustic is non-neutral)
        4. Sarcastic discourse markers and emojis
        """
        # STEP 1: Text-Only Lexical Analysis (PRIMARY SIGNAL)
        target_eval_text = raw_text or text_emotion.get("raw_text") or ""
        has_contrast, contrast_score, pos_match, neg_match = self._detect_lexical_contrast(target_eval_text)

        # Neural Code-Mixed Contrastive Incongruity from MuRIL MTL Head
        neural_sarcasm = float(text_emotion.get("sarcasm_score", 0.0))

        # STEP 2: Acoustic/Multimodal Mismatch (SECONDARY - only meaningful with real audio)
        text_label = text_emotion.get("label", "neutral")
        text_conf = text_emotion.get("confidence", 0.5)

        acoustic_label = acoustic_emotion.get("label", "neutral")
        acoustic_conf = acoustic_emotion.get("confidence", 0.5)

        v_text = self.EMOTION_VALENCE.get(text_label, 0.0) * text_conf
        v_acous = self.EMOTION_VALENCE.get(acoustic_label, 0.0) * acoustic_conf

        # Only use multimodal signal if acoustic is genuinely non-neutral (real audio input).
        # A neutral acoustic tone (synthetic/default) is NOT evidence against or for sarcasm.
        acoustic_is_real = acoustic_label != "neutral" or acoustic_conf < 0.5
        polarity_inversion = (
            acoustic_is_real and
            ((v_text > 0.25 and v_acous < -0.25) or (v_text < -0.25 and v_acous > 0.25))
        )

        valence_diff = abs(v_text - v_acous) / 2.0 if polarity_inversion else 0.0

        # Acoustic Pitch Variance Bonus
        pitch_spike = 0.0
        if acoustic_features and polarity_inversion:
            pitch_var = acoustic_features.get("pitch_variance", 0.0)
            if pitch_var > 0.04:
                pitch_spike = min(1.0, pitch_var * 10.0)

        # Contextual Adjustment
        context_modifier = 0.0
        if context and len(context) > 0 and polarity_inversion:
            last_turn = context[-1].get("text", "").lower()
            if any(w in last_turn for w in ["really?", "seriously?", "sure", "why"]):
                context_modifier = 0.10

        base_incongruity = (
            (self.sentiment_weight * valence_diff) +
            (self.pitch_weight * pitch_spike) +
            context_modifier
        )

        # STEP 3: Combine All Signals
        scores = [base_incongruity]
        if has_contrast:
            scores.append(contrast_score)
        if neural_sarcasm > 0.0:
            scores.append(neural_sarcasm)

        incongruity_score = round(float(min(1.0, max(scores))), 4)
        is_sarcastic = incongruity_score >= self.incongruity_threshold
        sarcasm_confidence = round(float(incongruity_score if is_sarcastic else (1.0 - incongruity_score)), 4)

        # STEP 4: Generate XAI plain-language reasoning
        if has_contrast and is_sarcastic:
            if neg_match == "standalone sarcastic phrase":
                reason = (
                    f"standalone sarcastic expression detected: '{pos_match}' \u2192 "
                    f"sarcasm score ({incongruity_score:.2f}) \u2265 threshold ({self.incongruity_threshold:.2f}) \u2192 sarcasm confirmed"
                )
            else:
                reason = (
                    f"semantic incongruity detected: positive praise ('{pos_match}') paired with "
                    f"negative condition ('{neg_match}') \u2192 contrast score ({incongruity_score:.2f}) "
                    f"\u2265 threshold ({self.incongruity_threshold:.2f}) \u2192 sarcasm confirmed"
                )
        elif is_sarcastic:
            reason = (
                f"text sentiment: {text_label} ({text_conf:.2f}) vs "
                f"acoustic tone: {acoustic_label} ({acoustic_conf:.2f}) \u2192 "
                f"incongruity score ({incongruity_score:.2f}) \u2265 "
                f"threshold ({self.incongruity_threshold:.2f}) \u2192 sarcasm likely"
            )
        else:
            reason = (
                f"text sentiment ({text_label}) and situational context are aligned \u2192 "
                f"incongruity score ({incongruity_score:.2f}) < "
                f"threshold ({self.incongruity_threshold:.2f}) \u2192 literal utterance"
            )

        return {
            "present": is_sarcastic,
            "confidence": sarcasm_confidence,
            "incongruity_score": incongruity_score,
            "reason": reason
        }
