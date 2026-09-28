"""
Module 2 — Language Identification (LID)
Identifies languages present at word/segment level in code-mixed Indian utterances.
Outputs:
  - language_segments: List of dicts, e.g. [{"text": "great", "lang": "en"}, {"text": "kaam", "lang": "hi"}]
  - detected_languages: List of unique language codes present
"""

import re
from typing import Dict, Any, List


class LanguageIdentifier:
    # Common English words in Indian code-mixed speech
    EN_LEXICON = {
        "is", "are", "really", "very", "good", "great", "job", "work", "project", 
        "nice", "super", "awesome", "today", "tomorrow", "boss", "office", "call", 
        "thank", "you", "thanks", "hello", "hi", "sir", "madam", "the", "and", "or",
        "but", "so", "for", "with", "this", "that", "was", "will", "have", "has"
    }

    # Common Hindi romanized words
    HI_ROMAN_LEXICON = {
        "aapka", "aap", "tum", "mera", "hum", "kaam", "bahut", "badhiya", "hai", 
        "kya", "kyun", "kahan", "kaise", "thik", "shukriya", "bhai", "dost", 
        "karna", "ho", "gaya", "nhi", "nahi", "accha", "bhi"
    }

    # Common Tamil romanized words
    TA_ROMAN_LEXICON = {
        "nalla", "velai", "romba", "irukku", "idhu", "adhu", "enna", "yaaru",
        "pannunga", "vanakkam", "nandri", "seri", "poya", "varum"
    }

    # Common Telugu romanized words
    TE_ROMAN_LEXICON = {
        "bagundi", "pani", "chala", "undhi", "enti", "evaru", "cheyandi", 
        "namaskaram", "dhanyavadalu", "sare", "vellu"
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("lid", {})
        self.backend = self.config.get("backend", "fasttext_xlmr")
        self.classifier = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Initializes HF XLM-R Language Detection pipeline or lexicon engine."""
        if self.backend == "heuristic":
            return

        try:
            from transformers import pipeline
            model_name = self.config.get("model_name", "papluca/xlm-roberta-base-language-detection")
            self.classifier = pipeline("text-classification", model=model_name)
        except Exception as e:
            print(f"[LID Warning] Could not load XLM-R classifier: {e}. Utilizing Indic code-mix token LID engine.")
            self.classifier = None

    def _detect_unicode_script(self, token: str) -> str:
        """Determines language code based on Unicode character blocks."""
        for char in token:
            code = ord(char)
            if 0x0900 <= code <= 0x097F:
                return "hi"  # Devanagari (Hindi/Marathi)
            elif 0x0B80 <= code <= 0x0BFF:
                return "ta"  # Tamil
            elif 0x0C00 <= code <= 0x0C7F:
                return "te"  # Telugu
            elif 0x0C80 <= code <= 0x0CFF:
                return "kn"  # Kannada
            elif 0x0D00 <= code <= 0x0D7F:
                return "ml"  # Malayalam
            elif 0x0980 <= code <= 0x09FF:
                return "bn"  # Bengali
            elif 0x0A80 <= code <= 0x0AFF:
                return "gu"  # Gujarati
            elif 0x0A00 <= code <= 0x0A7F:
                return "pa"  # Punjabi
            elif 0x0600 <= code <= 0x06FF:
                return "ur"  # Urdu
        return None

    def identify(self, transcript: str) -> Dict[str, Any]:
        """
        Classifies each token / segment into language tags.
        Returns:
          - language_segments: [{"text": str, "lang": str}]
          - detected_languages: list of unique language codes
        """
        words = re.findall(r'\b\w+\b|[^\w\s]', transcript)
        language_segments = []

        for word in words:
            clean_word = word.strip().lower()
            if not clean_word or not re.match(r'\w+', clean_word):
                continue

            # 1. Unicode script check
            script_lang = self._detect_unicode_script(clean_word)
            if script_lang:
                language_segments.append({"text": word, "lang": script_lang})
                continue

            # 2. Lexicon / Heuristic check for Romanized text
            if clean_word in self.EN_LEXICON:
                lang = "en"
            elif clean_word in self.HI_ROMAN_LEXICON:
                lang = "hi"
            elif clean_word in self.TA_ROMAN_LEXICON:
                lang = "ta"
            elif clean_word in self.TE_ROMAN_LEXICON:
                lang = "te"
            else:
                # 3. Model inference if available
                if self.classifier:
                    try:
                        res = self.classifier(clean_word)[0]
                        lang = res.get("label", "en").lower()[:2]
                    except Exception:
                        lang = "hi" if any(clean_word.endswith(x) for x in ["ka", "ki", "ke", "hai", "hu"]) else "en"
                else:
                    lang = "hi" if any(clean_word.endswith(x) for x in ["ka", "ki", "ke", "hai", "hu", "ya"]) else "en"

            language_segments.append({"text": word, "lang": lang})

        detected_langs = sorted(list(set(seg["lang"] for seg in language_segments)))
        if not detected_langs:
            detected_langs = ["hi", "en"]

        return {
            "language_segments": language_segments,
            "detected_languages": detected_langs,
            "is_code_mixed": len(detected_langs) > 1
        }
