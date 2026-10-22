"""
Module 2 — Language Identification (LID)
Identifies languages present at word/segment level in code-mixed Indian utterances.
Supports user-specified source language context (e.g. Tanglish, Hinglish, Telglish)
so tokenization and language classification prioritize the user's selected language.
Outputs:
  - language_segments: List of dicts, e.g. [{"text": "great", "lang": "en"}, {"text": "kaam", "lang": "hi"}]
  - detected_languages: List of unique language codes present
"""

import re
from typing import Dict, Any, List, Optional


class LanguageIdentifier:
    # Common English words in Indian code-mixed speech
    EN_LEXICON = {
        "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being",
        "really", "very", "good", "great", "job", "work", "project", "presentation",
        "nice", "super", "awesome", "today", "tomorrow", "yesterday", "boss", "office", "call", 
        "friend", "friends", "brother", "bro", "sister",
        "thank", "you", "thanks", "hello", "hi", "hey", "sir", "madam", "and", "or",
        "but", "so", "for", "with", "this", "that", "these", "those", "will", "have", "has", "had",
        "please", "party", "homework", "just", "my", "your", "can", "could", "do", "did", "done",
        "going", "come", "came", "see", "saw", "know", "lot", "much", "more", "now", "then",
        "presentation", "report", "file", "computer", "phone", "message", "meeting", "class",
        "problem", "solution", "idea", "ok", "okay", "alright", "fine", "yes", "no", "not"
    }

    # Common Hindi romanized words
    HI_ROMAN_LEXICON = {
        "aapka", "aapki", "aapke", "aap", "tum", "mera", "meri", "mere", "hum", "kaam", 
        "bahut", "badhiya", "hai", "hain", "hoon", "kya", "kyun", "kahan", "kaise", "thik", 
        "shukriya", "bhai", "dost", "karna", "karo", "kar", "ho", "gaya", "gayi", "gaye",
        "nhi", "nahi", "accha", "acchi", "acche", "bhi", "mat", "lo", "tension", "pakka",
        "kal", "tak", "aaj", "ab", "yeh", "ye", "woh", "wo", "sab", "kuch", "bolo", "batao",
        "samajh", "dekho", "suno", "bilkul", "bekar", "kamaal", "aur", "toh", "diya", "kiya"
    }

    # Common Tamil romanized words
    TA_ROMAN_LEXICON = {
        "nan", "naan", "iinaiku", "innaiku", "poren", "nalaiku", "varapa", "nee", "yen", "en",
        "mattum", "pandriya", "pannu", "pannunga", "nalla", "velai", "romba", "irukku", "illa",
        "illai", "idhu", "adhu", "edhu", "enna", "yaaru", "eppadi", "vanakkam", "nandri", "seri",
        "poya", "varum", "varren", "varen", "vandha", "solra", "theriyum", "puriyala", "machan",
        "da", "di", "pa", "bro", "unga", "ungala", "enoda", "unnoda", "avanga", "avan", "aval",
        "inga", "enga", "anga", "appo", "ippo", "eppo", "thaan", "than", "ah", "la", "kooda",
        "solunga", "kudunga", "vaanga", "ponga", "pesu", "paar", "paaru", "paathen"
    }

    # Common Telugu romanized words
    TE_ROMAN_LEXICON = {
        "bagundi", "pani", "chala", "undhi", "enti", "evaru", "cheyandi", "chesava",
        "namaskaram", "dhanyavadalu", "sare", "vellu", "ledu", "kadu", "andi", "meeru", "naa",
        "mee", "eppudu", "ekada", "ela", "avunu", "kadhu", "chesaru", "chestara", "ivvandi"
    }

    # Common Kannada romanized words
    KN_ROMAN_LEXICON = {
        "chennagide", "hegiddira", "beku", "beda", "yenu", "neevu", "nanna", "kelasa",
        "dhanyavadagalu", "namaskara", "illi", "alli", "yelli", "houdu", "illa"
    }

    # Common Malayalam romanized words
    ML_ROMAN_LEXICON = {
        "sheriyanu", "nanni", "pinne", "ivide", "avide", "entha", "njan", "ningal",
        "cheyyam", "und", "illa", "engane", "sundaram"
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("lid", {})
        self.backend = self.config.get("backend", "fasttext_xlmr")
        self.supported_langs = set(self.config.get("supported_languages", ["hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "pa", "ur", "en"]))
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
        except Exception:
            self.classifier = None

    def _detect_unicode_script(self, token: str) -> Optional[str]:
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

    def _parse_source_hint(self, source_lang: Optional[str]) -> tuple:
        """
        Parses source_lang into (base_indic_lang, is_code_mixed_expected).
        E.g.: 'ta-en' -> ('ta', True), 'ta' -> ('ta', False), 'hi-en' -> ('hi', True)
        """
        if not source_lang:
            return (None, False)
        
        sl = source_lang.strip().lower()
        if sl in ["ta-en", "tanglish", "ta_en"]:
            return ("ta", True)
        elif sl in ["hi-en", "hinglish", "hi_en"]:
            return ("hi", True)
        elif sl in ["te-en", "telglish", "teluglish", "te_en"]:
            return ("te", True)
        elif sl in ["kn-en", "kanglish", "kn_en"]:
            return ("kn", True)
        elif sl in ["ml-en", "manglish", "ml_en"]:
            return ("ml", True)
        elif sl in ["bn-en", "bengali-en"]:
            return ("bn", True)
        elif sl in ["ta", "tamil"]:
            return ("ta", False)
        elif sl in ["hi", "hindi"]:
            return ("hi", False)
        elif sl in ["te", "telugu"]:
            return ("te", False)
        elif sl in ["kn", "kannada"]:
            return ("kn", False)
        elif sl in ["ml", "malayalam"]:
            return ("ml", False)
        elif sl in ["en", "english"]:
            return ("en", False)
        
        parts = sl.split("-")
        return (parts[0], len(parts) > 1 and "en" in parts)

    def identify(self, transcript: str, source_lang: Optional[str] = None) -> Dict[str, Any]:
        """
        Classifies each token / segment into language tags.
        Takes user-selected source_lang as context (e.g. Tanglish -> ta, Hinglish -> hi).
        Returns:
          - language_segments: [{"text": str, "lang": str}]
          - detected_languages: list of unique language codes
          - is_code_mixed: boolean
        """
        base_indic, expects_cm = self._parse_source_hint(source_lang)
        words = [w.strip() for w in re.split(r'\s+', transcript.strip()) if w.strip()]
        language_segments = []

        for word in words:
            m = re.match(r"^([^\w\u0900-\u0DFF]*)([\w\u0900-\u0DFF'-]+)([^\w\u0900-\u0DFF]*)$", word)
            clean_word = m.group(2).lower() if m else word.strip().lower()
            if not clean_word:
                continue

            # 1. Unicode script check (Highest precedence)
            script_lang = self._detect_unicode_script(clean_word)
            if script_lang:
                language_segments.append({"text": word, "lang": script_lang})
                continue

            # 2. English lexicon check
            is_english = clean_word in self.EN_LEXICON

            # If user explicitly selected pure English
            if base_indic == "en":
                language_segments.append({"text": word, "lang": "en"})
                continue

            # If user selected a specific language (e.g., Tanglish, Hinglish, Telglish)
            if base_indic:
                if is_english:
                    # English word in code-mix
                    lang = "en"
                elif base_indic == "ta":
                    # Tanglish context: Non-English Romanized words belong to Tamil
                    lang = "ta"
                elif base_indic == "te":
                    # Telglish context: Non-English Romanized words belong to Telugu
                    lang = "te"
                elif base_indic == "hi":
                    # Hinglish context: Non-English Romanized words belong to Hindi
                    lang = "hi"
                elif base_indic == "kn":
                    lang = "kn"
                elif base_indic == "ml":
                    lang = "ml"
                else:
                    lang = base_indic
            else:
                # 3. No prior language specified: infer via lexicons
                if is_english:
                    lang = "en"
                elif clean_word in self.TA_ROMAN_LEXICON:
                    lang = "ta"
                elif clean_word in self.TE_ROMAN_LEXICON:
                    lang = "te"
                elif clean_word in self.HI_ROMAN_LEXICON:
                    lang = "hi"
                elif clean_word in self.KN_ROMAN_LEXICON:
                    lang = "kn"
                elif clean_word in self.ML_ROMAN_LEXICON:
                    lang = "ml"
                else:
                    # Model inference or contextual heuristic
                    # Check Tamil suffixes
                    if any(clean_word.endswith(s) for s in ["nga", "iya", "ren", "apa", "kku", "lai", "oda"]):
                        lang = "ta"
                    elif any(clean_word.endswith(s) for s in ["ndi", "undhi", "sava", "edu"]):
                        lang = "te"
                    elif any(clean_word.endswith(s) for s in ["ka", "ki", "ke", "hai", "hu", "ya"]):
                        lang = "hi"
                    elif self.classifier:
                        try:
                            res = self.classifier(clean_word)[0]
                            cand = res.get("label", "en").lower()[:2]
                            lang = cand if cand in self.supported_langs else "en"
                        except Exception:
                            lang = "en"
                    else:
                        lang = "en"

            language_segments.append({"text": word, "lang": lang})

        detected_langs = sorted(list(set(seg["lang"] for seg in language_segments)))
        if not detected_langs:
            detected_langs = [base_indic or "en"]

        return {
            "language_segments": language_segments,
            "detected_languages": detected_langs,
            "is_code_mixed": len(detected_langs) > 1
        }
