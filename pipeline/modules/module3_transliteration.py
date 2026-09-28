"""
Module 3 — Transliteration / Normalization (Pre-MT Cleanup)
Converts romanized text (Hinglish/Tanglish) to native scripts (Devanagari, Tamil, etc.)
using AI4Bharat IndicXlit / indic-transliteration package.
Configurable toggle per language.
"""

from typing import Dict, Any, List


class TextNormalizer:
    # Script mappings for indic_transliteration package
    SCRIPT_MAP = {
        "hi": "DEVANAGARI",
        "ta": "TAMIL",
        "te": "TELUGU",
        "kn": "KANNADA",
        "ml": "MALAYALAM",
        "bn": "BENGALI",
        "mr": "DEVANAGARI",
        "gu": "GUJARATI",
        "pa": "GURMUKHI",
        "ur": "URDU"
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("transliteration", {})
        self.enabled = config.get("pipeline", {}).get("enable_transliteration", True)
        self.transliterate_fn = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Initializes indic_transliteration package if installed."""
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
            self.sanscript = sanscript
            self.transliterate_fn = transliterate
        except ImportError:
            print("[Transliteration Warning] 'indic-transliteration' package not found. Using native script mapper.")
            self.transliterate_fn = None

    def normalize(self, transcript: str, language_segments: List[Dict[str, str]]) -> str:
        """
        Normalizes transcript by converting romanized words belonging to Indic languages
        into native script where appropriate.
        """
        if not self.enabled:
            return transcript

        normalized_tokens = []
        for seg in language_segments:
            word = seg.get("text", "")
            lang = seg.get("lang", "en")

            if lang in self.SCRIPT_MAP:
                mapped = self._fallback_indic_map(word, lang)
                if mapped != word:
                    normalized_tokens.append(mapped)
                elif self.transliterate_fn is not None:
                    try:
                        target_script_attr = getattr(self.sanscript, self.SCRIPT_MAP[lang])
                        converted = self.transliterate_fn(word.lower(), self.sanscript.ITRANS, target_script_attr)
                        normalized_tokens.append(converted if converted else word)
                    except Exception:
                        normalized_tokens.append(word)
                else:
                    normalized_tokens.append(word)
            else:
                normalized_tokens.append(word)

        return " ".join(normalized_tokens)

    def _fallback_indic_map(self, word: str, lang: str) -> str:
        """Rule-based transliteration mapping for colloquial romanized terms."""
        w = word.lower().strip()
        
        roman_to_devanagari = {
            "aapka": "आपका", "aapki": "आपकी", "aapke": "आपके", "aap": "आप",
            "mera": "मेरा", "meri": "मेरी", "mere": "मेरे", "main": "मैं", "mein": "में",
            "naam": "नाम", "kaam": "काम", "bahut": "बहुत", "badhiya": "बढ़िया",
            "accha": "अच्छा", "acchi": "अच्छी", "acche": "अच्छे", "hai": "है", "hain": "हैं", "hoon": "हूँ",
            "aur": "और", "ek": "एक", "yeh": "यह", "ye": "यह", "woh": "वह", "wo": "वह",
            "kya": "क्या", "kyun": "क्यों", "kahan": "कहाँ", "kaise": "कैसे", "kab": "कब",
            "kamaal": "कमाल", "kar": "कर", "diya": "दिया", "kiya": "किया", "karna": "करना",
            "toh": "तो", "bhi": "भी", "nahi": "नहीं", "nhi": "नहीं", "mat": "मत",
            "sab": "सब", "kuch": "कुछ", "shukriya": "शुक्रिया", "dhanyavad": "धन्यवाद",
            "namaste": "नमस्ते", "bhai": "भाई", "dost": "दोस्त", "thik": "ठीक",
            "raha": "रहा", "rahi": "रही", "rahe": "रहे", "tha": "था", "thi": "थी", "the": "थे",
            "samajh": "समझ", "aaya": "आया", "gaya": "गया", "dekho": "देखो", "suno": "सुनो",
            "bolo": "बोलो", "batao": "बताओ", "pani": "पानी", "khana": "खाना", "ghar": "घर",
            "bekar": "बेकार", "ganda": "गंदा", "sach": "सच", "jhooth": "झूठ"
        }

        roman_to_tamil = {
            "romba": "ரொம்ப", "nalla": "நல்ல", "velai": "வேலை", "idhu": "இது",
            "adhu": "அது", "enna": "என்ன", "yaaru": "யாரு", "eppadi": "எப்படி",
            "seri": "சரி", "nandri": "நன்றி", "vanakkam": "வணக்கம்", "varum": "வரும்",
            "irukku": "இருக்கு", "illai": "இல்லை", "enga": "எங்க", "unga": "உங்க",
            "thaan": "தான்", "pannunga": "பண்ணுங்க"
        }

        roman_to_telugu = {
            "chala": "చాలా", "bagundi": "బాగుంది", "pani": "పని", "meeru": "మీరు",
            "mee": "మీ", "enti": "ఏంటి", "evaru": "ఎవరు", "ela": "ఎలా",
            "undhi": "ఉంది", "dhanyavadalu": "ధన్యవాదాలు", "namaskaram": "నమస్కారం",
            "sare": "సరే", "avunu": "అవును", "kadhu": "కాదు", "cheyandi": "చేయండి"
        }

        if lang == "hi" or lang not in ["ta", "te"]:
            return roman_to_devanagari.get(w, word)
        elif lang == "ta":
            return roman_to_tamil.get(w, word)
        elif lang == "te":
            return roman_to_telugu.get(w, word)

        return word
