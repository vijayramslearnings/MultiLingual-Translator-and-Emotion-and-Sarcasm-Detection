"""
Advanced Preprocessing Pipeline for Indian Code-Mixed Languages.
Includes:
1. Phonetic Normalization & Elongation Squashing (e.g., 'booooodhiyaaaaa' -> 'badhiya')
2. Emoji Extraction & Paralinguistic Dense Anchor Vector Mapping
3. Token-Level Language Identification (LID: EN, IND, UNI)
4. Code-Switch Density & Punctuation Intensity Statistics
5. Auxiliary Script Channel Generation via Indic Transliteration (IndicXlit / Indic-Transliteration)
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch


@dataclass
class PreprocessedOutput:
    """Structured container for preprocessed code-mixed text features."""
    raw_text: str
    normalized_text: str
    tokens: List[str]
    token_lids: List[str]                  # List of 'EN', 'IND', or 'UNI'
    switch_count: int
    switch_density: float                  # Switches normalized by token transitions
    indic_ratio: float
    english_ratio: float
    universal_ratio: float
    punctuation_intensity: Dict[str, float]
    emojis: List[str]
    emoji_dense_vector: np.ndarray         # Paralinguistic anchor dense embedding (16-d)
    auxiliary_feature_vector: np.ndarray   # Concatenated statistical & paralinguistic vector (32-d)
    transliterated_indic_text: str         # Transliterated Brahmic script representation


class CodeMixedPreprocessor:
    """
    Production-grade preprocessing engine tailored specifically for Indian Code-Mixed Languages
    (Hinglish, Tamlish, Telglish, Bengalish, etc.).
    """

    # Valid geminate consonants or standard double vowels in Indian Romanization & English
    VALID_DOUBLES = {
        "aa", "ee", "oo", "ii", "uu", "ai", "au",
        "ch", "kh", "gh", "jh", "th", "dh", "ph", "bh", "sh",
        "tt", "dd", "nn", "ll", "rr", "ss", "pp", "kk", "cc", "mm", "bb"
    }

    # Common Indian Romanization elongation corrections
    ELONGATION_LEXICON = {
        "booooodhiyaaaaa": "badhiya",
        "boodhiya": "badhiya",
        "badhiyaaaaa": "badhiya",
        "badhiyaaa": "badhiya",
        "badhiyaa": "badhiya",
        "achhhhaaaa": "accha",
        "achhaaa": "accha",
        "achha": "accha",
        "acha": "accha",
        "bohot": "bahut",
        "bahutttt": "bahut",
        "plzzzz": "please",
        "pleaaaseee": "please",
        "pleaseeee": "please",
        "soooo": "so",
        "sooooo": "so",
        "rombaaaaa": "romba",
        "nallaaa": "nalla",
        "bagundhiiii": "bagundi",
        "bagundiiii": "bagundi",
        "bekaaaar": "bekar",
        "bekaaar": "bekar",
        "kamaaaal": "kamaal",
        "waaaah": "wah",
        "waah": "wah",
        "arre": "arrey",
        "arreeee": "arrey",
        "bhaaiii": "bhai",
        "bhaiiii": "bhai",
    }

    # Emoji affective lookup dictionary: [valence, arousal, dominance, sarcasm_affinity, anger, joy, sadness]
    # Scales: valence [-1.0, 1.0], arousal [0.0, 1.0], sarcasm_affinity [0.0, 1.0]
    EMOJI_AFFECT_MAP = {
        # Sarcastic / Mocking / Smug
        "😂": [0.6, 0.8, 0.6, 0.85, 0.05, 0.80, 0.02],
        "🤣": [0.7, 0.9, 0.7, 0.88, 0.05, 0.85, 0.02],
        "🙃": [-0.1, 0.5, 0.4, 0.95, 0.15, 0.20, 0.30],  # upside-down face: classic sarcasm
        "😏": [0.1, 0.5, 0.7, 0.92, 0.10, 0.30, 0.05],  # smirking
        "🙄": [-0.5, 0.6, 0.5, 0.90, 0.45, 0.05, 0.20],  # eye roll
        "🤡": [-0.4, 0.6, 0.3, 0.92, 0.30, 0.10, 0.35],  # clown
        "👏": [0.4, 0.7, 0.6, 0.75, 0.05, 0.65, 0.05],  # clapping (often sarcastic praise)
        "💀": [0.1, 0.8, 0.4, 0.80, 0.05, 0.60, 0.20],  # dead from laughter/shock
        "☠️": [-0.2, 0.7, 0.4, 0.75, 0.20, 0.40, 0.30],
        
        # High Positive / Joy / Gratitude
        "❤️": [0.9, 0.7, 0.8, 0.05, 0.01, 0.95, 0.01],
        "😍": [0.95, 0.85, 0.8, 0.08, 0.01, 0.98, 0.01],
        "🥰": [0.92, 0.70, 0.7, 0.05, 0.01, 0.95, 0.01],
        "😊": [0.80, 0.50, 0.6, 0.10, 0.01, 0.85, 0.02],
        "🙏": [0.65, 0.40, 0.5, 0.15, 0.01, 0.60, 0.05],  # Namaste / gratitude
        "🎉": [0.90, 0.90, 0.8, 0.20, 0.02, 0.92, 0.01],
        "🔥": [0.85, 0.85, 0.8, 0.25, 0.05, 0.85, 0.02],

        # Negative / Anger / Disgust
        "😡": [-0.85, 0.90, 0.8, 0.20, 0.95, 0.01, 0.10],
        "🤬": [-0.95, 0.98, 0.9, 0.30, 0.98, 0.01, 0.10],
        "😠": [-0.75, 0.80, 0.7, 0.25, 0.85, 0.02, 0.15],
        "🤮": [-0.90, 0.80, 0.5, 0.40, 0.50, 0.01, 0.60],
        "🤢": [-0.80, 0.65, 0.4, 0.35, 0.40, 0.01, 0.55],
        "💩": [-0.70, 0.60, 0.4, 0.65, 0.30, 0.10, 0.40],

        # Sadness / Crying
        "😭": [-0.50, 0.90, 0.3, 0.45, 0.10, 0.20, 0.90],
        "😢": [-0.70, 0.60, 0.2, 0.20, 0.10, 0.02, 0.85],
        "🥺": [-0.20, 0.55, 0.2, 0.15, 0.05, 0.20, 0.65],
        "💔": [-0.85, 0.75, 0.2, 0.20, 0.20, 0.02, 0.90],
        
        # Neutral / Thoughtful
        "🤔": [0.00, 0.40, 0.5, 0.50, 0.10, 0.10, 0.10],
        "😐": [-0.10, 0.20, 0.4, 0.60, 0.15, 0.05, 0.20],
        "😑": [-0.20, 0.20, 0.4, 0.65, 0.20, 0.05, 0.25]
    }

    # Core lexicons for high-precision, zero-latency local Token-level LID
    INDIC_ROMAN_STOPWORDS = {
        # Hindi / Urdu
        "hai", "hain", "hoon", "tha", "thi", "the", "kya", "kyun", "kaise", "kahan", "kab",
        "aap", "aapka", "aapki", "aapke", "tum", "tera", "teri", "tere", "mera", "meri", "mere",
        "hum", "humara", "yeh", "ye", "woh", "wo", "kuch", "sab", "bahut", "badhiya", "accha",
        "bekar", "bhai", "yaar", "boss", "arrey", "arre", "wah", "kamaal", "kar", "diya", "kiya",
        "nahi", "nhi", "mat", "toh", "bhi", "lekin", "par", "aur", "ek", "do", "karo", "raha",
        "rahi", "rahe", "gaya", "gayi", "gaye", "shukriya", "dhanyavad", "namaste", "pakka",
        # Tamil
        "naan", "nan", "neenga", "nee", "en", "yen", "unga", "avanga", "ivan", "avan",
        "enna", "edhu", "idhu", "adhu", "yaaru", "eppadi", "enga", "inga", "anga", "romba",
        "nalla", "velai", "seri", "machan", "da", "di", "theriyum", "puriyala", "poren",
        "varren", "innaiku", "iinaiku", "nalaiku", "pandriya", "pannunga", "solra", "paru",
        "nandri", "vanakkam", "irukku", "illai", "illa", "mattum", "thaan", "than", "ku",
        # Telugu
        "nenu", "meeru", "nuvvu", "enti", "evaru", "ela", "ekkada", "chala", "bagundi",
        "pani", "kadu", "ledu", "undi", "chesava", "andi", "cheppandi", "dhanyavadalu",
        # Bengali
        "ami", "tumi", "apni", "ki", "keno", "kothay", "khub", "bhalo", "kharap", "koro",
        "korchi", "hobe", "hoyeche", "shundor", "dhonnobad", "thik", "ache", "ekhon"
    }

    ENGLISH_COMMON_WORDS = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
        "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
        "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
        "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
        "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
        "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
        "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
        "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
        "really", "super", "presentation", "report", "project", "boss", "meeting", "code",
        "party", "homework", "please", "thanks", "sir", "office", "call", "email", "submission"
    }

    def __init__(self, target_indic_script: str = "Devanagari"):
        self.target_indic_script = target_indic_script
        self._init_transliteration_backend()

    def _init_transliteration_backend(self):
        """Initializes Indic transliteration engine (IndicXlit or indic-transliteration)."""
        self.transliteration_available = False
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
            self.sanscript = sanscript
            self.transliterate_fn = transliterate
            self.transliteration_available = True
        except ImportError:
            self.sanscript = None
            self.transliterate_fn = None

    # --------------------------------------------------------------------------
    # 1. Phonetic Normalization & Elongation Squashing
    # --------------------------------------------------------------------------
    def squash_phonetic_elongations(self, text: str) -> str:
        """
        Squashes arbitrary character elongations without destroying legitimate double
        letters in code-mixed languages (e.g. 'booooodhiyaaaaa' -> 'badhiya',
        'achhhhaaaa' -> 'accha', 'rombaaaaa' -> 'romba', 'sooooo' -> 'so').
        """
        tokens = text.split()
        normalized_tokens = []

        for token in tokens:
            lower_token = token.lower()
            # 1. Fast path: Direct lexicon match
            if lower_token in self.ELONGATION_LEXICON:
                # Retain capitalization if token was title or uppercase
                rep = self.ELONGATION_LEXICON[lower_token]
                if token.isupper():
                    normalized_tokens.append(rep.upper())
                elif token.istitle():
                    normalized_tokens.append(rep.title())
                else:
                    normalized_tokens.append(rep)
                continue

            # 2. General phonetic regex reduction:
            # Match 3 or more repeating identical characters
            # e.g., 'ooooo' -> check if 'oo' or 'o'
            def _reduce_match(match):
                char = match.group(1)
                full_run = match.group(0)
                # Check if 2 chars is a valid double letter
                two_char = (char + char).lower()
                if two_char in self.VALID_DOUBLES:
                    # In Indic transliteration, 'oo' is often 'u' or 'oo', 'aa' is long a
                    return char + char
                # Otherwise reduce to single char (e.g. 'yyyy' -> 'y', 'hhhh' -> 'h')
                return char

            # Regex replaces 3 or more repeated characters
            cleaned = re.sub(r'([a-zA-Z])\1{2,}', _reduce_match, token)

            # Check if cleaned version matches known lexicon
            cleaned_lower = cleaned.lower()
            if cleaned_lower in self.ELONGATION_LEXICON:
                cleaned = self.ELONGATION_LEXICON[cleaned_lower]
                if token.isupper():
                    cleaned = cleaned.upper()
                elif token.istitle():
                    cleaned = cleaned.title()

            normalized_tokens.append(cleaned)

        return " ".join(normalized_tokens)

    # --------------------------------------------------------------------------
    # 2. Emoji Extraction & Dense Anchor Vector Mapping
    # --------------------------------------------------------------------------
    def extract_and_embed_emojis(self, text: str, vector_dim: int = 16) -> Tuple[List[str], np.ndarray]:
        """
        Extracts emojis from text and projects them into a dense 16-d paralinguistic anchor vector.
        Does NOT strip emojis from original text; emojis represent crucial emotional ground-truth.
        """
        # Regex capturing emoji sequences across Unicode Emoji ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA70-\U0001FAFF"
            "]+",
            flags=re.UNICODE
        )
        
        extracted_emojis = []
        for char in text:
            if emoji_pattern.match(char):
                extracted_emojis.append(char)

        # Dense anchor embedding initialized to zeros (16 dimensions)
        # Dim 0: Mean valence [-1, 1]
        # Dim 1: Mean arousal [0, 1]
        # Dim 2: Mean dominance [0, 1]
        # Dim 3: Sarcasm affinity score [0, 1]
        # Dim 4: Joy/Anger conflict metric (incongruity signal)
        # Dim 5: Anger intensity
        # Dim 6: Joy intensity
        # Dim 7: Sadness intensity
        # Dim 8: Total emoji count (normalized log scale)
        # Dim 9: Distinct emoji count
        # Dim 10-15: Categorical affective flags
        dense_vec = np.zeros(vector_dim, dtype=np.float32)

        if not extracted_emojis:
            return extracted_emojis, dense_vec

        valences = []
        arousals = []
        dominances = []
        sarcasms = []
        angers = []
        joys = []
        sadnesses = []

        for emo in extracted_emojis:
            if emo in self.EMOJI_AFFECT_MAP:
                v, a, d, s, ang, joy, sad = self.EMOJI_AFFECT_MAP[emo]
                valences.append(v)
                arousals.append(a)
                dominances.append(d)
                sarcasms.append(s)
                angers.append(ang)
                joys.append(joy)
                sadnesses.append(sad)
            else:
                # Default unknown emoji baseline
                valences.append(0.0)
                arousals.append(0.5)
                dominances.append(0.5)
                sarcasms.append(0.2)
                angers.append(0.0)
                joys.append(0.3)
                sadnesses.append(0.0)

        n = len(extracted_emojis)
        mean_v = float(np.mean(valences))
        mean_a = float(np.mean(arousals))
        mean_d = float(np.mean(dominances))
        mean_s = float(np.mean(sarcasms))
        mean_ang = float(np.mean(angers))
        mean_joy = float(np.mean(joys))
        mean_sad = float(np.mean(sadnesses))

        # Incongruity signal: coexistence of high joy + high anger/sarcasm
        incongruity_flag = float(min(1.0, (mean_joy * mean_ang * 4.0) + (mean_s * 0.5)))

        dense_vec[0] = mean_v
        dense_vec[1] = mean_a
        dense_vec[2] = mean_d
        dense_vec[3] = mean_s
        dense_vec[4] = incongruity_flag
        dense_vec[5] = mean_ang
        dense_vec[6] = mean_joy
        dense_vec[7] = mean_sad
        dense_vec[8] = float(np.log1p(n))
        dense_vec[9] = float(len(set(extracted_emojis)) / n)
        dense_vec[10] = 1.0 if any(e in ["😂", "🤣", "🙃", "😏", "🙄", "🤡"] for e in extracted_emojis) else 0.0
        dense_vec[11] = 1.0 if any(e in ["😡", "🤬", "😠"] for e in extracted_emojis) else 0.0
        dense_vec[12] = 1.0 if any(e in ["😭", "😢", "💔"] for e in extracted_emojis) else 0.0
        dense_vec[13] = 1.0 if any(e in ["🙏"] for e in extracted_emojis) else 0.0
        dense_vec[14] = 1.0 if any(e in ["❤️", "😍", "🥰"] for e in extracted_emojis) else 0.0
        dense_vec[15] = 1.0 if n >= 3 else 0.0

        return extracted_emojis, dense_vec

    # --------------------------------------------------------------------------
    # 3. Token-Level Language Identification (LID)
    # --------------------------------------------------------------------------
    def identify_token_languages(self, tokens: List[str]) -> List[str]:
        """
        Classifies each token as 'EN' (English), 'IND' (Native Indian Language), or 'UNI' (Universal).
        Lightweight, deterministic, and latency-free local computational linguistic classifier.
        """
        lids = []
        for t in tokens:
            cleaned = re.sub(r'[^\w\s]', '', t).strip().lower()

            if not cleaned or t.isdigit() or any(unicodedata.category(c).startswith('S') for c in t):
                lids.append("UNI")
                continue

            # Check if token is written in Brahmic scripts (Devanagari, Tamil, Telugu, Bengali)
            has_indic_script = any(
                0x0900 <= ord(c) <= 0x0DFF for c in t
            )
            if has_indic_script:
                lids.append("IND")
                continue

            # Check lexicons
            if cleaned in self.INDIC_ROMAN_STOPWORDS:
                lids.append("IND")
            elif cleaned in self.ENGLISH_COMMON_WORDS:
                lids.append("EN")
            else:
                # Morphological heuristics for South Asian Romanized words:
                # Common Dravidian / Indo-Aryan Roman suffixes:
                # -unga, -iya, -iye, -gaya, -ungi, -adha, -kiren, -itu, -ku, -nu, -la, -da, -di
                if re.search(r'(iya|iye|unga|ittu|kiren|aadha|avanga|gaya|gayi|wala|wali|kar|bhi)$', cleaned):
                    lids.append("IND")
                elif len(cleaned) <= 2 and cleaned not in {"in", "on", "at", "to", "is", "it", "my", "me", "we", "he", "no", "so", "do", "go", "if", "as", "by", "up"}:
                    lids.append("IND")
                else:
                    # English default for unseen Latin tokens in standard Latin morphology
                    lids.append("EN")

        return lids

    # --------------------------------------------------------------------------
    # 4. Statistical Switch Metrics & Punctuation Intensity
    # --------------------------------------------------------------------------
    def compute_switch_and_punctuation_stats(
        self, raw_text: str, tokens: List[str], lids: List[str]
    ) -> Tuple[int, float, float, float, float, Dict[str, float]]:
        """
        Computes code-switch frequency, switch density across language boundaries,
        and punctuation intensity (exclamations, questions, caps ratio).
        """
        # Filter linguistic tokens for switch computation
        ling_lids = [lid for lid in lids if lid in ("EN", "IND")]
        switch_count = 0
        for i in range(1, len(ling_lids)):
            if ling_lids[i] != ling_lids[i - 1]:
                switch_count += 1

        total_transitions = max(1, len(ling_lids) - 1)
        switch_density = switch_count / total_transitions if ling_lids else 0.0

        n_tokens = max(1, len(tokens))
        indic_ratio = sum(1 for lid in lids if lid == "IND") / n_tokens
        english_ratio = sum(1 for lid in lids if lid == "EN") / n_tokens
        universal_ratio = sum(1 for lid in lids if lid == "UNI") / n_tokens

        # Punctuation & orthographic intensity
        exclamation_count = raw_text.count("!")
        question_count = raw_text.count("?")
        ellipsis_count = raw_text.count("...") + raw_text.count("…")
        
        letters = [c for c in raw_text if c.isalpha()]
        caps_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
        uppercase_words = sum(1 for t in tokens if t.isupper() and len(t) > 1)

        punct_stats = {
            "exclamation_count": float(exclamation_count),
            "question_count": float(question_count),
            "ellipsis_count": float(ellipsis_count),
            "caps_ratio": float(round(caps_ratio, 4)),
            "uppercase_words": float(uppercase_words),
            "punct_intensity_score": float(min(1.0, (exclamation_count * 0.25) + (question_count * 0.20) + (caps_ratio * 0.5)))
        }

        return switch_count, switch_density, indic_ratio, english_ratio, universal_ratio, punct_stats

    # --------------------------------------------------------------------------
    # 5. Transliteration Auxiliary Channel (IndicXlit / Indic-Transliteration)
    # --------------------------------------------------------------------------
    def transliterate_to_indic_channel(self, text: str, target_script: str = "Devanagari") -> str:
        """
        Converts Romanized code-mixed terms into native Brahmic scripts as an auxiliary feature channel.
        """
        if not self.transliteration_available or not text.strip():
            return text

        try:
            # Map script names to sanscript constants
            script_map = {
                "Devanagari": self.sanscript.DEVANAGARI,
                "Tamil": self.sanscript.TAMIL,
                "Telugu": self.sanscript.TELUGU,
                "Bengali": self.sanscript.BENGALI,
                "Kannada": self.sanscript.KANNADA,
                "Malayalam": self.sanscript.MALAYALAM
            }
            target = script_map.get(target_script, self.sanscript.DEVANAGARI)
            
            # Using ITRANS/HK scheme for transliterating Roman to native script
            return self.transliterate_fn(text.lower(), self.sanscript.ITRANS, target)
        except Exception:
            return text

    # --------------------------------------------------------------------------
    # Master Preprocess Method
    # --------------------------------------------------------------------------
    def preprocess(self, text: str) -> PreprocessedOutput:
        """
        Runs complete pre-processing workflow on raw code-mixed text.
        Produces cleaned text, paralinguistic vectors, token LIDs, switch stats,
        and concatenated auxiliary feature vector (32-d).
        """
        if not text or not text.strip():
            text = "neutral"

        # 1. Phonetic elongation squashing
        normalized = self.squash_phonetic_elongations(text)

        # 2. Extract and embed emojis (do not strip!)
        emojis, emoji_vector = self.extract_and_embed_emojis(normalized, vector_dim=16)

        # 3. Token-level LID
        # Tokenize by whitespace while preserving punctuation
        tokens = re.findall(r'\S+', normalized)
        token_lids = self.identify_token_languages(tokens)

        # 4. Statistical switch and punctuation stats
        (
            switch_count,
            switch_density,
            indic_ratio,
            english_ratio,
            universal_ratio,
            punct_stats
        ) = self.compute_switch_and_punctuation_stats(normalized, tokens, token_lids)

        # 5. Transliterated Indic channel
        indic_channel = self.transliterate_to_indic_channel(normalized, self.target_indic_script)

        # 6. Assemble concatenated 32-dimensional auxiliary feature vector:
        # [0:16]  = emoji dense vector (16-d)
        # [16:22] = code-mix & switch statistics (6-d)
        # [22:28] = punctuation & orthography intensity (6-d)
        # [28:32] = reserved contextual / contrastive prior flags (4-d)
        aux_vector = np.zeros(32, dtype=np.float32)
        aux_vector[0:16] = emoji_vector
        aux_vector[16] = float(switch_count)
        aux_vector[17] = float(switch_density)
        aux_vector[18] = float(indic_ratio)
        aux_vector[19] = float(english_ratio)
        aux_vector[20] = float(universal_ratio)
        aux_vector[21] = float(len(tokens))

        aux_vector[22] = punct_stats["exclamation_count"]
        aux_vector[23] = punct_stats["question_count"]
        aux_vector[24] = punct_stats["ellipsis_count"]
        aux_vector[25] = punct_stats["caps_ratio"]
        aux_vector[26] = punct_stats["uppercase_words"]
        aux_vector[27] = punct_stats["punct_intensity_score"]

        # Prior context incongruity flag: positive emoji + high punctuation intensity
        aux_vector[28] = float(emoji_vector[10] * punct_stats["punct_intensity_score"])
        # High switch density indicator
        aux_vector[29] = 1.0 if switch_density > 0.40 else 0.0
        # Code-mixed balance indicator (both languages present)
        aux_vector[30] = 1.0 if (indic_ratio > 0.2 and english_ratio > 0.2) else 0.0
        aux_vector[31] = float(len(emojis))

        return PreprocessedOutput(
            raw_text=text,
            normalized_text=normalized,
            tokens=tokens,
            token_lids=token_lids,
            switch_count=switch_count,
            switch_density=switch_density,
            indic_ratio=indic_ratio,
            english_ratio=english_ratio,
            universal_ratio=universal_ratio,
            punctuation_intensity=punct_stats,
            emojis=emojis,
            emoji_dense_vector=emoji_vector,
            auxiliary_feature_vector=aux_vector,
            transliterated_indic_text=indic_channel
        )
