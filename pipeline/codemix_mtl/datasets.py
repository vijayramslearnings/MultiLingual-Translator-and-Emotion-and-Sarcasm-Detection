"""
Dataset Pipelines & Benchmarks for Indian Code-Mixed Languages.
Includes PyTorch Dataset implementations for:
1. HASOC Dataset (Indo-Aryan & Dravidian Hate Speech & Nuanced Social Text)
2. SemEval Sentimix Dataset (Complex Explicit Text Switches & Sentiment)
3. DravidianLangTech Dataset (Tamil-English, Malayalam-English Nuances)
Includes CodeMixedDataCollator and Class Imbalance Weight Calculators.
"""

import os
import csv
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessor import CodeMixedPreprocessor, PreprocessedOutput


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Computes inverse frequency class weights to cleanly mitigate severe class imbalances:
    weight[c] = total_samples / (num_classes * count[c])
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 1.0
    # Normalize so mean weight is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


class BaseCodeMixedDataset(Dataset):
    """Base class handling tokenization and preprocessing for code-mixed datasets."""
    
    EMOTION_MAP = {
        "positive": 0, "pos": 0, "joy": 4, "happy": 4, "happiness": 4,
        "negative": 1, "neg": 1, "anger": 3, "angry": 3, "disgust": 3,
        "neutral": 2, "neu": 2, "mixed": 2, "mixed_feelings": 2,
        "sadness": 5, "sad": 5
    }

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: Optional[Any] = None,
        preprocessor: Optional[CodeMixedPreprocessor] = None,
        max_length: int = 128
    ):
        self.samples = samples
        self.tokenizer = tokenizer or self._init_default_tokenizer()
        self.preprocessor = preprocessor or CodeMixedPreprocessor()
        self.max_length = max_length

    def _init_default_tokenizer(self):
        import os
        if os.environ.get("INDIC_FAST_MODE") == "1":
            return None
        try:
            from transformers import AutoTokenizer
            try:
                return AutoTokenizer.from_pretrained("google/muril-base-cased", local_files_only=True)
            except Exception:
                return AutoTokenizer.from_pretrained("google/muril-base-cased", local_files_only=False)
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes text using MuRIL tokenizer or fallback token indexer."""
        if self.tokenizer is not None:
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        else:
            # Deterministic ASCII/hash token indexer fallback
            tokens = text.split()[:self.max_length]
            ids = [101] + [min(190000, abs(hash(t)) % 190000 + 1) for t in tokens] + [102]
            mask = [1] * len(ids)
            if len(ids) < self.max_length:
                pad_len = self.max_length - len(ids)
                ids += [0] * pad_len
                mask += [0] * pad_len
            else:
                ids = ids[:self.max_length]
                mask = mask[:self.max_length]
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        raw_text = str(item.get("text", ""))

        # 1. Advanced Preprocessing
        prep: PreprocessedOutput = self.preprocessor.preprocess(raw_text)

        # 2. Tokenize normalized string
        input_ids, attention_mask = self._tokenize(prep.normalized_text)

        # 3. Auxiliary feature tensor (32-d)
        aux_features = torch.tensor(prep.auxiliary_feature_vector, dtype=torch.float32)

        # 4. Target Labels
        emotion_label = item.get("emotion_label", 2) # default Neutral (2)
        sarcasm_label = item.get("sarcasm_label", 0) # default Literal (0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "aux_features": aux_features,
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
            "sarcasm_label": torch.tensor(sarcasm_label, dtype=torch.long),
            "raw_text": raw_text,
            "normalized_text": prep.normalized_text
        }


# ==============================================================================
# 1. HASOC Dataset (Hate Speech and Offensive Content in Code-Mixed Text)
# ==============================================================================
class HASOCDataset(BaseCodeMixedDataset):
    """
    PyTorch Dataset for HASOC benchmark (Indo-Aryan & Dravidian code-mixed social posts).
    Learns noisy, offensive, aggressive, and sarcastic social phenomena.
    """
    def __init__(
        self,
        file_path_or_samples: Union[str, List[Dict[str, Any]]],
        tokenizer: Optional[Any] = None,
        preprocessor: Optional[CodeMixedPreprocessor] = None,
        max_length: int = 128
    ):
        if isinstance(file_path_or_samples, str):
            samples = self._load_from_file(file_path_or_samples)
        else:
            samples = file_path_or_samples
        super().__init__(samples, tokenizer, preprocessor, max_length)

    def _load_from_file(self, path: str) -> List[Dict[str, Any]]:
        samples = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"HASOC file not found: {path}")

        delimiter = "\t" if path.endswith(".tsv") else ","
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                text = row.get("text") or row.get("tweet") or row.get("comment", "")
                task_1 = (row.get("task_1") or row.get("label", "NOT")).upper()
                task_2 = (row.get("task_2") or "NONE").upper()

                # Map HASOC labels to Emotion & Sarcasm tasks
                # HOF (Hate/Offensive) often exhibits Anger/Disgust (3) or mocking Sarcasm (1)
                is_hof = (task_1 == "HOF")
                is_sarcastic = 1 if (is_hof and any(w in text.lower() for w in ["wah", "great", "shabaash", "clown", "boss", "😂", "👏"])) else 0
                emotion_label = 3 if is_hof else 2  # 3: Anger, 2: Neutral

                samples.append({
                    "text": text,
                    "emotion_label": emotion_label,
                    "sarcasm_label": is_sarcastic,
                    "dataset_source": "HASOC",
                    "original_label": task_1
                })
        return samples


# ==============================================================================
# 2. SemEval Sentimix Dataset (Explicit Language Switch Sentiment & Sarcasm)
# ==============================================================================
class SemEvalSentimixDataset(BaseCodeMixedDataset):
    """
    PyTorch Dataset for SemEval-2020 Task 9 (Sentimix Hinglish / Bengali-English).
    Handles complex explicit text switches, token-level language boundaries, and sentiment.
    """
    def __init__(
        self,
        file_path_or_samples: Union[str, List[Dict[str, Any]]],
        tokenizer: Optional[Any] = None,
        preprocessor: Optional[CodeMixedPreprocessor] = None,
        max_length: int = 128
    ):
        if isinstance(file_path_or_samples, str):
            samples = self._load_from_file(file_path_or_samples)
        else:
            samples = file_path_or_samples
        super().__init__(samples, tokenizer, preprocessor, max_length)

    def _load_from_file(self, path: str) -> List[Dict[str, Any]]:
        samples = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"SemEval Sentimix file not found: {path}")

        # Parses either TSV / JSON or ConLL token-level switch formats
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    s_label = self.EMOTION_MAP.get(str(item.get("sentiment", "neutral")).lower(), 2)
                    samples.append({
                        "text": item.get("text", ""),
                        "emotion_label": s_label,
                        "sarcasm_label": int(item.get("sarcasm", 0)),
                        "dataset_source": "SemEval-Sentimix"
                    })
        else:
            delimiter = "\t" if path.endswith(".tsv") else ","
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    text = row.get("text") or row.get("sentence", "")
                    sent = (row.get("sentiment") or "neutral").lower()
                    emotion_label = self.EMOTION_MAP.get(sent, 2)
                    is_sarc = int(row.get("sarcasm", 1 if ("😂" in text and emotion_label == 1) else 0))
                    samples.append({
                        "text": text,
                        "emotion_label": emotion_label,
                        "sarcasm_label": is_sarc,
                        "dataset_source": "SemEval-Sentimix"
                    })
        return samples


# ==============================================================================
# 3. DravidianLangTech Dataset (Tamil-English, Malayalam-English Nuances)
# ==============================================================================
class DravidianLangTechDataset(BaseCodeMixedDataset):
    """
    PyTorch Dataset for DravidianLangTech shared tasks (Tamil-English, Malayalam-English, Kannada-English).
    Captures agglutinative phonetic transcriptions and regional South Asian sentiment subtleties.
    """
    def __init__(
        self,
        file_path_or_samples: Union[str, List[Dict[str, Any]]],
        tokenizer: Optional[Any] = None,
        preprocessor: Optional[CodeMixedPreprocessor] = None,
        max_length: int = 128
    ):
        if isinstance(file_path_or_samples, str):
            samples = self._load_from_file(file_path_or_samples)
        else:
            samples = file_path_or_samples
        super().__init__(samples, tokenizer, preprocessor, max_length)

    def _load_from_file(self, path: str) -> List[Dict[str, Any]]:
        samples = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"DravidianLangTech file not found: {path}")

        delimiter = "\t" if path.endswith(".tsv") else ","
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                text = row.get("text") or row.get("sentence", "")
                cat = (row.get("category") or row.get("sentiment") or "neutral").lower()
                
                # DravidianLangTech labels: Positive, Negative, Mixed_feelings, Neutral, not-Tamil
                if "pos" in cat:
                    emotion_label = 4  # Joy
                elif "neg" in cat:
                    emotion_label = 1  # Negative
                else:
                    emotion_label = 2  # Neutral

                is_sarc = int(row.get("sarcasm", 1 if ("super presentation" in text.lower() and "ah" in text.lower()) else 0))

                samples.append({
                    "text": text,
                    "emotion_label": emotion_label,
                    "sarcasm_label": is_sarc,
                    "dataset_source": "DravidianLangTech"
                })
        return samples


# ==============================================================================
# Data Collator for PyTorch DataLoader
# ==============================================================================
class CodeMixedDataCollator:
    """Collates individual dataset dictionary items into PyTorch batch tensors."""
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
        attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
        aux_features = torch.stack([item["aux_features"] for item in batch], dim=0)
        emotion_labels = torch.stack([item["emotion_label"] for item in batch], dim=0)
        sarcasm_labels = torch.stack([item["sarcasm_label"] for item in batch], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "aux_features": aux_features,
            "emotion_labels": emotion_labels,
            "sarcasm_labels": sarcasm_labels,
            "raw_texts": [item["raw_text"] for item in batch],
            "normalized_texts": [item["normalized_text"] for item in batch]
        }


# ==============================================================================
# Benchmark Dataset Generator (Immediate Offline Testing & Verification)
# ==============================================================================
def create_synthetic_benchmarks() -> Dict[str, List[Dict[str, Any]]]:
    """
    Provides curated, representative benchmark samples from HASOC, SemEval Sentimix,
    and DravidianLangTech, enabling instant end-to-end execution and offline training verification.
    """
    hasoc_samples = [
        {"text": "Arreeee booooodhiyaaaaa kaam kiya boss!! 😂👏 Bahut kamaal hai!", "emotion_label": 0, "sarcasm_label": 1},
        {"text": "Yeh naya policy report bilkul bekar aur ganda hai", "emotion_label": 3, "sarcasm_label": 0},
        {"text": "Aap tension mat lo bhai kal tak kaam pakka ho jayega", "emotion_label": 2, "sarcasm_label": 0},
        {"text": "Wah bhai kya dimag lagaya hai tune, pura server crash kar diya 🤡👏", "emotion_label": 1, "sarcasm_label": 1},
        {"text": "Bhai tera presentation dekh ke rona aa gaya itna boring tha 😭", "emotion_label": 5, "sarcasm_label": 0},
    ]

    sentimix_samples = [
        {"text": "I really loved this movie yaar! Acting was so super and songs are awesome ❤️🎉", "emotion_label": 4, "sarcasm_label": 0},
        {"text": "Ye customer support itna fast hai ki 3 saal baad reply aaya 😏🙃", "emotion_label": 1, "sarcasm_label": 1},
        {"text": "Aapka code is really clean and performance is great", "emotion_label": 0, "sarcasm_label": 0},
        {"text": "What a lovely surprise, train is delayed by only 8 hours!! 😂🙌", "emotion_label": 1, "sarcasm_label": 1},
        {"text": "Shukriya dost for helping with my project submission", "emotion_label": 4, "sarcasm_label": 0},
    ]

    dravidian_samples = [
        {"text": "Romba nalla velai sir thanks a lot unga support ku 🙏❤️", "emotion_label": 4, "sarcasm_label": 0},
        {"text": "Idhu thaan unga super presentation ah?? Romba kevalama irukku 🙄🤡", "emotion_label": 3, "sarcasm_label": 1},
        {"text": "Hi nan innaiku party ku poren so nalaiku varren", "emotion_label": 2, "sarcasm_label": 0},
        {"text": "Mee office work chala bagundi thank you so much", "emotion_label": 4, "sarcasm_label": 0},
        {"text": "Enna oru acting da, theatre la ellarum thoongitanga 🙃😂", "emotion_label": 1, "sarcasm_label": 1},
    ]

    return {
        "HASOC": hasoc_samples,
        "SemEval_Sentimix": sentimix_samples,
        "DravidianLangTech": dravidian_samples
    }
